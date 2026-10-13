"""Reliability + observability tests: webhook error classification, inline-review
timeout, Sentry gating, and the /readyz provider probe."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient

import prism.main as main_module
from prism.config import Settings
from prism.github.client import GitHubAPIError
from prism.logging import setup_logging
from prism.review.backends import BackendError

SECRET = "test-webhook-secret-for-reliability"
BODY = json.dumps(
    {
        "action": "opened",
        "installation": {"id": 424242},
        "repository": {"name": "myrepo", "owner": {"login": "myorg"}},
        "pull_request": {"number": 7},
    }
).encode()


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _headers(body: bytes) -> dict[str, str]:
    return {
        "X-Hub-Signature-256": _sign(body),
        "X-GitHub-Event": "pull_request",
    }


@pytest.fixture(autouse=True)
def _restore_logging() -> Any:
    """Some tests reconfigure structlog; restore the default JSON config after."""
    yield
    setup_logging(force=True)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "github_app_id": 123456,
        "github_webhook_secret": SECRET,
        "github_private_key_path": "",
        "redis_url": None,  # inline mode: review awaited in the request
        "review_backend": "stub",
        "rate_limit_webhook_per_min": 30,
    }
    base.update(overrides)
    return Settings(**base)


def _settings_with_dsn(dsn: str) -> Settings:
    # config.py owns the sentry_dsn field; set it via __dict__ so this works
    # whether or not the field is already declared there (extra="ignore"
    # drops unknown constructor kwargs until then).
    settings = _settings()
    settings.__dict__["sentry_dsn"] = dsn
    return settings


def _client_with_review_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None
) -> TestClient:
    """TestClient whose review_pr raises `error` (None = succeeds)."""
    settings = _settings()
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    async def _fake_review_pr(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if error is not None:
            raise error
        return {"status": "ok"}

    monkeypatch.setattr(main_module, "review_pr", _fake_review_pr)
    main_module._buckets.clear()
    return TestClient(main_module.create_app())


def _post_webhook(client: TestClient) -> Any:
    return client.post("/webhooks/github", content=BODY, headers=_headers(BODY))


# ---------------------------------------------------------------------------
# Item 1: webhook error classification
# ---------------------------------------------------------------------------


def test_transient_backend_error_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_review_error(monkeypatch, BackendError("provider blew up"))
    resp = _post_webhook(client)
    assert resp.status_code == 500
    assert resp.json()["error_type"] == "BackendError"


def test_github_api_error_422_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_review_error(monkeypatch, GitHubAPIError("unprocessable", status=422))
    resp = _post_webhook(client)
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_github_errors_return_200(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    client = _client_with_review_error(monkeypatch, GitHubAPIError("nope", status=status))
    resp = _post_webhook(client)
    assert resp.status_code == 200


@pytest.mark.parametrize("status", [None, 429, 500, 502, 503])
def test_transient_github_errors_return_500(
    monkeypatch: pytest.MonkeyPatch, status: int | None
) -> None:
    client = _client_with_review_error(monkeypatch, GitHubAPIError("retry me", status=status))
    resp = _post_webhook(client)
    assert resp.status_code == 500


def test_timeout_error_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_review_error(monkeypatch, TimeoutError())
    resp = _post_webhook(client)
    assert resp.status_code == 500
    assert resp.json()["error_type"] == "TimeoutError"


def test_unclassified_error_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_review_error(monkeypatch, ValueError("something weird"))
    resp = _post_webhook(client)
    assert resp.status_code == 500


def test_classify_keeps_structured_log_line(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    setup_logging(force=True)
    client = _client_with_review_error(monkeypatch, BackendError("provider blew up"))
    resp = _post_webhook(client)
    assert resp.status_code == 500
    out = capsys.readouterr().out
    assert "inline_review_failed" in out
    assert "BackendError" in out


# ---------------------------------------------------------------------------
# Item 2: inline review bounded by asyncio.wait_for
# ---------------------------------------------------------------------------


def test_inline_review_timeout_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """A review_pr that sleeps must trip the wait_for timeout (-> 500)."""
    settings = _settings()
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    async def _slow_review_pr(*args: Any, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(1.0)  # longer than the patched wait_for budget
        return {"status": "ok"}

    monkeypatch.setattr(main_module, "review_pr", _slow_review_pr)
    main_module._buckets.clear()

    real_wait_for: Callable[..., Any] = asyncio.wait_for

    async def _fast_wait_for(coro: Any, timeout: float) -> Any:
        del timeout  # the point of the test: force a small budget
        return await real_wait_for(coro, timeout=0.05)

    monkeypatch.setattr(asyncio, "wait_for", _fast_wait_for)
    client = TestClient(main_module.create_app())
    resp = _post_webhook(client)
    assert resp.status_code == 500
    assert resp.json()["error_type"] == "TimeoutError"


def test_inline_review_timeout_is_fifty_seconds() -> None:
    assert main_module.INLINE_REVIEW_TIMEOUT_S == 50.0


# ---------------------------------------------------------------------------
# Item 11: sentry gating
# ---------------------------------------------------------------------------


def test_sentry_not_initialized_without_dsn() -> None:
    assert main_module._init_sentry(_settings()) is False


def test_sentry_initialized_with_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    fake = ModuleType("sentry_sdk")

    def _fake_init(**kwargs: Any) -> None:
        calls.update(kwargs)

    fake.init = _fake_init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)

    dsn = "https://user@example.ingest.sentry.io/123"
    assert main_module._init_sentry(_settings_with_dsn(dsn)) is True
    assert calls["dsn"] == dsn
    assert calls["send_default_pii"] is False
    assert calls["traces_sample_rate"] == 0.0


def test_sentry_missing_package_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    setup_logging(force=True)
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)  # import -> ImportError
    assert (
        main_module._init_sentry(_settings_with_dsn("https://user@example.ingest.sentry.io/123"))
        is False
    )
    assert "sentry_sdk_not_installed" in capsys.readouterr().out


def test_sentry_dsn_never_logged(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    setup_logging(force=True)
    dsn = "https://sentry-secret-token-xyz@o1.ingest.sentry.io/4242"
    fake = ModuleType("sentry_sdk")
    fake.init = lambda **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    assert main_module._init_sentry(_settings_with_dsn(dsn)) is True
    out = capsys.readouterr().out
    assert dsn not in out
    assert "sentry-secret-token-xyz" not in out


def test_create_app_survives_missing_sentry_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    monkeypatch.setattr(main_module, "get_settings", lambda: _settings_with_dsn("https://x@y/1"))
    app = main_module.create_app()  # must not raise
    assert app.title == "PRism"


# ---------------------------------------------------------------------------
# Item 11: /readyz provider probe
# ---------------------------------------------------------------------------


def _provider_env(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    monkeypatch.setenv(
        "PRISM_PROVIDERS",
        json.dumps(
            [
                {
                    "name": "p1",
                    "base_url": base_url,
                    "api_key": "sk-probe-secret-key",
                    "model": "model-1",
                }
            ]
        ),
    )


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeAsyncClient:
    """Minimal stand-in for httpx.AsyncClient recording its calls."""

    last_url: str = ""
    last_headers: dict[str, str] = {}
    status_code: int = 200
    failure: Exception | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        type(self).last_url = url
        type(self).last_headers = dict(headers or {})
        if self.failure is not None:
            raise self.failure
        return _FakeResponse(self.status_code)


def _probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(main_module.httpx, "AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.last_url = ""
    _FakeAsyncClient.last_headers = {}
    _FakeAsyncClient.status_code = 200
    _FakeAsyncClient.failure = None
    return asyncio.run(main_module._check_backend(_settings(review_backend="api")))


def test_api_probe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _provider_env(monkeypatch, "https://api.example.com/v1")
    result = _probe(monkeypatch)
    assert result["ok"] is True
    assert result["probed"] is True
    assert result["backend"] == "api"
    assert _FakeAsyncClient.last_url == "https://api.example.com/v1/models"
    assert _FakeAsyncClient.last_headers.get("Authorization") == "Bearer sk-probe-secret-key"
    # providers list is kept; API key never appears in it
    assert result["providers"] == [
        {"name": "p1", "base_url": "https://api.example.com/v1", "model": "model-1"}
    ]
    assert "sk-probe-secret-key" not in json.dumps(result)


def test_api_probe_normalizes_bare_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _provider_env(monkeypatch, "https://api.example.com")
    result = _probe(monkeypatch)
    assert result["ok"] is True
    assert _FakeAsyncClient.last_url == "https://api.example.com/v1/models"


def test_api_probe_bad_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _provider_env(monkeypatch, "https://api.example.com/v1")
    _FakeAsyncClient.status_code = 401
    monkeypatch.setattr(main_module.httpx, "AsyncClient", _FakeAsyncClient)
    result = asyncio.run(main_module._check_backend(_settings(review_backend="api")))
    assert result["ok"] is False
    assert result["probed"] is True
    assert "401" in result["detail"]
    assert "sk-probe-secret-key" not in json.dumps(result)


def test_api_probe_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _provider_env(monkeypatch, "https://api.example.com/v1")
    import httpx

    _FakeAsyncClient.failure = httpx.ConnectError("connection refused")
    monkeypatch.setattr(main_module.httpx, "AsyncClient", _FakeAsyncClient)
    result = asyncio.run(main_module._check_backend(_settings(review_backend="api")))
    assert result["ok"] is False
    assert result["probed"] is True
    assert "ConnectError" in result["detail"]
    assert "sk-probe-secret-key" not in json.dumps(result)


def test_api_probe_no_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRISM_PROVIDERS", raising=False)
    result = asyncio.run(main_module._check_backend(_settings(review_backend="api")))
    assert result["ok"] is True
    assert result["probed"] is False
    assert result["providers"] == []


def test_api_probe_invalid_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRISM_PROVIDERS", "not-json")
    result = asyncio.run(main_module._check_backend(_settings(review_backend="api")))
    assert result["ok"] is False
    assert result["probed"] is False
