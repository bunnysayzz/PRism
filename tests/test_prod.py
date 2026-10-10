"""Production-hardening tests: health/readiness, rate limiting, request IDs,
secret redaction, and config validation."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import prism.main as main_module
from prism.config import Settings, safe_dump
from prism.logging import get_logger, setup_logging

SECRET = "test-webhook-secret-for-prod"
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


def _headers(body: bytes, secret: str = SECRET) -> dict[str, str]:
    return {
        "X-Hub-Signature-256": _sign(body, secret),
        "X-GitHub-Event": "pull_request",
    }


def _settings(**overrides: Any) -> Settings:
    """Settings with sane prod-test defaults; no env leakage."""
    base: dict[str, Any] = {
        "github_app_id": 123456,
        "github_webhook_secret": SECRET,
        "github_private_key_path": "",
        "redis_url": None,
        "review_backend": "stub",
        "rate_limit_webhook_per_min": 30,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _restore_logging() -> Any:
    """Some tests reconfigure structlog; restore the default JSON config after."""
    yield
    setup_logging(force=True)


@pytest.fixture()
def app_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Patch prism.main.get_settings and stub out the review job itself."""
    settings = _settings()
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    async def _fake_review_pr(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "ok"}

    monkeypatch.setattr(main_module, "review_pr", _fake_review_pr)
    main_module._buckets.clear()
    return settings


@pytest.fixture()
def client(app_settings: Settings) -> TestClient:
    return TestClient(main_module.create_app())


# ---------------------------------------------------------------------------
# /healthz + request-id middleware
# ---------------------------------------------------------------------------


def test_healthz_ok(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert resp.headers.get("X-Request-ID")  # generated when absent


def test_request_id_echoed(client: TestClient) -> None:
    resp = client.get("/healthz", headers={"X-Request-ID": "req-123"})
    assert resp.headers["X-Request-ID"] == "req-123"


def test_request_id_present_on_webhook(client: TestClient) -> None:
    resp = client.post("/webhooks/github", content=BODY, headers=_headers(BODY))
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID")


# ---------------------------------------------------------------------------
# /readyz
# ---------------------------------------------------------------------------


def test_readyz_ready_stub_backend(
    client: TestClient, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path / "key.pem"
    key.write_text("fake-pem")
    settings = _settings(github_private_key_path=str(key))
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    resp = client.get("/readyz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["checks"]["settings"]["ok"] is True
    assert data["checks"]["backend"]["ok"] is True
    assert "redis" not in data["checks"]  # unset => not checked


def test_readyz_not_ready_missing_settings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main_module, "get_settings", lambda: _settings(github_app_id=0, github_webhook_secret="")
    )
    resp = client.get("/readyz")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert data["checks"]["settings"]["ok"] is False
    assert data["checks"]["settings"]["problems"]  # details, no secret values
    joined = json.dumps(data["checks"])
    assert SECRET not in joined


def test_readyz_backend_probe_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(review_backend="vllm", vllm_base_url="http://127.0.0.1:1")
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    check = resp.json()["checks"]["backend"]
    assert check["ok"] is False
    assert check["probed"] is True


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_allows_under_limit(client: TestClient) -> None:
    for _ in range(3):
        resp = client.post("/webhooks/github", content=BODY, headers=_headers(BODY))
        assert resp.status_code == 200
        assert resp.json() == {"status": "enqueued"}


def test_rate_limit_blocks_over_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(rate_limit_webhook_per_min=2)
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    main_module._buckets.clear()

    codes = [
        client.post("/webhooks/github", content=BODY, headers=_headers(BODY)).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]


def test_rate_limit_429_has_retry_after(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(rate_limit_webhook_per_min=1)
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    main_module._buckets.clear()

    client.post("/webhooks/github", content=BODY, headers=_headers(BODY))
    resp = client.post("/webhooks/github", content=BODY, headers=_headers(BODY))
    assert resp.status_code == 429
    retry_after = resp.headers.get("Retry-After")
    assert retry_after is not None
    assert int(retry_after) >= 1
    assert resp.json() == {"error": "rate limit exceeded"}


def test_rate_limit_is_per_installation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One noisy installation must not starve another."""
    settings = _settings(rate_limit_webhook_per_min=1)
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    main_module._buckets.clear()

    other = BODY.replace(b"424242", b"999999")
    first = client.post("/webhooks/github", content=BODY, headers=_headers(BODY))
    other_ok = client.post("/webhooks/github", content=other, headers=_headers(other))
    limited = client.post("/webhooks/github", content=BODY, headers=_headers(BODY))
    assert (first.status_code, other_ok.status_code, limited.status_code) == (200, 200, 429)


# ---------------------------------------------------------------------------
# Secret safety
# ---------------------------------------------------------------------------


def test_bad_signature_never_leaks_secret(client: TestClient, capsys: Any) -> None:
    setup_logging(force=True)  # rebind stdout capture for this test
    resp = client.post("/webhooks/github", content=BODY, headers=_headers(BODY, "wrong"))
    assert resp.status_code == 401
    assert resp.json() == {"error": "invalid signature"}
    assert SECRET not in resp.text
    out = capsys.readouterr().out
    assert SECRET not in out
    assert "webhook_bad_signature" in out


def test_settings_redacted_masks_secrets() -> None:
    settings = _settings(fallback_api_key="sk-live-123", hf_token="hf_abc")
    dump = settings.redacted()
    blob = json.dumps(dump)
    assert "sk-live-123" not in blob
    assert "hf_abc" not in blob
    assert SECRET not in blob
    assert dump["github_webhook_secret"] == "***REDACTED***"
    assert dump["fallback_api_key"] == "***REDACTED***"
    assert dump["hf_token"] == "***REDACTED***"
    # Non-secret values survive untouched.
    assert dump["github_app_id"] == 123456
    assert dump["rate_limit_webhook_per_min"] == 30


def test_safe_dump_uses_redacted() -> None:
    settings = _settings()
    assert safe_dump(settings) == settings.redacted()


def test_log_redaction_masks_secret_kwargs(capsys: Any) -> None:
    setup_logging(force=True, log_format="json")
    log = get_logger("test-redact")
    log.info(
        "probe",
        api_key="sk-live-123",
        nested={"token": "tok-xyz", "safe": "visible"},
        items=[{"password": "p@ss"}],
        plain="hello",
    )
    out = capsys.readouterr().out
    assert "sk-live-123" not in out
    assert "tok-xyz" not in out
    assert "p@ss" not in out
    assert out.count("***REDACTED***") >= 3
    assert "visible" in out
    assert "hello" in out


def test_log_console_format_renders(capsys: Any) -> None:
    setup_logging(force=True, log_format="console")
    get_logger("test-console").info("hello console", api_key="sk-live-123")
    out = capsys.readouterr().out
    assert "hello console" in out
    assert "sk-live-123" not in out


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_confidence_verify_above_auto_post_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(confidence_verify=0.9, confidence_auto_post=0.8)


def test_confidence_verify_equal_to_auto_post_allowed() -> None:
    settings = _settings(confidence_verify=0.8, confidence_auto_post=0.8)
    assert settings.confidence_verify == 0.8


def test_new_settings_defaults_safe() -> None:
    settings = _settings()
    assert settings.log_format == "json"
    assert settings.rate_limit_webhook_per_min == 30
    assert settings.prism_adapter_dir is None
    assert settings.mlflow_tracking_uri is None
    assert settings.structured_output_max_retries == 2
    assert settings.eval_min_f1 == 0.70
