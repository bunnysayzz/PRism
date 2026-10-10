"""Tests for ConstrainedReviewBackend, ModelRouter, and probe_backend."""

import json
from typing import Any

import pytest

from prism.config import Settings
from prism.review.backends import APIBackend, BackendError, ReviewBackend, StubBackend
from prism.review.model_router import ModelRouter, ProbeResult, probe_backend
from prism.review.structured import ConstrainedReviewBackend

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@pytest.fixture(autouse=True)
def _scrub_proxy_env(monkeypatch):
    """httpx reads proxy env at client construction; the sandbox's are malformed."""
    for var in _PROXY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _finding_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": "a.py",
        "line": 10,
        "severity": "high",
        "category": "security",
        "title": "SQL injection via f-string",
        "explanation": (
            "User input is interpolated directly into the SQL query string, "
            "allowing arbitrary SQL execution."
        ),
        "suggestion": "use parameterized queries",
        "confidence": 0.9,
    }
    payload.update(overrides)
    return payload


def _review_json(*items: dict[str, Any]) -> str:
    return json.dumps({"findings": list(items)})


class FlakyBackend(ReviewBackend):
    """Replays canned responses in order; records every call."""

    name = "flaky"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        self.calls.append((system_prompt, user_prompt))
        text = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        return text, {"backend": self.name, "prompt_tokens": 100, "completion_tokens": 50}


class SamplingBackend(ReviewBackend):
    """Accepts temperature/top_p kwargs and records them."""

    name = "sampling"

    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self.seen = {"temperature": temperature, "top_p": top_p}
        return _review_json(_finding_payload()), {"backend": self.name}


# ---------------------------------------------------------------------------
# ConstrainedReviewBackend
# ---------------------------------------------------------------------------


async def test_schema_guidance_injected_into_system_prompt():
    inner = FlakyBackend([_review_json(_finding_payload())])
    backend = ConstrainedReviewBackend(inner)
    await backend.analyze("You are a reviewer.", "diff here")
    system_sent = inner.calls[0][0]
    assert "STRICT OUTPUT SCHEMA" in system_sent
    assert '"findings"' in system_sent


async def test_prose_wrapped_json_parsed_without_retry():
    raw = (
        "Here is my review:\n```json\n"
        + _review_json(_finding_payload())
        + "\n```\nHope this helps."
    )
    inner = FlakyBackend([raw])
    backend = ConstrainedReviewBackend(inner)
    findings, _ = await backend.analyze("system", "user")
    assert len(findings) == 1
    assert findings[0].title == "SQL injection via f-string"
    assert len(inner.calls) == 1  # no retry needed


async def test_retry_recovers_from_malformed_json():
    inner = FlakyBackend(["not json at all {{{", _review_json(_finding_payload())])
    backend = ConstrainedReviewBackend(inner, max_retries=2)
    findings, _ = await backend.analyze("system", "user")
    assert len(findings) == 1
    assert findings[0].title == "SQL injection via f-string"
    assert len(inner.calls) == 2  # initial + 1 repair retry
    repair_prompt = inner.calls[1][1]
    assert "not valid" in repair_prompt.lower()
    assert "validation errors" in repair_prompt.lower()


async def test_partially_invalid_entries_trigger_repair():
    invalid = {
        "path": "a.py",
        "line": 3,
        "severity": "low",
        "category": "style",
        "explanation": "long enough explanation text here",
        "confidence": 0.5,
        # missing "title" -> invalid
    }
    first = json.dumps({"findings": [_finding_payload(), invalid]})
    second = _review_json(_finding_payload(), _finding_payload(title="Another real issue"))
    inner = FlakyBackend([first, second])
    backend = ConstrainedReviewBackend(inner, max_retries=2)
    findings, _ = await backend.analyze("system", "user")
    assert len(inner.calls) == 2
    assert len(findings) == 2  # second (fully valid) response wins


async def test_empty_findings_not_retried_by_default():
    inner = FlakyBackend(['{"findings": []}'])
    backend = ConstrainedReviewBackend(inner, max_retries=2)
    findings, _ = await backend.analyze("system", "user")
    assert findings == []
    assert len(inner.calls) == 1


async def test_retry_on_empty_when_findings_expected():
    inner = FlakyBackend(['{"findings": []}', _review_json(_finding_payload())])
    backend = ConstrainedReviewBackend(inner, max_retries=2, retry_on_empty=True)
    findings, _ = await backend.analyze("system", "user")
    assert len(findings) == 1
    assert len(inner.calls) == 2


async def test_retries_exhausted_returns_best_effort():
    inner = FlakyBackend(["garbage"] * 5)
    backend = ConstrainedReviewBackend(inner, max_retries=2, strict=False)
    findings, _ = await backend.analyze("system", "user")
    assert findings == []
    assert len(inner.calls) == 3  # 1 initial + 2 retries, then give up


async def test_retries_exhausted_strict_raises():
    inner = FlakyBackend(["garbage"])
    backend = ConstrainedReviewBackend(inner, max_retries=1, strict=True)
    with pytest.raises(BackendError, match="schema validation failed"):
        await backend.analyze("system", "user")
    assert len(inner.calls) == 2


async def test_confidence_clamped_and_quantized():
    raw = _review_json(
        _finding_payload(confidence=1.5),
        _finding_payload(confidence=0.33333, title="Second issue here"),
    )
    inner = FlakyBackend([raw])
    backend = ConstrainedReviewBackend(inner)
    findings, _ = await backend.analyze("system", "user")
    assert [f.confidence for f in findings] == [1.0, 0.33]


async def test_cost_accounting_math():
    inner = FlakyBackend([_review_json(_finding_payload())])
    backend = ConstrainedReviewBackend(inner, price_table={"flaky": (0.01, 0.03)})
    _, usage = await backend.analyze("system", "user")
    # 100 prompt tokens @ $0.01/1k = $0.001; 50 completion @ $0.03/1k = $0.0015
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 50
    assert usage["estimated_cost_usd"] == pytest.approx(0.0025)
    assert usage["cost_model"] == "flaky"


async def test_cost_without_token_counts_is_zero():
    class NoUsageBackend(ReviewBackend):
        name = "nouss"

        async def complete(self, system_prompt, user_prompt):  # type: ignore[no-untyped-def]
            return _review_json(_finding_payload()), {"backend": self.name}

    backend = ConstrainedReviewBackend(NoUsageBackend(), price_table={"nouss": (1.0, 2.0)})
    _, usage = await backend.analyze("system", "user")
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["estimated_cost_usd"] == 0.0


async def test_cost_skipped_without_price_table():
    inner = FlakyBackend([_review_json(_finding_payload())])
    backend = ConstrainedReviewBackend(inner)
    _, usage = await backend.analyze("system", "user")
    assert "estimated_cost_usd" not in usage


async def test_temperature_plumbed_when_supported():
    inner = SamplingBackend()
    backend = ConstrainedReviewBackend(inner, temperature=0.7, top_p=0.9)
    findings, _ = await backend.analyze("system", "user")
    assert len(findings) == 1
    assert inner.seen == {"temperature": 0.7, "top_p": 0.9}


async def test_temperature_recorded_when_unsupported():
    inner = FlakyBackend([_review_json(_finding_payload())])
    backend = ConstrainedReviewBackend(inner, temperature=0.7)
    findings, usage = await backend.analyze("system", "user")
    assert len(findings) == 1
    assert usage["sampling"] == {"temperature": 0.7}


def test_invalid_sampling_params_rejected():
    with pytest.raises(ValueError, match="temperature"):
        ConstrainedReviewBackend(StubBackend(), temperature=5.0)
    with pytest.raises(ValueError, match="top_p"):
        ConstrainedReviewBackend(StubBackend(), top_p=1.5)


def test_wrapper_name_and_model_delegate():
    backend = ConstrainedReviewBackend(FlakyBackend(["{}"]))
    assert backend.name == "constrained:flaky"
    assert backend.model == "flaky"


def test_json_mode_detection():
    api = APIBackend("https://api.example/v1", "gpt-x", "k")
    assert ConstrainedReviewBackend(api)._json_mode() == "json_object"
    assert ConstrainedReviewBackend(StubBackend())._json_mode() == "prompt_only"


async def test_prompt_only_mode_demands_bare_json():
    inner = FlakyBackend([_review_json(_finding_payload())])
    backend = ConstrainedReviewBackend(inner)
    assert backend._json_mode() == "prompt_only"
    await backend.analyze("system", "user")
    assert "ONLY the JSON object" in inner.calls[0][0]


async def test_aclose_delegates():
    closed: list[bool] = []

    class CloseableBackend(ReviewBackend):
        name = "closeable"

        async def complete(self, system_prompt, user_prompt):  # type: ignore[no-untyped-def]
            return "{}", {}

        async def aclose(self) -> None:
            closed.append(True)

    await ConstrainedReviewBackend(CloseableBackend()).aclose()
    assert closed == [True]


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    kwargs: dict[str, Any] = {
        "hf_endpoint_url": "",
        "hf_token": "",
        "fallback_api_key": "",
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def test_router_prefers_adapter_when_config_present(tmp_path, monkeypatch):
    adapter = tmp_path / "my-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "x"}')
    monkeypatch.setenv("PRISM_ADAPTER_DIR", str(adapter))
    router = ModelRouter(_settings())
    assert router.route == "adapter"
    assert router.primary().name == "vllm"
    assert router.primary().model == "my-adapter"  # dir basename default
    assert json.dumps(router.describe())  # JSON-serializable


def test_router_adapter_model_override(tmp_path, monkeypatch):
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    monkeypatch.setenv("PRISM_ADAPTER_DIR", str(adapter))
    monkeypatch.setenv("PRISM_ADAPTER_MODEL", "custom-adapter-name")
    router = ModelRouter(_settings())
    assert router.route == "adapter"
    assert router.primary().model == "custom-adapter-name"


def test_router_ignores_adapter_dir_without_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_ADAPTER_DIR", str(tmp_path))  # no adapter_config.json
    router = ModelRouter(_settings(fallback_api_key="sk-test"))
    assert router.route == "api"
    assert router.primary().name == "api"


def test_router_precedence_adapter_over_hf_over_api(tmp_path, monkeypatch):
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    monkeypatch.setenv("PRISM_ADAPTER_DIR", str(adapter))
    full = _settings(
        hf_endpoint_url="https://endpoint.example", hf_token="tok", fallback_api_key="k"
    )
    assert ModelRouter(full).route == "adapter"
    monkeypatch.delenv("PRISM_ADAPTER_DIR")
    assert ModelRouter(full).route == "hf_endpoint"
    assert ModelRouter(_settings(fallback_api_key="k")).route == "api"


def test_router_hf_endpoint(monkeypatch):
    monkeypatch.delenv("PRISM_ADAPTER_DIR", raising=False)
    router = ModelRouter(_settings(hf_endpoint_url="https://endpoint.example", hf_token="tok"))
    assert router.route == "hf_endpoint"
    assert router.primary().name == "hf_endpoint"


def test_router_api_fallback(monkeypatch):
    monkeypatch.delenv("PRISM_ADAPTER_DIR", raising=False)
    router = ModelRouter(
        _settings(
            fallback_api_key="sk-test",
            fallback_api_base="https://api.example/v1",
            fallback_model="gpt-x",
        )
    )
    assert router.route == "api"
    assert router.primary().name == "api"


def test_router_stub_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("PRISM_ADAPTER_DIR", raising=False)
    router = ModelRouter(_settings())
    assert router.route == "stub"
    assert isinstance(router.primary(), StubBackend)


def test_router_describe_has_no_secrets(monkeypatch):
    monkeypatch.delenv("PRISM_ADAPTER_DIR", raising=False)
    router = ModelRouter(_settings(fallback_api_key="sk-secret-test"))
    dumped = json.dumps(router.describe())
    assert "sk-secret-test" not in dumped
    assert router.describe()["route"] == "api"


# ---------------------------------------------------------------------------
# probe_backend
# ---------------------------------------------------------------------------


async def test_probe_ok():
    result = await probe_backend(StubBackend())
    assert isinstance(result, ProbeResult)
    assert result.ok is True
    assert result.latency_s >= 0.0
    assert result.backend == "stub"
    assert result.error is None


async def test_probe_failure():
    class BoomBackend(ReviewBackend):
        name = "boom"

        async def complete(self, system_prompt, user_prompt):  # type: ignore[no-untyped-def]
            raise BackendError("connection refused")

    result = await probe_backend(BoomBackend(), timeout_s=5.0)
    assert result.ok is False
    assert "connection refused" in (result.error or "")
    assert result.latency_s >= 0.0
