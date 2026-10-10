"""Tests for backup API providers: parsing, presets, chain failover, routing."""

import json

import pytest

from prism.config import Settings
from prism.review import providers
from prism.review.backends import BackendError, ChainedBackend, ReviewBackend
from prism.review.model_router import ModelRouter

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


def _env(*items: dict) -> dict[str, str]:
    return {"PRISM_PROVIDERS": json.dumps(list(items))}


def test_parse_empty():
    assert providers.parse_provider_specs("") == []
    assert providers.parse_provider_specs("  ") == []


def test_parse_preset_resolves_base_url():
    specs = providers.parse_provider_specs(
        json.dumps(
            [
                {
                    "name": "ar",
                    "preset": "agentrouter",
                    "api_key": "dsk-test",
                    "model": "deepseek/deepseek-v4-flash",
                },
            ]
        )
    )
    assert len(specs) == 1
    s = specs[0]
    assert s.base_url == "https://agentrouter-5c61ba90b667.herokuapp.com/v1"
    assert s.model == "deepseek/deepseek-v4-flash"
    assert s.timeout_s == 120.0


def test_parse_explicit_base_url_and_timeout():
    specs = providers.parse_provider_specs(
        json.dumps(
            [
                {
                    "name": "custom",
                    "base_url": "https://llm.example.com/v1/",
                    "api_key": "k",
                    "model": "m",
                    "timeout_s": 60,
                },
            ]
        )
    )
    assert specs[0].base_url == "https://llm.example.com/v1"  # trailing slash stripped
    assert specs[0].timeout_s == 60.0


def test_parse_missing_api_key():
    with pytest.raises(ValueError, match="api_key"):
        providers.parse_provider_specs(
            json.dumps([{"name": "x", "model": "m", "preset": "openai"}])
        )


def test_parse_missing_model():
    with pytest.raises(ValueError, match="model"):
        providers.parse_provider_specs(
            json.dumps([{"name": "x", "preset": "openai", "api_key": "k"}])
        )


def test_parse_unknown_preset():
    with pytest.raises(ValueError, match="unknown preset"):
        providers.parse_provider_specs(
            json.dumps([{"name": "x", "preset": "nope", "api_key": "k", "model": "m"}])
        )


def test_parse_bad_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        providers.parse_provider_specs("{oops")


def test_describe_specs_has_no_secrets():
    specs = providers.parse_provider_specs(
        json.dumps(
            [
                {
                    "name": "ar",
                    "preset": "agentrouter",
                    "api_key": "dsk-SUPERSECRET",
                    "model": "deepseek/deepseek-v4-flash",
                },
            ]
        )
    )
    desc = providers.describe_specs(specs)
    assert json.dumps(desc).count("dsk-SUPERSECRET") == 0
    assert desc[0]["model"] == "deepseek/deepseek-v4-flash"


class _FailBackend(ReviewBackend):
    name = "fail"

    async def complete(self, system_prompt, user_prompt):
        raise BackendError("boom")

    async def aclose(self):
        return None


class _OkBackend(ReviewBackend):
    name = "ok"

    def __init__(self, model="m"):
        self.model = model

    async def complete(self, system_prompt, user_prompt):
        return '{"ok": true}', {"backend": self.name}

    async def aclose(self):
        return None


async def test_chain_fails_over_to_next():
    chain = ChainedBackend([_FailBackend(), _OkBackend()])
    text, meta = await chain.complete("sys", "user")
    assert text == '{"ok": true}'
    assert meta["chain_used"] == "ok"


async def test_chain_raises_when_all_fail():
    chain = ChainedBackend([_FailBackend(), _FailBackend()])
    with pytest.raises(BackendError, match="all providers in chain failed"):
        await chain.complete("sys", "user")


def test_chain_needs_backends():
    with pytest.raises(ValueError):
        ChainedBackend([])


def _settings_no_backends(**overrides):
    base = {
        "fallback_api_key": "",
        "hf_endpoint_url": "",
        "hf_token": "",
        "review_backend": "vllm",
    }
    base.update(overrides)
    return Settings(**base)


def test_router_selects_providers_chain(monkeypatch):
    monkeypatch.setenv(
        "PRISM_PROVIDERS",
        json.dumps(
            [
                {
                    "name": "ar",
                    "preset": "agentrouter",
                    "api_key": "dsk-test",
                    "model": "deepseek/deepseek-v4-flash",
                },
                {"name": "oai", "preset": "openai", "api_key": "sk-test", "model": "gpt-4o-mini"},
            ]
        ),
    )
    monkeypatch.delenv("PRISM_ADAPTER_DIR", raising=False)
    router = ModelRouter(_settings_no_backends())
    assert router.route == "providers"
    assert isinstance(router.primary(), ChainedBackend)
    desc = router.describe()
    assert json.dumps(desc).count("dsk-test") == 0  # no secrets in describe
    assert desc["detail"]["providers"][0]["name"] == "ar"


def test_router_falls_back_to_single_api_without_providers(monkeypatch):
    monkeypatch.delenv("PRISM_PROVIDERS", raising=False)
    monkeypatch.delenv("PRISM_ADAPTER_DIR", raising=False)
    router = ModelRouter(_settings_no_backends(fallback_api_key="sk-x"))
    assert router.route == "api"


def test_router_stub_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("PRISM_PROVIDERS", raising=False)
    monkeypatch.delenv("PRISM_ADAPTER_DIR", raising=False)
    router = ModelRouter(_settings_no_backends())
    assert router.route == "stub"


def test_base_url_v1_normalized_once():
    """Both 'host' and 'host/v1' conventions must yield one '/v1' in the URL."""
    from prism.review.backends import APIBackend

    for base in ("https://example.com", "https://example.com/v1", "https://example.com/v1/"):
        b = APIBackend(base, "m", "k", timeout_s=5)
        assert b.base_url == "https://example.com", base
