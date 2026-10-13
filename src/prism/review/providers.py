"""Backup API providers: run reviews without a self-hosted model.

A user can configure any number of OpenAI-compatible API providers (API keys)
as backup via the PRISM_PROVIDERS env var (JSON array). The router builds a
ChainedBackend that tries each provider in order and fails over to the next
one when a provider errors - so one dead key or outage doesn't kill reviews.

Example:
    PRISM_PROVIDERS='[
      {"name": "agentrouter", "preset": "agentrouter",
       "api_key": "dsk-...", "model": "deepseek/deepseek-v4-flash"},
      {"name": "openai-backup", "preset": "openai",
       "api_key": "sk-...", "model": "gpt-4o-mini", "timeout_s": 60}
    ]'

Each entry: name (label for logs), preset (optional, fills in base_url),
base_url (optional if preset given), api_key (required), model (required),
timeout_s (optional, default 40).

Secrets never appear in logs or /readyz output - only provider names and
models are exposed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.review.backends import APIBackend


PROVIDER_PRESETS: dict[str, str] = {
    # name -> default base_url (OpenAI-compatible /v1)
    "agentrouter": "https://agentrouter-5c61ba90b667.herokuapp.com/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

DEFAULT_TIMEOUT_S = 40.0


@dataclass
class ProviderSpec:
    """One configured backup API provider."""

    name: str
    base_url: str
    api_key: str
    model: str
    timeout_s: float = DEFAULT_TIMEOUT_S
    extra: dict[str, str] = field(default_factory=dict)


def parse_provider_specs(raw: str) -> list[ProviderSpec]:
    """Parse the PRISM_PROVIDERS JSON env value into ProviderSpecs.

    Raises ValueError with a clear message on any config problem.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"PRISM_PROVIDERS is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("PRISM_PROVIDERS must be a JSON array of provider objects")
    specs: list[ProviderSpec] = []
    for i, item in enumerate(data):
        where = f"PRISM_PROVIDERS[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")
        name = str(item.get("name") or f"provider-{i}")
        preset = str(item.get("preset") or "").strip().lower()
        base_url = str(item.get("base_url") or "").strip().rstrip("/")
        if not base_url and preset:
            if preset not in PROVIDER_PRESETS:
                raise ValueError(
                    f"{where}: unknown preset {preset!r}; known presets: {sorted(PROVIDER_PRESETS)}"
                )
            base_url = PROVIDER_PRESETS[preset]
        if not base_url:
            raise ValueError(f"{where} ({name}): need 'base_url' or a known 'preset'")
        api_key = str(item.get("api_key") or "").strip()
        if not api_key:
            raise ValueError(f"{where} ({name}): 'api_key' is required")
        model = str(item.get("model") or "").strip()
        if not model:
            raise ValueError(f"{where} ({name}): 'model' is required")
        try:
            timeout_s = float(item.get("timeout_s", DEFAULT_TIMEOUT_S))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where} ({name}): 'timeout_s' must be a number") from exc
        specs.append(
            ProviderSpec(
                name=name,
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout_s=timeout_s,
            )
        )
    return specs


def provider_specs_from_env(env: dict[str, str] | None = None) -> list[ProviderSpec]:
    """Read PRISM_PROVIDERS from the environment (or a given mapping)."""
    source = env if env is not None else os.environ
    return parse_provider_specs(source.get("PRISM_PROVIDERS", ""))


def describe_specs(specs: list[ProviderSpec]) -> list[dict[str, str]]:
    """JSON-safe summary of providers for /readyz. Never includes secrets."""
    return [{"name": s.name, "base_url": s.base_url, "model": s.model} for s in specs]


def build_provider_backends(specs: list[ProviderSpec]) -> list[APIBackend]:
    """Build one APIBackend per provider spec (lazy import avoids cycles)."""
    from prism.review.backends import APIBackend

    return [APIBackend(s.base_url, s.model, s.api_key, timeout_s=s.timeout_s) for s in specs]
