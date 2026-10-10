"""
Model routing: choose the primary ReviewBackend at startup.

Precedence (first match wins):
  1. PRISM_ADAPTER_DIR set and containing adapter_config.json
     -> VLLMBackend, served via vLLM with --enable-lora.
  2. HF endpoint configured (hf_endpoint_url + hf_token in Settings)
     -> HFEndpointBackend.
  3. Fallback API key present (fallback_api_key in Settings)
     -> APIBackend (generic OpenAI-compatible API).
  4. Otherwise -> StubBackend (dev only).

The choice is logged once at startup and exposed via describe() for /readyz
(never includes secrets). probe_backend() is a tiny transport-level health
check used by readiness.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism.config import Settings, get_settings
from prism.logging import get_logger
from prism.review.backends import (
    APIBackend,
    BackendError,
    HFEndpointBackend,
    ReviewBackend,
    StubBackend,
    VLLMBackend,
)

log = get_logger(__name__)

ADAPTER_CONFIG_FILENAME = "adapter_config.json"


@dataclass
class ProbeResult:
    """Result of probe_backend(): transport-level health check."""

    backend: str
    ok: bool
    latency_s: float
    error: str | None = None


class ModelRouter:
    """
    Select the primary ReviewBackend at startup with clear precedence.

    Args:
        settings: Settings to read endpoint URLs, models, and keys from.
            Defaults to the cached global settings. The adapter route is
            driven by env vars PRISM_ADAPTER_DIR (required) and
            PRISM_ADAPTER_MODEL (optional; defaults to the adapter dir name,
            which is what vLLM --enable-lora serves the adapter as).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._backend, self._route, self._detail = self._select()
        log.info(
            "model_router_selected",
            route=self._route,
            backend=self._backend.name,
            model=getattr(self._backend, "model", self._backend.name),
        )

    def _select(self) -> tuple[ReviewBackend, str, dict[str, str]]:
        settings = self._settings
        adapter_dir = os.environ.get("PRISM_ADAPTER_DIR", "").strip()
        backend: ReviewBackend
        if adapter_dir:
            adapter_path = Path(adapter_dir)
            if (adapter_path / ADAPTER_CONFIG_FILENAME).is_file():
                model = os.environ.get("PRISM_ADAPTER_MODEL", "").strip() or adapter_path.name
                backend = VLLMBackend(
                    settings.vllm_base_url, model, timeout_s=settings.vllm_timeout_s
                )
                detail = {
                    "adapter_dir": str(adapter_path),
                    "base_url": settings.vllm_base_url,
                    "model": model,
                }
                return backend, "adapter", detail
            log.warning(
                "router_adapter_dir_ignored",
                adapter_dir=str(adapter_path),
                reason=f"{ADAPTER_CONFIG_FILENAME} not found",
            )
        if settings.hf_endpoint_url and settings.hf_token:
            backend = HFEndpointBackend(
                settings.hf_endpoint_url, settings.hf_token, timeout_s=settings.hf_timeout_s
            )
            return backend, "hf_endpoint", {"endpoint_url": settings.hf_endpoint_url}
        if settings.fallback_api_key:
            backend = APIBackend(
                settings.fallback_api_base,
                settings.fallback_model,
                settings.fallback_api_key,
                timeout_s=settings.vllm_timeout_s,
            )
            detail = {"base_url": settings.fallback_api_base, "model": settings.fallback_model}
            return backend, "api", detail
        return StubBackend(), "stub", {}

    def primary(self) -> ReviewBackend:
        """The selected primary backend."""
        return self._backend

    @property
    def route(self) -> str:
        """Which precedence rule matched: adapter | hf_endpoint | api | stub."""
        return self._route

    def describe(self) -> dict[str, Any]:
        """JSON-serializable routing info for /readyz. Never contains secrets."""
        return {
            "backend": self._backend.name,
            "model": getattr(self._backend, "model", self._backend.name),
            "route": self._route,
            "detail": self._detail,
        }


async def probe_backend(backend: ReviewBackend, timeout_s: float = 10.0) -> ProbeResult:
    """
    Transport-level health check: send a tiny prompt, measure latency.

    Returns ok=True for any successful complete(); BackendError, timeouts,
    and transport errors return ok=False with the error. Response content
    is not validated — this probes reachability, not review quality.
    """
    start = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s):
            await backend.complete(
                "You are a health-check probe.",
                'Reply with exactly this JSON and nothing else: {"status": "ok"}',
            )
    except (BackendError, TimeoutError, OSError) as exc:
        return ProbeResult(
            backend=backend.name,
            ok=False,
            latency_s=round(time.monotonic() - start, 3),
            error=str(exc)[:200],
        )
    return ProbeResult(backend=backend.name, ok=True, latency_s=round(time.monotonic() - start, 3))
