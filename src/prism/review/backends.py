"""
Review backends: LLM providers behind a common interface.

- ReviewBackend.analyze(system_prompt, user_prompt) -> list[Finding]
- VLLMBackend: OpenAI-compatible chat completions (primary, fine-tuned model),
  requests response_format={"type": "json_object"}.
- HFEndpointBackend: Hugging Face Inference Endpoint (text-generation style).
- APIBackend: generic OpenAI-compatible API — also serves as the STRONGER
  fallback model for the second-pass verification in the confidence gate.

All backends: httpx with explicit timeouts, structured BackendError types,
robust JSON extraction (markdown fences stripped by schemas.parse_findings).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from prism.logging import get_logger
from prism.review.schemas import Finding, parse_findings

log = get_logger(__name__)


class BackendError(RuntimeError):
    """The backend failed (transport, timeout, bad response)."""


class ReviewBackend(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        """Return (raw assistant text, usage dict). Raise BackendError on failure."""

    async def analyze(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[list[Finding], dict[str, Any]]:
        """Return (findings, usage dict). Raise BackendError on failure."""
        text, usage = await self.complete(system_prompt, user_prompt)
        findings = parse_findings(text or "")
        log.info("backend_analyze_ok", backend=self.name, findings=len(findings))
        return findings, usage


def _chat_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class _OpenAICompatibleBackend(ReviewBackend):
    """Shared logic for OpenAI-style /v1/chat/completions endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_s: float,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        """Raw completion for the OpenAI-compatible chat endpoint."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _chat_messages(system_prompt, user_prompt),
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise BackendError(f"{self.name}: request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"{self.name}: malformed response: {exc}") from exc
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        return content or "", {"backend": self.name, "model": self.model, **usage}


class VLLMBackend(_OpenAICompatibleBackend):
    """Primary backend: fine-tuned model served by vLLM (OpenAI-compatible)."""

    name = "vllm"

    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0) -> None:
        super().__init__(base_url, model, api_key=None, timeout_s=timeout_s)


class APIBackend(_OpenAICompatibleBackend):
    """Generic OpenAI-compatible API backend; used for the stronger fallback."""

    name = "api"

    def __init__(self, base_url: str, model: str, api_key: str, timeout_s: float = 120.0) -> None:
        super().__init__(base_url, model, api_key=api_key, timeout_s=timeout_s)


class HFEndpointBackend(ReviewBackend):
    """Hugging Face Inference Endpoint backend."""

    name = "hf_endpoint"

    def __init__(self, endpoint_url: str, token: str, timeout_s: float = 120.0) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        # HF text-generation endpoints take a single prompt string.
        prompt = (
            f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>\n<assistant>\n"
        )
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.1,
                "return_full_text": False,
            },
        }
        try:
            resp = await self._client.post(self.endpoint_url, json=payload)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise BackendError(f"hf_endpoint: request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"hf_endpoint: HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            data = resp.json()
            if isinstance(data, list):
                content = data[0]["generated_text"]
            else:
                content = data.get("generated_text", "")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"hf_endpoint: malformed response: {exc}") from exc
        return content or "", {"backend": self.name}


class StubBackend(ReviewBackend):
    """
    Test/eval backend: replays canned findings. Used by the eval runner
    (--backend stub) and unit tests. Not for production.
    """

    name = "stub"

    def __init__(self, canned: dict[str, list[Finding]] | None = None) -> None:
        # canned: path -> findings to return for that file
        self.canned = canned or {}

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        import json as _json

        findings = self.canned.get("__all__", [])
        payload = {
            "findings": [
                {
                    "path": f.path,
                    "line": f.line,
                    "severity": f.severity.value,
                    "category": f.category,
                    "title": f.title,
                    "explanation": f.explanation,
                    "suggestion": f.suggestion,
                    "confidence": f.confidence,
                    "language_hint": f.language_hint,
                }
                for f in findings
            ]
        }
        return _json.dumps(payload), {"backend": self.name, "stub": True}

    async def aclose(self) -> None:  # noqa: D102
        return None


def build_backend(settings: Any, *, for_verification: bool = False) -> ReviewBackend:
    """
    Factory. `for_verification=True` returns the stronger fallback backend
    (APIBackend) used for the confidence-gate second pass.
    """
    if for_verification or settings.review_backend == "api":
        return APIBackend(
            settings.fallback_api_base,
            settings.fallback_model,
            settings.fallback_api_key,
            timeout_s=settings.vllm_timeout_s,
        )
    if settings.review_backend == "hf_endpoint":
        return HFEndpointBackend(
            settings.hf_endpoint_url, settings.hf_token, timeout_s=settings.hf_timeout_s
        )
    if settings.review_backend == "stub":
        return StubBackend()
    return VLLMBackend(
        settings.vllm_base_url, settings.vllm_model, timeout_s=settings.vllm_timeout_s
    )
