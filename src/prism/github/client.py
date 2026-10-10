"""
Low-level GitHub API client: httpx AsyncClient with auth headers, retries,
exponential backoff, and rate-limit respect.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from prism.config import Settings
from prism.logging import get_logger

log = get_logger(__name__)

API_BASE = "https://api.github.com"
API_VERSION = "2022-11-28"
_ACCEPT = "application/vnd.github+json"

# Retry on these statuses (in addition to transport errors).
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 5
_BASE_BACKOFF_S = 1.0


class GitHubAPIError(RuntimeError):
    """Non-retryable GitHub API error (or retries exhausted)."""

    def __init__(self, message: str, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class GitHubClient:
    """
    Thin async wrapper around httpx for the GitHub REST API.

    Auth: either an App JWT (for /app/* endpoints) or an installation token.
    Never logs the token value.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        app_jwt: str | None = None,
        installation_token: str | None = None,
    ) -> None:
        if app_jwt and installation_token:
            raise ValueError("pass either app_jwt or installation_token, not both")
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            timeout=httpx.Timeout(settings.github_timeout_s),
            headers={
                "Accept": _ACCEPT,
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "PRism/0.1.0",
                **self._auth_headers(app_jwt, installation_token),
            },
        )

    @staticmethod
    def _auth_headers(app_jwt: str | None, installation_token: str | None) -> dict[str, str]:
        if app_jwt:
            return {"Authorization": f"Bearer {app_jwt}"}
        if installation_token:
            return {"Authorization": f"Bearer {installation_token}"}
        return {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GitHubClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- core request machinery -------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = await self._client.request(
                    method, path, params=params, json=json, headers=headers
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                await self._sleep_backoff(attempt)
                continue

            if resp.status_code == 403 and self._rate_limit_exhausted(resp):
                await self._sleep_until_reset(resp)
                continue

            if resp.status_code in _RETRY_STATUSES:
                last_error = GitHubAPIError(
                    f"GitHub {method} {path} -> {resp.status_code} (retryable)",
                    status=resp.status_code,
                )
                await self._sleep_backoff(attempt, retry_after=resp.headers.get("retry-after"))
                continue

            if resp.status_code >= 400:
                raise GitHubAPIError(
                    f"GitHub {method} {path} -> {resp.status_code}: {resp.text[:500]}",
                    status=resp.status_code,
                    body=resp.text[:2000],
                )
            if resp.status_code == 204 or not resp.content:
                return None
            return resp.json()

        raise GitHubAPIError(
            f"GitHub {method} {path} failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    async def get(self, path: str, **kwargs: Any) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Any:
        return await self.request("POST", path, **kwargs)

    async def paginate(self, path: str, *, params: dict[str, Any] | None = None) -> list[Any]:
        """Collect all pages of a list endpoint (per_page=100)."""
        items: list[Any] = []
        page = 1
        params = dict(params or {})
        while True:
            params.update({"per_page": 100, "page": page})
            batch = await self.get(path, params=params)
            if not isinstance(batch, list) or not batch:
                break
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return items

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _rate_limit_exhausted(resp: httpx.Response) -> bool:
        return bool(resp.headers.get("x-ratelimit-remaining") == "0")

    async def _sleep_until_reset(self, resp: httpx.Response) -> None:
        reset = resp.headers.get("x-ratelimit-reset")
        wait_s = 60.0
        if reset and reset.isdigit():
            wait_s = max(1.0, float(reset) - time.time() + 2.0)
        log.warning("github_rate_limit_exhausted", sleeping_s=round(wait_s, 1))
        await asyncio.sleep(wait_s)

    async def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)
        else:
            delay = _BASE_BACKOFF_S * (2 ** (attempt - 1))
        await asyncio.sleep(delay)
