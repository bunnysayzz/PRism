"""
GitHub App authentication.

Flow:
  1. Sign a short-lived JWT with the App's RSA private key (RS256).
     Claims: iss = App ID, iat = now - 60s (clock skew), exp = iat + 10min (GitHub max).
  2. Exchange it for an installation access token:
     POST /app/installations/{installation_id}/access_tokens
  3. Cache the token in memory, expiring 60s before GitHub's expiry.

The private key is loaded from a file path (GITHUB_PRIVATE_KEY_PATH) only —
never from env contents, never logged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC

import jwt  # PyJWT

from prism.config import Settings
from prism.github.client import GitHubClient
from prism.logging import get_logger

log = get_logger(__name__)

# GitHub allows JWTs valid for at most 10 minutes.
_JWT_LIFETIME_S = 10 * 60
_CLOCK_SKEW_S = 60
# Refresh cached installation tokens this far before their real expiry.
_TOKEN_REFRESH_SKEW_S = 60


def create_app_jwt(settings: Settings, now: float | None = None) -> str:
    """
    Create a GitHub App JWT.

    Claims per GitHub docs:
      - iss: the App ID
      - iat: issued-at, set 60s in the past to tolerate clock skew
      - exp: iat + 10 minutes (GitHub rejects longer lifetimes)
    """
    now = int(now if now is not None else time.time())
    payload = {
        "iss": str(settings.github_app_id),
        "iat": now - _CLOCK_SKEW_S,
        "exp": now - _CLOCK_SKEW_S + _JWT_LIFETIME_S,
    }
    private_key = settings.private_key_pem()
    return jwt.encode(payload, private_key, algorithm="RS256")


@dataclass
class _CachedToken:
    token: str
    expires_at: float  # epoch seconds (already skewed 60s early)


class InstallationTokenCache:
    """In-memory cache of installation tokens, one entry per installation."""

    def __init__(self) -> None:
        self._tokens: dict[int, _CachedToken] = {}

    def get(self, installation_id: int) -> str | None:
        cached = self._tokens.get(installation_id)
        if cached and cached.expires_at > time.time():
            return cached.token
        self._tokens.pop(installation_id, None)
        return None

    def put(self, installation_id: int, token: str, expires_at_iso: str) -> None:
        # GitHub returns ISO8601, e.g. "2026-10-07T12:34:56Z".
        expires_at = _parse_iso8601(expires_at_iso) - _TOKEN_REFRESH_SKEW_S
        self._tokens[installation_id] = _CachedToken(token=token, expires_at=expires_at)


def _parse_iso8601(value: str) -> float:
    from datetime import datetime

    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


_token_cache = InstallationTokenCache()


async def get_installation_token(
    settings: Settings,
    installation_id: int,
    client: GitHubClient | None = None,
) -> str:
    """
    Return a valid installation access token, using the cache when possible.

    Endpoint: POST /app/installations/{installation_id}/access_tokens
    Authenticated with the App JWT (not with a previous installation token).
    """
    cached = _token_cache.get(installation_id)
    if cached:
        return cached

    app_jwt = create_app_jwt(settings)
    own_client = client is None
    if own_client:
        client = GitHubClient(settings, app_jwt=app_jwt)
    assert client is not None
    try:
        log.info("requesting_installation_token", installation_id=installation_id)
        data = await client.post(
            f"/app/installations/{installation_id}/access_tokens",
            json={"permissions": {"pull_requests": "write", "contents": "read"}},
        )
        token: str = data["token"]
        _token_cache.put(installation_id, token, data["expires_at"])
        return token
    finally:
        if own_client:
            await client.aclose()
