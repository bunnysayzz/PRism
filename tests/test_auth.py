"""Tests for GitHub App authentication (installation token exchange)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from prism.config import Settings
from prism.github import auth
from prism.github.auth import get_installation_token
from prism.github.client import GitHubAPIError

_FUTURE = "2999-01-01T00:00:00Z"


def _settings() -> Settings:
    return Settings(
        github_app_id=123456,
        github_webhook_secret="secret",
        github_private_key_path="",
        review_backend="stub",
    )


@pytest.fixture(autouse=True)
def _no_jwt(monkeypatch: Any) -> None:
    """Skip real JWT signing; the token exchange doesn't need it."""
    monkeypatch.setattr(auth, "create_app_jwt", lambda settings: "fake-jwt")
    auth._token_cache._tokens.clear()


def _client(**kwargs: Any) -> AsyncMock:
    client = AsyncMock()
    client.post.side_effect = kwargs["side_effect"]
    return client


async def test_422_falls_back_to_no_permissions() -> None:
    """First call 422s -> retry with empty JSON, token returned."""
    client = _client(
        side_effect=[
            GitHubAPIError("permissions not granted", status=422),
            {"token": "tok-fallback", "expires_at": _FUTURE},
        ]
    )
    token = await get_installation_token(_settings(), 111, client=client)
    assert token == "tok-fallback"
    assert client.post.await_count == 2
    first_kwargs = client.post.await_args_list[0].kwargs
    assert first_kwargs["json"] == {"permissions": {"pull_requests": "write", "contents": "read"}}
    second_kwargs = client.post.await_args_list[1].kwargs
    assert second_kwargs["json"] == {}
    assert "permissions" not in second_kwargs["json"]


async def test_non_422_error_propagates() -> None:
    client = _client(side_effect=[GitHubAPIError("server blew up", status=500)])
    with pytest.raises(GitHubAPIError):
        await get_installation_token(_settings(), 222, client=client)
    assert client.post.await_count == 1


async def test_successful_first_call_no_retry() -> None:
    client = _client(side_effect=[{"token": "tok-ok", "expires_at": _FUTURE}])
    token = await get_installation_token(_settings(), 333, client=client)
    assert token == "tok-ok"
    assert client.post.await_count == 1
