"""Tests for per-repo .prism.yml config: parsing, fetching, and merging."""

import base64

import pytest

from prism.config import RepoConfig, load_repo_config, merge_repo_config
from prism.github.client import GitHubAPIError
from prism.github.repoconfig import fetch_repo_file_config


def _server() -> RepoConfig:
    return RepoConfig(
        {
            "severity_threshold": "low",
            "categories": {"style": True, "security": True},
            "max_findings_per_file": 10,
            "ignore_globs": ["**/dist/**"],
            "instructions": "server instruction",
            "path_instructions": {"src/**": "server note"},
            "confidence_auto_post": 0.9,
            "confidence_verify": 0.7,
        }
    )


def test_merge_repo_wins_scalars():
    merged = merge_repo_config(
        _server(),
        {
            "severity_threshold": "high",
            "max_findings_per_file": 3,
            "instructions": "repo instruction",
            "confidence_auto_post": 0.75,
        },
    )
    assert merged["severity_threshold"] == "high"
    assert merged["max_findings_per_file"] == 3
    assert merged["instructions"] == "repo instruction"
    assert merged["confidence_auto_post"] == 0.75
    # Untouched keys fall back to the server config.
    assert merged["confidence_verify"] == 0.7


def test_merge_dicts_merged_per_key_repo_wins():
    merged = merge_repo_config(
        _server(),
        {
            "path_instructions": {"src/**": "repo note", "docs/**": "new note"},
            "categories": {"style": False},
        },
    )
    assert merged["path_instructions"] == {"src/**": "repo note", "docs/**": "new note"}
    assert merged["categories"] == {"style": False, "security": True}


def test_merge_ignore_globs_union_deduped_ordered():
    merged = merge_repo_config(_server(), {"ignore_globs": ["**/dist/**", "**/vendor/**"]})
    assert merged["ignore_globs"] == ["**/dist/**", "**/vendor/**"]


def test_merge_does_not_mutate_server():
    server = _server()
    merge_repo_config(server, {"severity_threshold": "high", "ignore_globs": ["**/x/**"]})
    assert server["severity_threshold"] == "low"
    assert server["ignore_globs"] == ["**/dist/**"]


def test_merge_profile_chill():
    merged = merge_repo_config(_server(), {"review_profile": "chill"})
    assert merged["confidence_auto_post"] == 0.9
    assert merged["confidence_verify"] == 0.75
    assert merged["max_findings_per_file"] == 5
    assert merged["severity_threshold"] == "medium"


def test_merge_profile_assertive():
    merged = merge_repo_config(_server(), {"review_profile": "assertive"})
    assert merged["confidence_auto_post"] == 0.7
    assert merged["confidence_verify"] == 0.5
    assert merged["max_findings_per_file"] == 15
    assert merged["severity_threshold"] == "low"


def test_merge_explicit_keys_beat_profile():
    merged = merge_repo_config(_server(), {"review_profile": "chill", "max_findings_per_file": 42})
    assert merged["max_findings_per_file"] == 42  # explicit wins over profile
    assert merged["confidence_verify"] == 0.75  # profile still fills the rest


def test_merge_unknown_profile_ignored():
    merged = merge_repo_config(_server(), {"review_profile": "turbo"})
    assert merged["max_findings_per_file"] == 10
    assert merged["severity_threshold"] == "low"


def test_load_repo_config_parses_new_keys(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "instructions: be kind\n"
        "path_instructions:\n"
        "  src/**: use pathlib\n"
        "confidence_auto_post: 0.95\n"
        "confidence_verify: 0.8\n",
        encoding="utf-8",
    )
    cfg = load_repo_config(cfg_file)
    assert cfg["instructions"] == "be kind"
    assert cfg["path_instructions"] == {"src/**": "use pathlib"}
    assert cfg["confidence_auto_post"] == 0.95
    assert cfg["confidence_verify"] == 0.8


def test_load_repo_config_new_keys_default(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("severity_threshold: high\n", encoding="utf-8")
    cfg = load_repo_config(cfg_file)
    assert cfg["instructions"] == ""
    assert cfg["path_instructions"] == {}
    assert cfg["confidence_auto_post"] is None
    assert cfg["confidence_verify"] is None


class _FakeClient:
    def __init__(self, payload=None, exc=None):
        self.payload = payload
        self.exc = exc
        self.paths: list[str] = []

    async def get(self, path):
        self.paths.append(path)
        if self.exc is not None:
            raise self.exc
        return self.payload


def _payload(text: str) -> dict[str, str]:
    return {"content": base64.b64encode(text.encode()).decode(), "encoding": "base64"}


@pytest.mark.asyncio
async def test_fetch_returns_parsed_dict():
    client = _FakeClient(_payload("instructions: be kind\nreview_profile: chill\n"))
    data = await fetch_repo_file_config(client, "o", "r")
    assert data == {"instructions": "be kind", "review_profile": "chill"}
    assert client.paths == ["/repos/o/r/contents/.prism.yml"]


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_404():
    client = _FakeClient(exc=GitHubAPIError("not found", status=404))
    assert await fetch_repo_file_config(client, "o", "r") == {}


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_api_error():
    client = _FakeClient(exc=GitHubAPIError("server blew up", status=500))
    assert await fetch_repo_file_config(client, "o", "r") == {}


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_transport_error():
    client = _FakeClient(exc=RuntimeError("boom"))
    assert await fetch_repo_file_config(client, "o", "r") == {}


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_bad_yaml():
    client = _FakeClient(_payload("{oops"))
    assert await fetch_repo_file_config(client, "o", "r") == {}


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_non_dict_yaml():
    client = _FakeClient(_payload("- just\n- a list\n"))
    assert await fetch_repo_file_config(client, "o", "r") == {}


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_bad_base64():
    client = _FakeClient({"content": "!!!not-base64!!!", "encoding": "base64"})
    assert await fetch_repo_file_config(client, "o", "r") == {}
