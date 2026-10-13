"""Tests for the worker: per-repo config merge and pr_summary plumbing."""

from types import SimpleNamespace

import pytest

import prism.worker as worker_mod
from prism.config import RepoConfig


class _ClientCtx:
    """Async context manager wrapper around a fake client."""

    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *exc):
        return False


async def _run_review_pr(monkeypatch, repo_file_cfg):
    captured: dict = {}
    settings = SimpleNamespace(log_level="INFO", redis_url=None)
    monkeypatch.setattr(worker_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(worker_mod, "setup_logging", lambda level: None)
    monkeypatch.setattr(
        worker_mod,
        "load_repo_config",
        lambda: RepoConfig(
            {
                "repos_allowlist": [],
                "severity_threshold": "low",
                "ignore_globs": ["**/dist/**"],
                "instructions": "server says",
                "path_instructions": {},
                "confidence_auto_post": 0.9,
                "confidence_verify": 0.7,
            }
        ),
    )
    monkeypatch.setattr(worker_mod, "get_installation_token", lambda settings, iid: _coro("tok"))
    monkeypatch.setattr(worker_mod, "GitHubClient", lambda *a, **k: _ClientCtx(object()))
    monkeypatch.setattr(
        worker_mod, "fetch_repo_file_config", lambda client, o, r: _coro(repo_file_cfg)
    )

    async def fake_review(settings, cfg, client, owner, repo, pr_number):
        captured["cfg"] = cfg
        return SimpleNamespace(
            findings=[],
            skipped_files=[],
            model="vllm",
            backend_errors=[],
            pr_summary="Great PR that adds retries.",
        )

    monkeypatch.setattr(worker_mod, "review_pull_request", fake_review)
    monkeypatch.setattr(
        worker_mod.gh_reviews, "get_pr_head_sha", lambda c, o, r, n: _coro("abc123")
    )

    posted: dict = {}

    async def fake_post(client, owner, repo, pr_number, sha, summary, findings):
        posted["summary"] = summary
        return {"id": 1}

    monkeypatch.setattr(worker_mod.gh_reviews, "post_review", fake_post)

    result = await worker_mod.review_pr({}, 123, "o", "r", 42)
    return result, captured, posted


def _coro(value):
    async def _inner():
        return value

    return _inner()


@pytest.mark.asyncio
async def test_review_pr_merges_repo_file_config(monkeypatch):
    result, captured, _ = await _run_review_pr(
        monkeypatch, {"instructions": "repo says", "review_profile": "chill"}
    )
    assert result == {"status": "ok", "findings": 0}
    cfg = captured["cfg"]
    assert cfg["instructions"] == "repo says"  # repo file wins
    assert cfg["confidence_verify"] == 0.75  # chill profile applied
    assert cfg["confidence_auto_post"] == 0.9  # server value kept


@pytest.mark.asyncio
async def test_review_pr_passes_pr_summary_to_render(monkeypatch):
    _, _, posted = await _run_review_pr(monkeypatch, {})
    assert "<details><summary>PR summary</summary>" in posted["summary"]
    assert "Great PR that adds retries." in posted["summary"]
