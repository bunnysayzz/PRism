"""Tests for the worker: per-repo config merge and pr_summary plumbing."""

from types import SimpleNamespace

import pytest

import prism.worker as worker_mod
from prism.config import RepoConfig
from prism.review.engine import make_finding
from prism.review.schemas import Severity


class _ClientCtx:
    """Async context manager wrapper around a fake client."""

    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *exc):
        return False


def _result_ns(result_kwargs):
    result_kwargs = result_kwargs or {}
    return SimpleNamespace(
        findings=result_kwargs.get("findings", []),
        skipped_files=result_kwargs.get("skipped_files", []),
        model="vllm",
        backend_errors=result_kwargs.get("backend_errors", []),
        pr_summary="Great PR that adds retries.",
    )


async def _run_review_pr(
    monkeypatch, repo_file_cfg, result_kwargs=None, accept_ticket_context=False
):
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

    if accept_ticket_context:

        async def fake_review(settings, cfg, client, owner, repo, pr_number, ticket_context=None):
            captured["ticket_context"] = ticket_context
            captured["cfg"] = cfg
            return _result_ns(result_kwargs)

    else:

        async def fake_review(settings, cfg, client, owner, repo, pr_number):
            captured["cfg"] = cfg
            return _result_ns(result_kwargs)

    monkeypatch.setattr(worker_mod, "review_pull_request", fake_review)
    monkeypatch.setattr(
        worker_mod.gh_reviews, "get_pr_head_sha", lambda c, o, r, n: _coro("abc123")
    )

    posted: dict = {}

    async def fake_post(client, owner, repo, pr_number, sha, summary, findings, event="COMMENT"):
        posted["summary"] = summary
        posted["event"] = event
        posted["findings"] = findings
        return {"id": 1}

    monkeypatch.setattr(worker_mod.gh_reviews, "post_review", fake_post)

    labels: dict = {}

    async def fake_labels(client, owner, repo, pr_number, verdict):
        labels["verdict"] = verdict
        return None

    monkeypatch.setattr(worker_mod.gh_reviews, "post_pr_labels", fake_labels)

    result = await worker_mod.review_pr({}, 123, "o", "r", 42)
    return result, captured, posted, labels


def _coro(value):
    async def _inner():
        return value

    return _inner()


@pytest.mark.asyncio
async def test_review_pr_merges_repo_file_config(monkeypatch):
    result, captured, _, _ = await _run_review_pr(
        monkeypatch, {"instructions": "repo says", "review_profile": "chill"}
    )
    assert result["status"] == "ok"
    assert result["findings"] == 0
    assert result["verdict"] == "Safe to merge"
    assert result["merge_score"] == 1.0
    cfg = captured["cfg"]
    assert cfg["instructions"] == "repo says"  # repo file wins
    assert cfg["confidence_verify"] == 0.75  # chill profile applied
    assert cfg["confidence_auto_post"] == 0.9  # server value kept


@pytest.mark.asyncio
async def test_review_pr_passes_pr_summary_to_render(monkeypatch):
    _, _, posted, _ = await _run_review_pr(monkeypatch, {})
    assert "<details><summary>PR summary</summary>" in posted["summary"]
    assert "Great PR that adds retries." in posted["summary"]
    assert "**Verdict:** Safe to merge (score 1.00)" in posted["summary"]
    assert "**Estimated effort:** 1/5 · ~5 min" in posted["summary"]


@pytest.mark.asyncio
async def test_review_pr_approves_truly_clean_pr(monkeypatch):
    result, _, posted, labels = await _run_review_pr(monkeypatch, {})
    assert posted["event"] == "APPROVE"
    assert posted["findings"] == []
    assert result["verdict"] == "Safe to merge"
    assert result["merge_score"] == 1.0
    assert labels["verdict"] == "Safe to merge"


@pytest.mark.asyncio
async def test_review_pr_comments_when_actionable_findings(monkeypatch):
    finding = make_finding(severity=Severity.high, confidence=0.95)
    result, _, posted, labels = await _run_review_pr(monkeypatch, {}, {"findings": [finding]})
    assert posted["event"] == "COMMENT"
    assert len(posted["findings"]) == 1  # actionable posted inline
    assert result["merge_score"] == pytest.approx(1 - 0.35 * 0.95)
    assert result["verdict"] == "Needs careful review"
    assert labels["verdict"] == "Needs careful review"


@pytest.mark.asyncio
async def test_review_pr_comments_when_only_nitpicks(monkeypatch):
    nitpick = make_finding(severity=Severity.low, confidence=0.5)
    result, _, posted, _ = await _run_review_pr(monkeypatch, {}, {"findings": [nitpick]})
    assert posted["event"] == "COMMENT"  # nitpicks block APPROVE
    assert posted["findings"] == []  # nitpicks never posted inline
    assert "Nitpick comments (1)" in posted["summary"]
    assert result["verdict"] == "Safe to merge"  # 1 - 0.05*0.5 = 0.975


@pytest.mark.asyncio
async def test_review_pr_comments_on_backend_errors(monkeypatch):
    result, _, posted, _ = await _run_review_pr(monkeypatch, {}, {"backend_errors": ["a.py: boom"]})
    assert posted["event"] == "COMMENT"
    assert "Review incomplete" in posted["summary"]
    assert "Safe to merge" not in posted["summary"]  # never bless an incomplete review
    assert result["verdict"] == "Needs careful review"  # ...not even in labels/metadata


@pytest.mark.asyncio
async def test_review_pr_comments_on_skipped_files(monkeypatch):
    _, _, posted, _ = await _run_review_pr(monkeypatch, {}, {"skipped_files": ["big.bin"]})
    assert posted["event"] == "COMMENT"


@pytest.mark.asyncio
async def test_review_pr_passes_ticket_context_when_engine_supports(monkeypatch):
    # Tracker module is absent; ticket_context falls back to None, but the
    # kwarg is still passed when the engine signature accepts it.
    _, captured, _, _ = await _run_review_pr(monkeypatch, {}, accept_ticket_context=True)
    assert captured["ticket_context"] is None


@pytest.mark.asyncio
async def test_review_pr_omits_ticket_context_when_engine_lacks_it(monkeypatch):
    # Default fake_review takes no ticket_context kwarg: must not TypeError.
    result, captured, _, _ = await _run_review_pr(monkeypatch, {})
    assert result["status"] == "ok"
    assert "ticket_context" not in captured
