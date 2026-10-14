"""Tests for the ephemeral in-memory review analytics store."""

import pytest

from prism.review import analytics


@pytest.fixture(autouse=True)
def _clean_store():
    analytics._STORE.clear()
    yield
    analytics._STORE.clear()


def test_record_and_aggregate():
    analytics.record_review(
        "myorg",
        "myrepo",
        1,
        {
            "findings": 3,
            "nitpicks": 1,
            "merge_score": 0.8,
            "verdict": "changes_requested",
            "backend_errors": [],
            "severities": {"high": 2, "medium": 1},
        },
    )
    analytics.record_review(
        "myorg",
        "myrepo",
        2,
        {
            "findings": 1,
            "nitpicks": 0,
            "merge_score": 0.9,
            "verdict": "approve",
            "backend_errors": [],
            "severities": {"medium": 1},
        },
    )
    stats = analytics.get_repo_stats("myorg", "myrepo")
    assert stats["total_reviews"] == 2
    assert stats["findings_per_pr_last20"] == [3, 1]
    assert stats["severity_distribution"] == {"high": 2, "medium": 2}
    assert stats["avg_merge_score"] == pytest.approx(0.85)
    assert stats["verdict_counts"] == {"changes_requested": 1, "approve": 1}


def test_only_wanted_keys_stored():
    analytics.record_review("o", "r", 1, {"findings": 2, "model": "secret-sauce", "api_key": "x"})
    stored = analytics._STORE[("o", "r")][0]
    assert stored["findings"] == 2
    assert "model" not in stored
    assert "api_key" not in stored


def test_findings_as_list_counts():
    analytics.record_review("o", "r", 1, {"findings": ["a", "b", "c"]})
    stats = analytics.get_repo_stats("o", "r")
    assert stats["findings_per_pr_last20"] == [3]


def test_cap_drops_oldest():
    for pr in range(1, 211):
        analytics.record_review("o", "r", pr, {"findings": pr})
    stats = analytics.get_repo_stats("o", "r")
    assert stats["total_reviews"] == 200
    bucket = analytics._STORE[("o", "r")]
    assert bucket[0]["pr"] == 11  # prs 1-10 dropped
    assert bucket[-1]["pr"] == 210


def test_findings_series_capped_at_last20():
    for pr in range(1, 26):
        analytics.record_review("o", "r", pr, {"findings": pr})
    stats = analytics.get_repo_stats("o", "r")
    assert stats["findings_per_pr_last20"] == list(range(6, 26))


def test_repos_are_isolated():
    analytics.record_review("o", "r1", 1, {"findings": 5})
    analytics.record_review("o", "r2", 1, {"findings": 7})
    assert analytics.get_repo_stats("o", "r1")["total_reviews"] == 1
    assert analytics.get_repo_stats("o", "r2")["findings_per_pr_last20"] == [7]


def test_empty_repo_stats():
    stats = analytics.get_repo_stats("nobody", "nothing")
    assert stats["total_reviews"] == 0
    assert stats["findings_per_pr_last20"] == []
    assert stats["severity_distribution"] == {}
    assert stats["avg_merge_score"] is None
    assert stats["verdict_counts"] == {}


def test_record_never_raises_on_garbage():
    analytics.record_review("o", "r", 1, None)  # type: ignore[arg-type]
    analytics.record_review("o", "r", 2, "junk")  # type: ignore[arg-type]
    analytics.record_review("o", "r", 3, ["junk"])  # type: ignore[arg-type]
    analytics.record_review("o", "r", 4, {"findings": object(), "merge_score": "high"})
    analytics.record_review(None, None, 5, {})  # type: ignore[arg-type]
    stats = analytics.get_repo_stats("o", "r")
    assert stats["total_reviews"] == 4
    assert stats["findings_per_pr_last20"] == [0, 0, 0, 0]
    assert stats["avg_merge_score"] is None


def test_get_repo_stats_never_raises_on_garbage():
    analytics.record_review(
        "o",
        "r",
        1,
        {"findings": "many", "severities": {"high": "lots"}, "verdict": 42, "merge_score": True},
    )
    stats = analytics.get_repo_stats("o", "r")
    assert stats["severity_distribution"] == {"high": 0}
    assert stats["verdict_counts"] == {"42": 1}
    assert stats["avg_merge_score"] is None
