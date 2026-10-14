"""
Ephemeral in-memory review analytics.

NOTE: this store lives in process memory. On serverless (Vercel) it resets
on every cold start and is not shared across instances, so the aggregates
are best-effort, not durable metrics. A persistent store (SQLite/Postgres)
is the follow-up; the record/get function contract stays the same.
"""

from __future__ import annotations

import time
from typing import Any

# Key: (owner, repo). Values: newest entries appended at the end.
_STORE: dict[tuple[str, str], list[dict[str, Any]]] = {}

#: Max entries kept per repo; oldest are dropped first.
_MAX_ENTRIES_PER_REPO = 200

#: Stat keys we keep from the recorded stats dict.
_WANTED_KEYS = ("findings", "nitpicks", "merge_score", "verdict", "backend_errors")

#: How many recent PRs feed the findings_per_pr_last20 series.
_LAST_N = 20


def record_review(owner: str, repo: str, pr_number: int, stats: dict[str, Any]) -> None:
    """Record one review's stats for a repo. Never raises.

    Contract (called by worker.py via lazy import - keep the signature):
    record_review(owner, repo, pr_number, stats).
    """
    try:
        entry: dict[str, Any] = {"ts": time.time(), "pr": pr_number}
        if isinstance(stats, dict):
            for key in _WANTED_KEYS:
                if key in stats:
                    entry[key] = stats[key]
            severities = stats.get("severities")
            if isinstance(severities, dict):
                entry["severities"] = dict(severities)
        bucket = _STORE.setdefault((str(owner), str(repo)), [])
        bucket.append(entry)
        if len(bucket) > _MAX_ENTRIES_PER_REPO:
            del bucket[: len(bucket) - _MAX_ENTRIES_PER_REPO]
    except Exception:
        pass  # analytics must never break a review


def get_repo_stats(owner: str, repo: str) -> dict[str, Any]:
    """Aggregate the recorded reviews for a repo into summary stats."""
    entries = _STORE.get((str(owner), str(repo)), [])
    recent = entries[-_LAST_N:]

    severity_distribution: dict[str, int] = {}
    for entry in entries:
        severities = entry.get("severities")
        if isinstance(severities, dict):
            for name, count in severities.items():
                key = str(name)
                severity_distribution[key] = severity_distribution.get(key, 0) + _as_int(count)

    scores = [
        float(entry["merge_score"])
        for entry in entries
        if isinstance(entry.get("merge_score"), (int, float))
        and not isinstance(entry.get("merge_score"), bool)
    ]
    verdict_counts: dict[str, int] = {}
    for entry in entries:
        verdict = entry.get("verdict")
        if verdict is not None:
            name = str(verdict)
            verdict_counts[name] = verdict_counts.get(name, 0) + 1

    return {
        "owner": owner,
        "repo": repo,
        "total_reviews": len(entries),
        "findings_per_pr_last20": [_as_count(entry.get("findings")) for entry in recent],
        "severity_distribution": severity_distribution,
        "avg_merge_score": (sum(scores) / len(scores)) if scores else None,
        "verdict_counts": verdict_counts,
    }


def _as_count(value: Any) -> int:
    """Coerce a findings stat (count or list) to an int."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


def _as_int(value: Any) -> int:
    """Coerce a severity bucket to an int; garbage becomes 0."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0
