"""Tests for the prism.data pipeline: cleaner, dedup, splits, format, collector, CLI."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import collect_pr_data  # noqa: E402

from prism.data import cleaner, dedup, format, splits  # noqa: E402
from prism.data.cleaner import CleanerConfig  # noqa: E402
from prism.data.collector import (  # noqa: E402
    Checkpoint,
    CollectorConfig,
    CollectorError,
    GitHubCollector,
    load_raw_samples,
)
from prism.data.models import CleanSample, RawSample, TrainingRecord  # noqa: E402
from prism.review.schemas import parse_findings  # noqa: E402

SAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "records.jsonl"


def _load_sample_raws() -> list[RawSample]:
    """Parse the handcrafted sample dataset into RawSamples."""
    return [
        RawSample.model_validate(json.loads(line))
        for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


_PATCH = "@@ -1,2 +1,2 @@\n ctx\n-old\n+new\n"
_BODY = (
    "This change looks incorrect because the new code path breaks the existing "
    "behavior for edge cases we already handle."
)


def _raw(**overrides: Any) -> RawSample:
    data: dict[str, Any] = {
        "sample_id": "o/r#1#src/a.py#c1",
        "repo": "o/r",
        "pr_number": 1,
        "pr_title": "t",
        "path": "src/a.py",
        "patch": _PATCH,
        "comment_id": 1,
        "comment_author": "alice",
        "comment_author_type": "User",
        "comment_body": _BODY,
        "comment_line": 2,
        "comment_original_line": 2,
        "comment_side": "RIGHT",
    }
    data.update(overrides)
    return RawSample.model_validate(data)


def _clean(**overrides: Any) -> CleanSample:
    data: dict[str, Any] = {
        "sample_id": "o/r#1#src/a.py#c1",
        "repo": "o/r",
        "pr_number": 1,
        "pr_title": "t",
        "path": "src/a.py",
        "language": "python",
        "patch": _PATCH,
        "comment_id": 1,
        "comment_author": "alice",
        "comment_body": _BODY,
        "comment_line": 2,
        "comment_side": "RIGHT",
    }
    data.update(overrides)
    return CleanSample.model_validate(data)


# ---------------------------------------------------------------- cleaner ---


def test_cleaner_keeps_good_sample() -> None:
    sample, reason = cleaner.clean_sample(_raw(), CleanerConfig())
    assert reason == ""
    assert sample is not None
    assert sample.language == "python"
    assert sample.comment_line == 2


def test_cleaner_falls_back_to_original_line() -> None:
    sample, reason = cleaner.clean_sample(
        _raw(comment_line=None, comment_original_line=7), CleanerConfig()
    )
    assert reason == ""
    assert sample is not None
    assert sample.comment_line == 7


def test_cleaner_drops_bot_comments() -> None:
    for author, author_type in [
        ("dependabot[bot]", "User"),
        ("github-actions[bot]", "Bot"),
        ("renovate", "User"),
        ("codecov-io", "User"),
    ]:
        sample, reason = cleaner.clean_sample(
            _raw(comment_author=author, comment_author_type=author_type), CleanerConfig()
        )
        assert sample is None, author
        assert reason == "bot", author


def test_cleaner_drops_unanchored_comments() -> None:
    sample, reason = cleaner.clean_sample(
        _raw(comment_line=None, comment_original_line=None), CleanerConfig()
    )
    assert sample is None
    assert reason == "unanchored"


def test_cleaner_drops_huge_diffs() -> None:
    big_patch = "@@ -1 +1 @@\n" + "x" * 50_000
    sample, reason = cleaner.clean_sample(_raw(patch=big_patch), CleanerConfig())
    assert sample is None
    assert reason == "huge_diff"


def test_cleaner_length_guards() -> None:
    sample, reason = cleaner.clean_sample(_raw(comment_body="lgtm"), CleanerConfig())
    assert sample is None and reason == "comment_too_short"
    sample, reason = cleaner.clean_sample(_raw(comment_body="x" * 9_000), CleanerConfig())
    assert sample is None and reason == "comment_too_long"


def test_cleaner_language_detection() -> None:
    assert cleaner.detect_language("src/a.py") == "python"
    assert cleaner.detect_language("web/app.jsx") == "javascript"
    assert cleaner.detect_language("infra/main.tf") == "terraform"
    assert cleaner.detect_language("Dockerfile") == "dockerfile"
    assert cleaner.detect_language("Makefile") == "make"
    assert cleaner.detect_language("notes.xyz") == ""
    sample, reason = cleaner.clean_sample(_raw(path="notes.xyz"), CleanerConfig())
    assert sample is None and reason == "unknown_language"


def test_cleaner_redacts_secrets() -> None:
    body = "looks fine but rotate this: ghp_abcdefghij1234567890 right away please"
    patch = _PATCH + '+API_KEY = "sk-live-abc123XYZ"\n'
    sample, reason = cleaner.clean_sample(_raw(comment_body=body, patch=patch), CleanerConfig())
    assert reason == ""
    assert sample is not None
    assert "ghp_abcdefghij1234567890" not in sample.comment_body
    assert "[REDACTED]" in sample.comment_body
    assert "sk-live-abc123XYZ" not in sample.patch
    assert "[REDACTED]" in sample.patch


def test_cleaner_unicode_normalization() -> None:
    body = "This is ﬁne but the logic\u200b is wrong here, please double check it."
    sample, reason = cleaner.clean_sample(_raw(comment_body=body), CleanerConfig())
    assert reason == ""
    assert sample is not None
    assert "\u200b" not in sample.comment_body  # zero-width space stripped
    assert "fine" in sample.comment_body  # ﬁ ligature NFKC-folded


def test_clean_all_counts_reasons() -> None:
    raws = [
        _raw(),
        _raw(sample_id="o/r#1#src/b.py#c2", comment_author="dependabot[bot]"),
        _raw(sample_id="o/r#1#src/c.py#c3", comment_line=None, comment_original_line=None),
    ]
    kept, reasons = cleaner.clean_all(raws, CleanerConfig())
    assert len(kept) == 1
    assert reasons["kept"] == 1
    assert reasons["bot"] == 1
    assert reasons["unanchored"] == 1


# ------------------------------------------------------------------ dedup ---


def _body_words(tag: str, n: int = 100) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def test_dedup_exact_duplicates() -> None:
    a = _clean()
    b = _clean(sample_id="o/r#2#src/a.py#c9", pr_number=2, comment_id=9)
    kept, dropped = dedup.dedupe([a, b])
    assert len(kept) == 1 and kept[0].sample_id == a.sample_id
    assert dropped == [1]


def test_dedup_near_duplicates() -> None:
    words = _body_words("fox").split()
    words[50] = "changed"  # exactly one token differs -> Jaccard ~0.95
    a = _clean(comment_body=_body_words("fox"))
    b = _clean(sample_id="o/r#2#src/a.py#c9", comment_body=" ".join(words))
    pairs = dedup.find_duplicate_pairs([a, b], threshold=0.85)
    assert pairs == [(0, 1)]
    kept, dropped = dedup.dedupe([a, b], threshold=0.85)
    assert len(kept) == 1 and dropped == [1]


def test_dedup_keeps_distinct_samples() -> None:
    a = _clean(comment_body=_body_words("fox"))
    b = _clean(sample_id="x#1#f#c2", comment_body=_body_words("wolf"))
    kept, dropped = dedup.dedupe([a, b], threshold=0.85)
    assert len(kept) == 2 and dropped == []


def test_dedup_across_repos() -> None:
    # Same diff + comment text in a different repo (e.g. a fork) still dedups.
    a = _clean(repo="acme/lib")
    b = _clean(sample_id="fork/lib#1#src/a.py#c1", repo="someone-else/lib")
    kept, dropped = dedup.dedupe([a, b])
    assert len(kept) == 1 and dropped == [1]


def test_dedup_empty_input() -> None:
    assert dedup.dedupe([]) == ([], [])


# ----------------------------------------------------------------- splits ---


def _split_samples() -> list[CleanSample]:
    samples: list[CleanSample] = []
    for repo, count in [("r1", 5), ("r2", 3), ("r3", 2), ("r4", 2)]:
        for i in range(count):
            samples.append(_clean(sample_id=f"{repo}#1#f{i}#c{i}", repo=repo, pr_number=1))
    return samples


def _ids(split: dict[str, list[CleanSample]]) -> dict[str, list[str]]:
    return {name: sorted(s.sample_id for s in items) for name, items in split.items()}


def test_splits_deterministic() -> None:
    samples = _split_samples()
    kwargs: dict[str, Any] = {"key": lambda s: s.repo}
    assert _ids(splits.stratified_split(samples, seed=42, **kwargs)) == _ids(
        splits.stratified_split(samples, seed=42, **kwargs)
    )


def test_splits_no_repo_leakage_and_full_coverage() -> None:
    samples = _split_samples()
    result = splits.stratified_split(samples, seed=7, key=lambda s: s.repo)
    repo_sets = splits.repo_sets(result, key=lambda s: s.repo)
    assert repo_sets["train"].isdisjoint(repo_sets["val"])
    assert repo_sets["train"].isdisjoint(repo_sets["test"])
    assert repo_sets["val"].isdisjoint(repo_sets["test"])
    total = sum(len(v) for v in result.values())
    assert total == len(samples)
    assert all(len(v) > 0 for v in result.values())  # 4 repos -> no empty split


def test_splits_single_repo_goes_to_train() -> None:
    samples = [_clean(sample_id=f"r1#1#f{i}#c{i}") for i in range(3)]
    result = splits.stratified_split(samples, seed=1, key=lambda s: s.repo)
    assert len(result["train"]) == 3
    assert result["val"] == [] and result["test"] == []


def test_splits_invalid_ratios() -> None:
    samples = _split_samples()
    with pytest.raises(ValueError):
        splits.stratified_split(
            samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.2, key=lambda s: s.repo
        )


def test_write_manifest(tmp_path: Path) -> None:
    samples = _split_samples()
    result = splits.stratified_split(samples, seed=42, key=lambda s: s.repo)
    manifest = splits.write_manifest(
        result,
        out_path=tmp_path / "manifest.json",
        seed=42,
        ratios={"train": 0.9, "val": 0.05, "test": 0.05},
        key=lambda s: s.repo,
        id_key=lambda s: s.sample_id,
    )
    assert (tmp_path / "manifest.json").exists()
    assert manifest["seed"] == 42
    assert sum(v["count"] for v in manifest["splits"].values()) == len(samples)
    assert manifest["splits"]["train"]["sample_ids"]


# ----------------------------------------------------------------- format ---


def test_format_record_matches_finding_schema() -> None:
    raws = cleaner.clean_all(_load_sample_raws(), CleanerConfig())[0]
    assert len(raws) == 12  # all handcrafted samples survive cleaning
    record = format.build_record(raws[0])
    assert isinstance(record, TrainingRecord)
    findings = parse_findings(record.assistant)
    assert len(findings) == 1
    f = findings[0]
    assert f.path == "src/cart.py" and f.line == 19
    assert f.severity.value == "critical" and f.category == "security"
    assert record.meta["source"] == "handcrafted"
    # The reviewer comment is the label source, not model input.
    assert "SQL injection waiting to happen" not in record.user
    assert "```diff" in record.user  # diff present, with new-file line numbers
    assert "    19 | +    query" in record.user


def test_format_all_sample_findings_validate() -> None:
    cleaned, _ = cleaner.clean_all(_load_sample_raws(), CleanerConfig())
    for sample in cleaned:
        record = format.build_record(sample)
        findings = parse_findings(record.assistant)
        assert len(findings) == 1, sample.sample_id
        assert findings[0].line > 0


def test_format_heuristic_finding() -> None:
    body = (
        "SQL injection here — the user input is concatenated into the query. "
        "This is a serious vulnerability.\n"
        '```python\ncursor.execute("SELECT * FROM t WHERE x = %s", (x,))\n```'
    )
    sample = _clean(comment_body=body, finding=None)
    finding_obj, source = format.finding_for_sample(sample)
    assert source == "heuristic"
    assert finding_obj.category == "security"
    assert finding_obj.severity.value == "high"
    assert finding_obj.confidence == 0.6
    assert "cursor.execute" in finding_obj.suggestion
    assert finding_obj.title  # non-empty, <= 160 chars
    # Heuristic output also validates against the schema.
    assert len(parse_findings(format.build_record(sample).assistant)) == 1


def test_format_numbered_diff() -> None:
    patch = "@@ -1,2 +10,2 @@\n ctx\n-old\n+new"
    numbered = format.numbered_diff(patch)
    assert "    10 |  ctx" in numbered
    assert "       | -old" in numbered
    assert "    11 | +new" in numbered


def test_token_estimate_and_length_filter() -> None:
    assert format.estimate_tokens("a" * 400) == 100
    assert format.estimate_tokens("") == 1
    record = TrainingRecord(system="s", user="u", assistant="a")
    kept, dropped = format.filter_by_length([record], max_tokens=10_000)
    assert kept == [record] and dropped == []
    kept, dropped = format.filter_by_length([record], max_tokens=1)
    assert kept == [] and dropped == [record]


# --------------------------------------------------------------- collector ---


def _github_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    page = int(request.url.params.get("page", "1"))
    if path == "/repos/o/r/pulls":
        if page == 1:
            return httpx.Response(
                200,
                json=[
                    {"number": 1, "title": "Add thing", "review_comments": 2},
                    {"number": 2, "title": "No comments", "review_comments": 0},
                ],
            )
        return httpx.Response(200, json=[])
    if path == "/repos/o/r/pulls/1/files":
        if page == 1:
            return httpx.Response(
                200, json=[{"filename": "a.py", "status": "modified", "patch": _PATCH}]
            )
        return httpx.Response(200, json=[])
    if path == "/repos/o/r/pulls/1/comments":
        if page == 1:
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 11,
                        "user": {"login": "alice", "type": "User"},
                        "body": _BODY,
                        "path": "a.py",
                        "line": 2,
                        "original_line": 2,
                        "side": "RIGHT",
                    },
                    {
                        "id": 12,
                        "user": {"login": "bob", "type": "User"},
                        "body": "general remark with no anchor here",
                        "path": None,
                        "line": None,
                        "original_line": None,
                        "side": "RIGHT",
                    },
                ],
            )
        return httpx.Response(200, json=[])
    return httpx.Response(404, json={"message": "not found"})


def test_collector_collects_anchored_samples() -> None:
    async def _run() -> list[RawSample]:
        collector = GitHubCollector("test-token", transport=httpx.MockTransport(_github_handler))
        async with collector:
            prs = await collector.collect_pr_list("o/r")
            assert len(prs) == 2
            return await collector.collect_pr("o/r", prs[0])

    samples = asyncio.run(_run())
    assert len(samples) == 1  # unanchored comment skipped
    s = samples[0]
    assert s.sample_id == "o/r#1#a.py#c11"
    assert s.comment_author == "alice"
    assert s.comment_line == 2
    assert s.patch == _PATCH


def test_collector_run_skips_prs_without_comments_and_resumes(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _github_handler(request)

    async def _run() -> Any:
        collector = GitHubCollector("t", transport=httpx.MockTransport(handler))
        config = CollectorConfig(
            repos=["o/r"],
            out_dir=tmp_path / "raw",
            checkpoint_path=tmp_path / "raw" / "checkpoint.json",
        )
        return await collector.run(config)

    stats = asyncio.run(_run())
    assert stats.prs_collected == 1
    assert stats.prs_skipped_no_comments == 1
    assert stats.samples_written == 1
    assert (tmp_path / "raw" / "raw.jsonl").exists()
    assert "o/r#1" in Checkpoint(tmp_path / "raw" / "checkpoint.json").processed

    # Second run: PR list is re-fetched, but no per-PR file/comment calls happen.
    calls.clear()
    stats2 = asyncio.run(_run())
    assert stats2.prs_skipped_checkpoint == 1
    assert stats2.samples_written == 0
    assert not any(p.endswith(("/files", "/comments")) for p in calls)


def test_collector_rate_limit_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import time as _time

    sleeps: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(
                403,
                json={"message": "rate limited"},
                headers={
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": str(int(_time.time())),
                },
            )
        return httpx.Response(200, json=[])

    async def _run() -> Any:
        collector = GitHubCollector("t", transport=httpx.MockTransport(handler))
        async with collector:
            return await collector._request("GET", "/repos/o/r/pulls")

    result = asyncio.run(_run())
    assert result == []
    assert state["n"] == 2  # retried after backoff
    assert sleeps  # actually slept


def test_collector_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(CollectorError, match="GH_TOKEN"):
        GitHubCollector.from_env()
    with pytest.raises(CollectorError):
        GitHubCollector("  ")


def test_collector_invalid_repo() -> None:
    async def _run() -> None:
        collector = GitHubCollector("t", transport=httpx.MockTransport(_github_handler))
        await collector.run(CollectorConfig(repos=["not-a-repo"]))

    with pytest.raises(CollectorError, match="owner/name"):
        asyncio.run(_run())


def test_load_raw_samples_dedupes(tmp_path: Path) -> None:
    p = tmp_path / "raw.jsonl"
    line = json.dumps(_raw().model_dump(mode="json"))
    p.write_text(line + "\n" + line + "\n" + "not json\n", encoding="utf-8")
    samples = load_raw_samples(p)
    assert len(samples) == 1


# --------------------------------------------------------------------- CLI ---


def test_parse_repos(tmp_path: Path) -> None:
    assert collect_pr_data.parse_repos("o/a, o/b ") == ["o/a", "o/b"]
    f = tmp_path / "repos.txt"
    f.write_text("o/a\no/b, o/c\n", encoding="utf-8")
    assert collect_pr_data.parse_repos(f"@{f}") == ["o/a", "o/b", "o/c"]
    with pytest.raises(ValueError):
        collect_pr_data.parse_repos("bad-repo")


def test_cli_end_to_end_on_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed = tmp_path / "processed"
    argv = [
        "collect-pr-data",
        "--input",
        str(SAMPLES_PATH),
        "--out",
        str(tmp_path / "raw"),
        "--processed",
        str(processed),
        "--seed",
        "42",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert collect_pr_data.main() == 0

    records: dict[str, list[TrainingRecord]] = {}
    for name in ("train", "val", "test"):
        p = processed / f"{name}.jsonl"
        assert p.exists(), name
        records[name] = format.read_jsonl(p)
        assert len(records[name]) > 0, f"{name} split is empty"

    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["ratios"] == {"train": 0.9, "val": 0.05, "test": 0.05}

    # All 12 samples assigned exactly once; no repo in two splits.
    all_ids = [r.meta["sample_id"] for rs in records.values() for r in rs]
    assert len(all_ids) == 12 and len(set(all_ids)) == 12
    repo_sets: dict[str, set[str]] = {}
    for name, rs in records.items():
        repo_sets[name] = {r.meta["repo"] for r in rs}
    assert repo_sets["train"].isdisjoint(repo_sets["val"])
    assert repo_sets["train"].isdisjoint(repo_sets["test"])
    assert repo_sets["val"].isdisjoint(repo_sets["test"])

    # Every assistant output validates against the Finding schema.
    for rs in records.values():
        for record in rs:
            findings = parse_findings(record.assistant)
            assert len(findings) == 1
            assert findings[0].line > 0
            assert record.meta["source"] in ("handcrafted", "heuristic")
