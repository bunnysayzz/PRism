#!/usr/bin/env python3
"""CLI: collect-pr-data — run the full PRism data pipeline.

Pipeline: collector -> cleaner -> dedup -> splits -> format.

Usage:
    # Live collection from public repos (needs GH_TOKEN), then full pipeline:
    python scripts/collect_pr_data.py --repos owner/a,owner/b \\
        --out data/raw --processed data/processed --max-prs 200

    # Re-run the pipeline from an existing raw.jsonl (no network, no token):
    python scripts/collect_pr_data.py --input data/raw/raw.jsonl \\
        --processed data/processed

    # Repo list from a file (one owner/name per line, or comma-separated):
    python scripts/collect_pr_data.py --repos @repos.txt --max-prs 200

Outputs (under --processed):
    train.jsonl / val.jsonl / test.jsonl — one TrainingRecord per line
    manifest.json — seed, ratios, per-split counts/repos/sample_ids

JSONL record formats
--------------------
data/raw/raw.jsonl — one RawSample per line (one per anchored review comment):
    {"sample_id": "owner/repo#123#src/x.py#c456", "repo": "owner/repo",
     "pr_number": 123, "pr_title": "Fix thing", "path": "src/x.py",
     "patch": "@@ -1,3 +1,4 @@\\n ...", "comment_id": 456,
     "comment_author": "octocat", "comment_author_type": "User",
     "comment_body": "This looks off because...", "comment_line": 42,
     "comment_original_line": 42, "comment_side": "RIGHT",
     "finding": null}

data/processed/{train,val,test}.jsonl — one TrainingRecord per line:
    {"system": "You are PRism, ...",
     "user": "Repository: owner/repo — PR #123: ...\\n```diff\\n...\\n```",
     "assistant": "{\\"findings\\": [{\\"path\\": ..., \\"line\\": ..., ...}]}",
     "meta": {"sample_id": "...", "repo": "owner/repo", "pr_number": 123,
              "path": "src/x.py", "language": "python", "comment_id": 456,
              "source": "heuristic|handcrafted"}}

data/processed/manifest.json:
    {"seed": 42, "ratios": {"train": 0.9, "val": 0.05, "test": 0.05},
     "splits": {"train": {"count": N, "repos": [...], "sample_ids": [...]}, ...}}

Environment:
    GH_TOKEN — required for live collection only (classic PAT with the
               `public_repo` scope, or a fine-grained token with read access
               to public repositories). Never pass it as a CLI flag.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running as `python scripts/collect_pr_data.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prism.data import cleaner, dedup, format, splits  # noqa: E402
from prism.data.cleaner import CleanerConfig  # noqa: E402
from prism.data.collector import (  # noqa: E402
    CollectorConfig,
    CollectorError,
    GitHubCollector,
    load_raw_samples,
)
from prism.logging import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)


def parse_repos(value: str | None) -> list[str]:
    """Parse --repos: comma-separated list or @file (one owner/name per line)."""
    if not value:
        return []
    if value.startswith("@"):
        text = Path(value[1:]).read_text(encoding="utf-8")
        parts = [p for line in text.splitlines() for p in line.split(",")]
    else:
        parts = value.split(",")
    repos = [p.strip() for p in parts if p.strip()]
    for repo in repos:
        segs = repo.split("/")
        if len(segs) != 2 or not all(s.strip() for s in segs):
            raise ValueError(f"invalid repo {repo!r}: expected 'owner/name'")
    return repos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect PR review data and build SFT train/val/test splits."
    )
    parser.add_argument(
        "--repos",
        default=None,
        help="Comma-separated 'owner/name' list, or @file with one per line.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Existing raw.jsonl to process instead of live collection (no GH_TOKEN needed).",
    )
    parser.add_argument("--out", default="data/raw", help="Raw output dir (default: data/raw).")
    parser.add_argument(
        "--processed",
        default="data/processed",
        help="Processed output dir for train/val/test.jsonl + manifest.json.",
    )
    parser.add_argument("--max-prs", type=int, default=200, help="Max PRs per repo.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file path (default: <out>/checkpoint.json).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for splits + dedup.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="MinHash Jaccard threshold for near-duplicates.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Drop SFT records above this estimated token count.",
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent PR fetches.")
    parser.add_argument(
        "--min-review-comments",
        type=int,
        default=1,
        help="Skip PRs with fewer inline review comments.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


async def _amain(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    processed_dir = Path(args.processed)
    checkpoint = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint.json"

    # -- 1. collect ------------------------------------------------------
    if args.input:
        raw_samples = load_raw_samples(args.input)
    else:
        repos = parse_repos(args.repos)
        if not repos:
            raise CollectorError("--repos is required for live collection (or use --input).")
        config = CollectorConfig(
            repos=repos,
            max_prs_per_repo=args.max_prs,
            out_dir=out_dir,
            checkpoint_path=checkpoint,
            concurrency=args.concurrency,
            min_review_comments=args.min_review_comments,
        )
        collector = GitHubCollector.from_env()
        stats = await collector.run(config)
        log.info("collection_stats", stats=vars(stats))
        raw_samples = load_raw_samples(out_dir / "raw.jsonl")

    # -- 2. clean ---------------------------------------------------------
    cleaned, drop_reasons = cleaner.clean_all(raw_samples, CleanerConfig())

    # -- 3. dedup ---------------------------------------------------------
    deduped, dropped_idx = dedup.dedupe(cleaned, threshold=args.dedup_threshold, seed=args.seed)

    # -- 4. split ---------------------------------------------------------
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    split_lists = splits.stratified_split(
        deduped,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        key=lambda s: s.repo,
        id_key=lambda s: s.sample_id,
    )

    # -- 5. format --------------------------------------------------------
    processed_dir.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    for name in ("train", "val", "test"):
        records = format.build_records(split_lists[name])
        records, too_long = format.filter_by_length(records, args.max_tokens)
        format.write_jsonl(processed_dir / f"{name}.jsonl", records)
        total_kept += len(records)
        log.info("split_written", split=name, records=len(records), too_long=len(too_long))

    manifest = splits.write_manifest(
        split_lists,
        out_path=processed_dir / "manifest.json",
        seed=args.seed,
        ratios=ratios,
        key=lambda s: s.repo,
        id_key=lambda s: s.sample_id,
    )

    summary = {
        "raw_samples": len(raw_samples),
        "cleaned": len(cleaned),
        "drop_reasons": dict(drop_reasons),
        "deduped": len(deduped),
        "dedup_dropped": len(dropped_idx),
        "splits": {name: manifest["splits"][name]["count"] for name in ("train", "val", "test")},
        "records_written": total_kept,
        "processed_dir": str(processed_dir),
    }
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return asyncio.run(_amain(args))
    except (CollectorError, ValueError) as exc:
        log.error("pipeline_failed", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
