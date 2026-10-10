"""
Deterministic, repo-stratified train/val/test splits.

Stratification is by repository: a repo's samples always land in a single
split, so the model can never train on one PR of a repo and be evaluated on
another (repo leakage). Assignment is deterministic for a given seed: repos
are shuffled with ``random.Random(seed)`` and then greedily assigned —
whole repos go to train until the train target fraction is reached, with the
last two shuffled repos reserved for val/test so every split is non-empty
whenever there are >= 3 repos.

``write_manifest()`` records the seed, ratios, per-split counts, repos and
sample ids so a split can be audited or reproduced later.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from prism.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

SPLIT_NAMES = ("train", "val", "test")


def stratified_split(
    samples: list[T],
    *,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    key: Callable[[T], str],
    id_key: Callable[[T], str] | None = None,
) -> dict[str, list[T]]:
    """
    Split samples into train/val/test with no repo in more than one split.

    ``key`` extracts the repo (stratification group) from a sample; ``id_key``
    is only used for logging. Raises ValueError on invalid ratios.
    """
    for name, ratio in (("train", train_ratio), ("val", val_ratio), ("test", test_ratio)):
        if ratio <= 0:
            raise ValueError(f"{name}_ratio must be positive, got {ratio}")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    by_repo: dict[str, list[T]] = {}
    for sample in samples:
        by_repo.setdefault(key(sample), []).append(sample)

    repos = list(by_repo)
    rng = random.Random(seed)
    rng.shuffle(repos)

    splits: dict[str, list[T]] = {"train": [], "val": [], "test": []}
    n_repos = len(repos)
    if n_repos == 0:
        return splits
    if n_repos == 1:
        splits["train"] = list(by_repo[repos[0]])
        return splits
    if n_repos == 2:
        splits["train"] = list(by_repo[repos[0]])
        splits["test"] = list(by_repo[repos[1]])
        return splits

    # Reserve the last two shuffled repos so val/test are never empty.
    val_repos = [repos[-2]]
    test_repos = [repos[-1]]
    candidates = repos[:-2]

    total = len(samples)
    target_train = train_ratio * total
    train_repos: list[str] = []
    leftover: list[str] = []
    train_count = 0
    for repo in candidates:
        if train_count < target_train:
            train_repos.append(repo)
            train_count += len(by_repo[repo])
        else:
            leftover.append(repo)

    # Overflow repos go to whichever of val/test is furthest below its target.
    val_target = val_ratio * total
    test_target = test_ratio * total
    val_count = len(by_repo[val_repos[0]])
    test_count = len(by_repo[test_repos[0]])
    for repo in leftover:
        if (val_target - val_count) >= (test_target - test_count):
            val_repos.append(repo)
            val_count += len(by_repo[repo])
        else:
            test_repos.append(repo)
            test_count += len(by_repo[repo])

    for repo in train_repos:
        splits["train"].extend(by_repo[repo])
    for repo in val_repos:
        splits["val"].extend(by_repo[repo])
    for repo in test_repos:
        splits["test"].extend(by_repo[repo])

    log.info(
        "split_done",
        seed=seed,
        train=len(splits["train"]),
        val=len(splits["val"]),
        test=len(splits["test"]),
        train_repos=len(train_repos),
    )
    return splits


def repo_sets(splits: dict[str, list[T]], *, key: Callable[[T], str]) -> dict[str, set[str]]:
    """Map split name -> set of repos it contains (for leakage checks)."""
    return {name: {key(s) for s in items} for name, items in splits.items()}


def write_manifest(
    splits: dict[str, list[T]],
    *,
    out_path: str | Path,
    seed: int,
    ratios: dict[str, float],
    key: Callable[[T], str],
    id_key: Callable[[T], str] | None = None,
) -> dict[str, Any]:
    """Write a JSON manifest describing the split; returns the manifest dict."""
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "ratios": ratios,
        "splits": {},
    }
    for name in SPLIT_NAMES:
        items = splits.get(name, [])
        entry: dict[str, Any] = {
            "count": len(items),
            "repos": sorted({key(s) for s in items}),
        }
        if id_key is not None:
            entry["sample_ids"] = [id_key(s) for s in items]
        manifest["splits"][name] = entry
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("manifest_written", path=str(path))
    return manifest
