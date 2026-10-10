"""PRism data pipeline: GitHub collection -> cleaning -> dedup -> splits -> SFT format.

Pipeline stages (each a module in this package):

    1. ``collector`` — async GitHub REST collection of PR diffs + inline review
       comments from public repos. Auth via ``GH_TOKEN``; rate-limit aware;
       idempotent via a checkpoint file.
    2. ``cleaner``   — filtering (bots, unanchored comments, huge diffs),
       language detection, unicode normalization, secret redaction.
    3. ``dedup``     — MinHash/LSH near-duplicate removal on
       (normalized diff + comment text), across repos.
    4. ``splits``    — deterministic, repo-stratified train/val/test splits
       (no repo appears in two splits) + manifest.
    5. ``format``    — SFT ``(system, user, assistant)`` triples where the
       assistant is strict JSON matching the ``Finding`` schema.

Entry point: ``scripts/collect_pr_data.py`` (CLI ``collect-pr-data``).

Shared record models live in ``prism.data.models``.
"""

from __future__ import annotations

from prism.data import cleaner, collector, dedup, format, models, splits
from prism.data.models import CleanSample, RawSample, TrainingRecord

__all__ = [
    "CleanSample",
    "RawSample",
    "TrainingRecord",
    "cleaner",
    "collector",
    "dedup",
    "format",
    "models",
    "splits",
]
