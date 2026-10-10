"""
Dedup: avoid re-posting findings that are already on the PR.

fingerprint(finding) = sha256(path + normalized context + category).
dedupe_against_existing() drops findings whose fingerprint already appears in
the bodies of existing bot comments (the fingerprint is embedded in a hidden
HTML comment when rendering, so exact re-detection is cheap).
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.review.schemas import Finding

_FINGERPRINT_RE = re.compile(r"<!-- PRism fp:([0-9a-f]{16}) -->")


def _normalize(text: str) -> str:
    # Collapse whitespace; case-insensitive; strip code fences.
    text = re.sub(r"```\w*\n?", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def fingerprint(finding: Finding) -> str:
    """
    Stable 16-hex-char fingerprint of a finding.

    Uses path + normalized (title + category + explanation head) — NOT the
    line number, so a finding that shifts by a few lines after a rebase still
    dedups instead of double-posting.
    """
    basis = "|".join(
        [
            finding.path.strip().lower(),
            finding.category.strip().lower(),
            _normalize(finding.title),
            _normalize(finding.explanation)[:200],
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def embed_fingerprint(body: str, fp: str) -> str:
    """Embed the fingerprint as a hidden HTML comment in the comment body."""
    return f"{body}\n<!-- PRism fp:{fp} -->"


def extract_fingerprints(comment_body: str) -> set[str]:
    return set(_FINGERPRINT_RE.findall(comment_body or ""))


def dedupe_against_existing(
    findings: list[Finding], existing_comments: list[dict[str, object]]
) -> list[Finding]:
    """Drop findings already posted (by fingerprint match in existing bodies)."""
    seen: set[str] = set()
    for comment in existing_comments:
        body = comment.get("body") if isinstance(comment, dict) else None
        seen |= extract_fingerprints(str(body or ""))
    kept = [f for f in findings if fingerprint(f) not in seen]
    return kept
