"""
Near-duplicate detection across repos with MinHash + LSH.

Duplicate key: ``(normalized diff patch + normalized comment body)`` — the
repo is deliberately excluded so the same review copied across forks (or the
same snippet reviewed twice) collapses to one sample.

Implementation is stdlib-only:
  - word k-shingles hashed with SHA-256,
  - MinHash signatures via the ``(a + i*b) mod 2**61-1`` permutation trick,
  - LSH banding for candidate generation, then exact estimated-Jaccard
    verification against ``threshold``.

``dedupe()`` is the main entry point: it keeps the first occurrence of each
duplicate group and returns the dropped original indices.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from prism.data.models import CleanSample
from prism.logging import get_logger

log = get_logger(__name__)

_MOD = (1 << 61) - 1  # Mersenne prime for the permutation trick
_SHINGLE_K = 5


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization for duplicate comparison: lowercase + collapse ws."""
    return re.sub(r"\s+", " ", text).strip().lower()


def dedup_key(sample: CleanSample) -> str:
    """The text two samples are compared on: normalized patch + comment body."""
    return f"{normalize_for_dedup(sample.patch)}\n@@@\n{normalize_for_dedup(sample.comment_body)}"


def _shingles(text: str, k: int = _SHINGLE_K) -> set[bytes]:
    tokens = text.split()
    if not tokens:
        return set()
    if len(tokens) < k:
        return {text.encode("utf-8")}
    return {" ".join(tokens[i : i + k]).encode("utf-8") for i in range(len(tokens) - k + 1)}


def minhash_signature(shingles: set[bytes], num_perm: int = 128, seed: int = 42) -> tuple[int, ...]:
    """
    MinHash signature of a shingle set.

    For each shingle, derive two 64-bit hashes (a, b) from SHA-256 and use
    ``h_i = (a + i*b) mod 2**61-1`` as the i-th permutation hash. The
    signature entry is the minimum over shingles. Empty input -> all zeros.
    """
    sig = [_MOD] * num_perm
    for shingle in shingles:
        digest = hashlib.sha256(shingle).digest()
        a = (int.from_bytes(digest[:8], "big") + seed) % _MOD
        b = (int.from_bytes(digest[8:16], "big") % (_MOD - 1)) + 1
        for i in range(num_perm):
            h = (a + i * b) % _MOD
            if h < sig[i]:
                sig[i] = h
    if not shingles:
        return tuple([0] * num_perm)
    return tuple(sig)


def estimate_jaccard(sig1: tuple[int, ...], sig2: tuple[int, ...]) -> float:
    """Fraction of equal signature entries ~= Jaccard similarity of shingle sets."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2, strict=True) if a == b)
    return matches / len(sig1)


class LSHIndex:
    """Locality-sensitive hashing index over MinHash signatures."""

    def __init__(self, num_perm: int = 128, bands: int = 16) -> None:
        if num_perm % bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self._buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    def add(self, idx: int, signature: tuple[int, ...]) -> None:
        for b in range(self.bands):
            band = tuple(signature[b * self.rows : (b + 1) * self.rows])
            self._buckets.setdefault((b, band), []).append(idx)

    def candidate_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for bucket in self._buckets.values():
            if len(bucket) < 2:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    pairs.add((a, b) if a < b else (b, a))
        return pairs


def find_duplicate_pairs(
    samples: Sequence[CleanSample],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
    bands: int = 16,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """
    Find near-duplicate sample index pairs (estimated Jaccard >= threshold).

    LSH generates candidates; every candidate pair is verified with the
    MinHash estimated Jaccard before being reported.
    """
    signatures = [
        minhash_signature(_shingles(dedup_key(s)), num_perm=num_perm, seed=seed) for s in samples
    ]
    index = LSHIndex(num_perm=num_perm, bands=bands)
    for idx, sig in enumerate(signatures):
        index.add(idx, sig)
    pairs = [
        (i, j)
        for i, j in sorted(index.candidate_pairs())
        if estimate_jaccard(signatures[i], signatures[j]) >= threshold
    ]
    log.info("duplicate_pairs_found", pairs=len(pairs), samples=len(samples))
    return pairs


def _duplicate_groups(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find grouping of duplicate pairs into connected components."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def dedupe(
    samples: list[CleanSample],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
    bands: int = 16,
    seed: int = 42,
) -> tuple[list[CleanSample], list[int]]:
    """
    Remove near-duplicates, keeping the first occurrence of each group.

    Returns ``(kept_samples, dropped_indices)`` with dropped indices sorted.
    """
    if len(samples) < 2:
        return list(samples), []
    pairs = find_duplicate_pairs(
        samples, threshold=threshold, num_perm=num_perm, bands=bands, seed=seed
    )
    dropped: set[int] = set()
    for group in _duplicate_groups(len(samples), pairs):
        dropped.update(group[1:])  # keep group[0] (first occurrence)
    kept = [s for i, s in enumerate(samples) if i not in dropped]
    log.info("dedupe_done", kept=len(kept), dropped=len(dropped))
    return kept, sorted(dropped)
