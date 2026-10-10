"""
Eval metric library for PRism.

Pure functions for scoring review quality against golden cases. No I/O, no
network, no backend calls — everything here is deterministic math over
findings, which makes it trivially unit-testable.

Matching
--------
A predicted finding matches an expected issue iff the categories are equal
and ``abs(predicted.line - expected.line) <= line_tolerance`` (default 2).
Each expected issue is matched at most once (greedy, in prediction order),
so duplicate predictions of the same issue count as one TP + extra FPs.

Classification
--------------
precision / recall / F1, plus per-category and per-severity breakdowns and a
severity-weighted F1. Conventions mirror ``prism.eval.runner``:

- no predictions        -> precision = 1.0 (nothing wrong was said)
- no expected issues    -> recall    = 1.0 (nothing was missed)

False positives
---------------
Reported as ``false_positive_rate = FP / (TP + FP)`` — the fraction of
reported findings that were wrong (a.k.a. false discovery rate). A strict
false-positive rate ``FP / (FP + TN)`` is undefined here because the
population of "true negatives" (every line the reviewer stayed silent on)
is not a meaningful denominator for open-ended detection, so we document
the convention explicitly and also report raw FP-per-case.

Calibration
-----------
Expected calibration error (ECE) of the model's confidence scores: bucket
predictions by confidence, compare mean predicted confidence against
empirical precision in each bucket, and take the count-weighted mean of the
absolute gaps. Well-calibrated confidence (0.9 means right ~90% of the
time) is what makes the confidence gate in the review engine trustworthy.

Latency
-------
Mean / min / max / p50 / p95 over per-case pipeline wall times (linear
interpolation for percentiles).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.review.schemas import Finding

#: Severity weights shared with the golden runner's weighted F1.
SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}

#: Default line tolerance for matching predicted vs expected findings.
LINE_TOLERANCE_DEFAULT = 2


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedIssue:
    """Ground-truth issue from a golden case."""

    line: int
    category: str
    severity: str


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching predictions against expected issues."""

    predicted_is_tp: tuple[bool, ...]
    """Per predicted finding: True if it matched an expected issue."""
    expected_matched: tuple[bool, ...]
    """Per expected issue: True if some prediction matched it."""
    pairs: tuple[tuple[int, int], ...]
    """(predicted_index, expected_index) for each TP."""


@dataclass(frozen=True)
class ClassificationMetrics:
    """Precision/recall/F1 triple with the raw counts behind it."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class CalibrationBin:
    """One confidence bucket for the calibration table."""

    bin_start: float
    bin_end: float
    count: int
    mean_confidence: float
    empirical_precision: float


@dataclass(frozen=True)
class LatencyStats:
    """Summary of per-case pipeline latencies (seconds)."""

    count: int
    mean: float
    min: float
    max: float
    p50: float
    p95: float


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_findings(
    predicted: Sequence[Finding],
    expected: Sequence[ExpectedIssue],
    line_tolerance: int = LINE_TOLERANCE_DEFAULT,
) -> MatchResult:
    """
    Match predicted findings against expected issues.

    A prediction matches an expectation iff categories are equal and the
    lines are within ``line_tolerance``. Each expected issue matches at most
    one prediction (greedy, prediction order).
    """
    predicted_is_tp: list[bool] = []
    expected_matched = [False] * len(expected)
    pairs: list[tuple[int, int]] = []
    for pi, finding in enumerate(predicted):
        hit: int | None = None
        for ei, exp in enumerate(expected):
            if expected_matched[ei]:
                continue
            if exp.category == finding.category and abs(exp.line - finding.line) <= line_tolerance:
                hit = ei
                break
        if hit is not None:
            expected_matched[hit] = True
            predicted_is_tp.append(True)
            pairs.append((pi, hit))
        else:
            predicted_is_tp.append(False)
    return MatchResult(
        predicted_is_tp=tuple(predicted_is_tp),
        expected_matched=tuple(expected_matched),
        pairs=tuple(pairs),
    )


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """
    Precision, recall, F1 from counts.

    Empty denominators follow the runner convention: precision is 1.0 when
    nothing was predicted, recall is 1.0 when nothing was expected, and F1
    is 0.0 when both are 0.0.
    """
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def classification_metrics(tp: int, fp: int, fn: int) -> ClassificationMetrics:
    """Bundle raw counts with their precision/recall/F1."""
    precision, recall, f1 = prf(tp, fp, fn)
    return ClassificationMetrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def _counts_from_match(
    predicted: Sequence[Finding],
    expected: Sequence[ExpectedIssue],
    match: MatchResult,
    key: str,
) -> dict[str, ClassificationMetrics]:
    """
    Per-group P/R/F1 where ``key`` is "category" or "severity".

    TP/FP are attributed by the *predicted* finding's group; FN by the
    *expected* issue's group. (For TPs both agree by the matching rule.)
    """
    groups: dict[str, list[int]] = {}  # group -> [tp, fp, fn]
    for pi, finding in enumerate(predicted):
        group = getattr(finding, key)
        group = group.value if hasattr(group, "value") else str(group)
        counts = groups.setdefault(group, [0, 0, 0])
        counts[0 if match.predicted_is_tp[pi] else 1] += 1
    for ei, exp in enumerate(expected):
        if not match.expected_matched[ei]:
            counts = groups.setdefault(getattr(exp, key), [0, 0, 0])
            counts[2] += 1
    return {
        group: classification_metrics(tp, fp, fn) for group, (tp, fp, fn) in sorted(groups.items())
    }


def per_category_metrics(
    predicted: Sequence[Finding],
    expected: Sequence[ExpectedIssue],
    match: MatchResult,
) -> dict[str, ClassificationMetrics]:
    """Precision/recall/F1 broken down by finding category."""
    return _counts_from_match(predicted, expected, match, "category")


def per_severity_metrics(
    predicted: Sequence[Finding],
    expected: Sequence[ExpectedIssue],
    match: MatchResult,
) -> dict[str, ClassificationMetrics]:
    """Precision/recall/F1 broken down by severity."""
    return _counts_from_match(predicted, expected, match, "severity")


def severity_weighted_prf(
    predicted: Sequence[Finding],
    expected: Sequence[ExpectedIssue],
    match: MatchResult,
    weights: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    """
    Severity-weighted precision/recall/F1.

    TP/FP mass comes from the predicted finding's severity weight; FN mass
    from the missed expected issue's severity weight. Missing a critical
    hurts ~4x more than missing a low.
    """
    weights = weights if weights is not None else SEVERITY_WEIGHTS

    def _w(sev: object) -> float:
        name = sev.value if hasattr(sev, "value") else str(sev)
        return weights.get(name, 1.0)

    zipped = zip(predicted, match.predicted_is_tp, strict=True)
    wtp = sum(_w(f.severity) for f, is_tp in zipped if is_tp)
    wfp = sum(
        _w(f.severity)
        for f, is_tp in zip(predicted, match.predicted_is_tp, strict=True)
        if not is_tp
    )
    wfn = sum(
        _w(e.severity) for e, hit in zip(expected, match.expected_matched, strict=True) if not hit
    )
    return prf(wtp, wfp, wfn)


# ---------------------------------------------------------------------------
# False positives
# ---------------------------------------------------------------------------


def false_positive_rate(tp: int, fp: int) -> float:
    """
    Fraction of reported findings that were wrong: FP / (TP + FP).

    (Strictly this is the *false discovery rate*. A textbook FPR needs a
    true-negative population, which is undefined for open-ended detection —
    see the module docstring. 0.0 when nothing was predicted.)
    """
    return fp / (tp + fp) if (tp + fp) else 0.0


def false_positives_per_case(total_fp: int, n_cases: int) -> float:
    """Mean false positives per golden case. 0.0 when there are no cases."""
    return total_fp / n_cases if n_cases else 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """
    Bucket predictions by confidence; report empirical precision per bucket.

    ``confidences[i]`` is the model's reported confidence for prediction i,
    ``correct[i]`` whether it was a true positive. Buckets are equal-width
    ``[b/n, (b+1)/n)``; the last bucket includes 1.0.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: list[CalibrationBin] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        members = [
            i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == n_bins - 1 and c == hi)
        ]
        if members:
            mean_conf = sum(confidences[i] for i in members) / len(members)
            precision = sum(1 for i in members if correct[i]) / len(members)
        else:
            mean_conf, precision = 0.0, 0.0
        bins.append(
            CalibrationBin(
                bin_start=lo,
                bin_end=hi,
                count=len(members),
                mean_confidence=mean_conf,
                empirical_precision=precision,
            )
        )
    return bins


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """
    Expected calibration error: count-weighted mean |precision - confidence|.

    0.0 is perfect calibration; 1.0 is maximally wrong. Empty input -> 0.0.
    """
    total = len(confidences)
    if total == 0:
        return 0.0
    return sum(
        (b.count / total) * abs(b.empirical_precision - b.mean_confidence)
        for b in calibration_bins(confidences, correct, n_bins)
    )


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def latency_stats(latencies: Sequence[float]) -> LatencyStats:
    """Mean/min/max/p50/p95 over per-case latencies (seconds)."""
    vals = sorted(latencies)
    count = len(vals)
    return LatencyStats(
        count=count,
        mean=(sum(vals) / count) if count else 0.0,
        min=vals[0] if vals else 0.0,
        max=vals[-1] if vals else 0.0,
        p50=_percentile(vals, 50),
        p95=_percentile(vals, 95),
    )


@dataclass(frozen=True)
class AggregateReport:
    """All aggregate metrics for one backend run (candidate or baseline)."""

    n_cases: int
    micro: ClassificationMetrics = field(default_factory=lambda: classification_metrics(0, 0, 0))
    weighted_f1: float = 0.0
    per_category: dict[str, ClassificationMetrics] = field(default_factory=dict)
    per_severity: dict[str, ClassificationMetrics] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    false_positives_per_case: float = 0.0
    calibration_error: float = 0.0
    calibration_table: list[CalibrationBin] = field(default_factory=list)
    latency: LatencyStats = field(default_factory=lambda: latency_stats([]))
