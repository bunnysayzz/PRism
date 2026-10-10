"""
Golden dataset: format spec, validation, and growth plan.

Format (one JSON object per line in ``eval/golden.jsonl``)::

    {
      "id": "sec-sqli-fstring",            # unique, stable across edits
      "path": "app/users.py",             # repo-relative file path
      "patch": "@@ -40,4 +40,8 @@ ...",    # unified diff; findings anchor here
      "expected": [                       # ground-truth issues (may be [])
        {"line": 42, "category": "security", "severity": "critical"}
      ],
      "stub_findings": [ ... ]            # optional: full Finding dicts the
                                          # stub backend replays for this case
    }

Rules:

- ``line`` is a *new-file* line number and must be commentable, i.e. part of
  the diff (added or context lines — see ``prism.diff.hunks``).
- ``category`` comes from ``CATEGORIES``; ``severity`` from ``SEVERITIES``.
- ``stub_findings`` entries must validate as ``prism.review.schemas.Finding``;
  the stub backend replays them verbatim through the real pipeline
  (confidence gate + line validation), so give them realistic confidences
  (>= the auto-post threshold if the finding should survive the gate).

Growing the set toward 500 cases
--------------------------------
The 30-case seed set is handcrafted for correctness. Scale it with the
production feedback loop instead of hand-writing 470 more:

1. Mine ``FeedbackStore``: every reviewed PR logs findings; the second-pass
   verifier verdicts and any human reactions (dismissed / confirmed) label
   them. Verified true positives become positive cases; verifier-rejected or
   human-dismissed findings become *negative* cases (``"expected": []``) —
   the set must measure precision, not just recall.
2. Stratify by category using ``TARGET_DISTRIBUTION`` so rare classes
   (concurrency, security) are not drowned by style nits.
3. Deduplicate by normalized patch hash (strip whitespace/comments) — near
   duplicates teach the harness nothing and inflate scores.
4. Keep the human audit: every mined case gets a one-line human sign-off
   before it lands; re-audit a random 5% sample quarterly for label drift.
5. Version the file (``eval/golden.v2.jsonl``) when the distribution changes
   materially so old reports stay comparable.

``validate_golden()`` is the gatekeeper for both handcrafted and mined
cases: schema, vocabulary, line-in-patch, unique ids, valid stub findings.
"""

from __future__ import annotations

import json
from pathlib import Path

from prism.diff.hunks import commentable_lines
from prism.review.schemas import Finding

#: Fixed finding-category vocabulary for golden cases.
CATEGORIES: tuple[str, ...] = (
    "security",
    "correctness",
    "performance",
    "concurrency",
    "style",
    "maintainability",
)

#: Fixed severity vocabulary for golden cases.
SEVERITIES: tuple[str, ...] = ("critical", "high", "medium", "low")

#: Target category mix when growing toward the 500-case benchmark.
TARGET_DISTRIBUTION: dict[str, int] = {
    "security": 110,
    "correctness": 150,
    "performance": 70,
    "concurrency": 60,
    "style": 60,
    "maintainability": 50,
}

_REQUIRED_KEYS = ("id", "path", "patch", "expected")


def validate_golden(path: str | Path) -> list[str]:
    """
    Validate a golden JSONL file. Returns a list of error strings; empty
    means the file is valid.

    Checks: valid JSON per line, required keys, unique non-empty ids,
    non-empty path/patch, parseable hunks, expected entries with a positive
    int line inside the patch's commentable lines and vocabulary
    category/severity, and well-formed ``stub_findings`` when present.
    """
    golden_path = Path(path)
    if not golden_path.is_file():
        return [f"golden file not found: {golden_path}"]
    errors: list[str] = []
    seen_ids: set[str] = set()
    for lineno, raw in enumerate(golden_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        tag = f"line {lineno}"
        try:
            case = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"{tag}: invalid JSON: {exc}")
            continue
        if not isinstance(case, dict):
            errors.append(f"{tag}: case must be a JSON object")
            continue
        errors.extend(_validate_case(case, tag, seen_ids))
    return errors


def _validate_case(case: dict[str, object], tag: str, seen_ids: set[str]) -> list[str]:
    errors: list[str] = []
    for key in _REQUIRED_KEYS:
        if key not in case:
            errors.append(f"{tag}: missing required key {key!r}")
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append(f"{tag}: 'id' must be a non-empty string")
    elif case_id in seen_ids:
        errors.append(f"{tag}: duplicate id {case_id!r}")
    else:
        seen_ids.add(case_id)
    path = case.get("path")
    if not isinstance(path, str) or not path.strip():
        errors.append(f"{tag}: 'path' must be a non-empty string")
    patch = case.get("patch")
    commentable: set[int] = set()
    if not isinstance(patch, str) or not patch.strip():
        errors.append(f"{tag}: 'patch' must be a non-empty string")
    else:
        commentable = commentable_lines(patch)
        if not commentable:
            errors.append(f"{tag}: 'patch' has no parseable hunks")
    expected = case.get("expected")
    if not isinstance(expected, list):
        errors.append(f"{tag}: 'expected' must be a list")
    else:
        for i, item in enumerate(expected):
            errors.extend(_validate_expected(item, f"{tag} expected[{i}]", commentable))
    stub_findings = case.get("stub_findings", [])
    if not isinstance(stub_findings, list):
        errors.append(f"{tag}: 'stub_findings' must be a list")
    else:
        for i, item in enumerate(stub_findings):
            if not isinstance(item, dict):
                errors.append(f"{tag}: stub_findings[{i}] must be an object")
                continue
            try:
                Finding.model_validate(item)
            except Exception as exc:  # noqa: BLE001 — report, don't raise
                errors.append(f"{tag}: stub_findings[{i}] invalid Finding: {exc}")
    return errors


def _validate_expected(item: object, tag: str, commentable: set[int]) -> list[str]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return [f"{tag}: must be an object"]
    line = item.get("line")
    if not isinstance(line, int) or isinstance(line, bool) or line <= 0:
        errors.append(f"{tag}: 'line' must be a positive integer")
    elif commentable and line not in commentable:
        errors.append(f"{tag}: line {line} is not commentable in the patch")
    category = item.get("category")
    if category not in CATEGORIES:
        errors.append(f"{tag}: 'category' must be one of {list(CATEGORIES)}")
    severity = item.get("severity")
    if severity not in SEVERITIES:
        errors.append(f"{tag}: 'severity' must be one of {list(SEVERITIES)}")
    return errors
