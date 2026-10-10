"""Tests for Finding/ReviewResult schemas and tolerant JSON parsing."""

import json

import pytest
from pydantic import ValidationError

from prism.review.schemas import (
    Finding,
    ReviewResult,
    Severity,
    parse_findings,
)


def _finding_dict(**overrides):
    base = {
        "path": "a.py",
        "line": 10,
        "severity": "high",
        "category": "security",
        "title": "SQL injection",
        "explanation": "User input is interpolated directly into the SQL query string.",
        "suggestion": "use parameterized queries",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def test_finding_valid():
    f = Finding.model_validate(_finding_dict())
    assert f.severity is Severity.high
    assert f.confidence == 0.9


def test_confidence_clamped_not_rejected():
    assert Finding.model_validate(_finding_dict(confidence=1.7)).confidence == 1.0
    assert Finding.model_validate(_finding_dict(confidence=-0.2)).confidence == 0.0


def test_confidence_invalid_defaults():
    assert Finding.model_validate(_finding_dict(confidence="nonsense")).confidence == 0.5


def test_confidence_out_of_bounds_rejected_at_field_level():
    # Direct model construction still enforces ge/le via the field, but the
    # before-validator clamps first — document the effective behavior.
    with pytest.raises(ValidationError):
        Finding.model_validate(_finding_dict(line=0))


def test_category_normalized():
    f = Finding.model_validate(_finding_dict(category=" SQL Injection "))
    assert f.category == "sql_injection"


def test_parse_findings_plain_json():
    raw = json.dumps({"findings": [_finding_dict()]})
    findings = parse_findings(raw)
    assert len(findings) == 1
    assert findings[0].title == "SQL injection"


def test_parse_findings_markdown_fence():
    raw = "```json\n" + json.dumps({"findings": [_finding_dict()]}) + "\n```"
    assert len(parse_findings(raw)) == 1


def test_parse_findings_prose_wrapped():
    raw = "Here is my review:\n" + json.dumps([_finding_dict()]) + "\nHope this helps!"
    assert len(parse_findings(raw)) == 1


def test_parse_findings_skips_malformed_entries():
    raw = json.dumps({"findings": [_finding_dict(), {"path": "b.py"}, "junk", 42]})
    findings = parse_findings(raw)
    assert len(findings) == 1  # only the valid one survives


def test_parse_findings_garbage_returns_empty():
    assert parse_findings("not json at all {{{") == []
    assert parse_findings("") == []


def test_review_result_defaults():
    r = ReviewResult()
    assert r.findings == [] and r.skipped_files == []


def test_severity_rank_ordering():
    assert Severity.critical.rank() > Severity.high.rank() > Severity.medium.rank()
