"""Tests for prism.feedback.export: the feedback-to-training-data exporter."""

from __future__ import annotations

import json
from typing import Any

from prism.feedback.export import export_training_data


class _FakeStore:
    """Minimal FeedbackStore stand-in: only the two methods export needs."""

    def __init__(
        self,
        findings: list[dict[str, Any]],
        outcomes: dict[str, dict[str, Any]],
    ) -> None:
        self._findings = findings
        self._outcomes = outcomes

    def load_findings(self) -> list[dict[str, Any]]:
        return self._findings

    def load_outcomes(self) -> dict[str, dict[str, Any]]:
        return self._outcomes


def _finding(fid: str) -> dict[str, Any]:
    return {
        "finding_id": fid,
        "finding": {
            "path": "src/app.py",
            "line": 42,
            "category": "security",
            "explanation": "SQL injection via f-string",
        },
    }


def test_accepted_becomes_positive_example() -> None:
    store = _FakeStore([_finding("f1")], {"f1": {"outcome": "accepted"}})
    pairs = export_training_data(store)
    assert len(pairs) == 1
    completion = json.loads(pairs[0]["completion"])
    assert completion["findings"][0]["path"] == "src/app.py"


def test_dismissed_becomes_empty_findings() -> None:
    store = _FakeStore([_finding("f1")], {"f1": {"outcome": "dismissed"}})
    pairs = export_training_data(store)
    assert len(pairs) == 1
    assert pairs[0]["completion"] == '{"findings": []}'


def test_edited_uses_human_note_as_explanation() -> None:
    store = _FakeStore(
        [_finding("f1")],
        {"f1": {"outcome": "edited", "note": "human corrected text"}},
    )
    pairs = export_training_data(store)
    completion = json.loads(pairs[0]["completion"])
    assert completion["findings"][0]["explanation"] == "human corrected text"


def test_edited_without_note_keeps_original_explanation() -> None:
    store = _FakeStore([_finding("f1")], {"f1": {"outcome": "edited"}})
    pairs = export_training_data(store)
    completion = json.loads(pairs[0]["completion"])
    assert completion["findings"][0]["explanation"] == "SQL injection via f-string"


def test_findings_without_outcome_are_skipped() -> None:
    store = _FakeStore([_finding("f1"), _finding("f2")], {"f1": {"outcome": "accepted"}})
    pairs = export_training_data(store)
    assert len(pairs) == 1


def test_prompt_contains_file_line_and_category() -> None:
    store = _FakeStore([_finding("f1")], {"f1": {"outcome": "accepted"}})
    prompt = export_training_data(store)[0]["prompt"]
    assert "src/app.py" in prompt
    assert "42" in prompt
    assert "security" in prompt


def test_below_minimum_warns_but_still_returns_pairs(capsys: Any) -> None:
    # prism.logging is structlog: the warning is emitted as JSON on stdout,
    # not through stdlib logging, so caplog cannot capture it.
    store = _FakeStore([_finding("f1")], {"f1": {"outcome": "accepted"}})
    pairs = export_training_data(store, min_outcomes=50)
    assert len(pairs) == 1
    assert "export_below_minimum" in capsys.readouterr().out
