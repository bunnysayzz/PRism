"""
Pydantic v2 schemas for review findings.

The model is instructed to output strict JSON matching the Finding schema;
`parse_findings()` is the tolerant entry point that validates model output.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Severity(StrEnum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"

    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]


class Finding(BaseModel):
    model_config = {"extra": "ignore"}  # tolerate extra keys from the model

    path: str = Field(description="Repo-relative file path")
    line: int = Field(gt=0, description="New-file line number in the PR head")
    severity: Severity
    category: str = Field(
        description="Issue category, e.g. security, correctness, performance, style"
    )
    title: str = Field(min_length=3, max_length=160)
    explanation: str = Field(min_length=10, description="Why this is a problem")
    suggestion: str = Field(default="", description="Fixed code snippet (may be empty)")
    confidence: float = Field(ge=0.0, le=1.0, description="Model self-reported confidence")
    language_hint: str = Field(default="", description="Code fence language for the suggestion")

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        # Be liberal: clamp out-of-range values instead of rejecting the finding.
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, f))

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v: Any) -> str:
        return str(v).strip().lower().replace(" ", "_") or "general"


class FileReview(BaseModel):
    path: str
    findings: list[Finding] = Field(default_factory=list)


class ReviewResult(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    model: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)


def parse_findings(raw: str | list[dict[str, Any]] | dict[str, Any]) -> list[Finding]:
    """
    Parse model output into validated Findings.

    Accepts a JSON string (possibly wrapped in markdown fences), a list of
    dicts, or {"findings": [...]}. Invalid entries are skipped, not fatal —
    one malformed finding must not sink the whole file review.
    """
    data = _extract_json(raw) if isinstance(raw, str) else raw
    items: list[Any]
    if isinstance(data, dict):
        items = data.get("findings", [])
    elif isinstance(data, list):
        items = data
    else:
        return []
    findings: list[Finding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            findings.append(Finding.model_validate(item))
        except Exception:
            continue  # skip malformed entries
    return findings


def _extract_json(text: str) -> Any:
    """Strip markdown fences and parse JSON; {} on failure."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove first fence line and trailing fence.
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    # Some models wrap JSON in prose — find the outermost object/array.
    start_obj, start_arr = cleaned.find("{"), cleaned.find("[")
    starts = [s for s in (start_obj, start_arr) if s != -1]
    if starts:
        start = min(starts)
        end_char = "}" if cleaned[start] == "{" else "]"
        end = cleaned.rfind(end_char)
        if end > start:
            cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
