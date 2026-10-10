"""
Feedback store: JSONL log of every posted finding + human outcomes.

Every kept finding is appended to findings.jsonl with a stable finding_id.
Humans record outcomes (accepted / dismissed / edited) via record_outcome(),
which appends to outcomes.jsonl. The export module joins them into
fine-tune training pairs for the LoRA retraining loop.

Outcomes can be recorded from GitHub comment reactions / replies by a
follow-up job, or manually during early operation.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from prism.logging import get_logger
from prism.review.dedup import fingerprint
from prism.review.schemas import ReviewResult

log = get_logger(__name__)

Outcome = Literal["accepted", "dismissed", "edited"]


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class FeedbackStore:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.findings_path = self.dir / "findings.jsonl"
        self.outcomes_path = self.dir / "outcomes.jsonl"

    def log_review(self, owner: str, repo: str, pr_number: int, result: ReviewResult) -> list[str]:
        """Append one record per finding. Returns the assigned finding_ids."""
        ids: list[str] = []
        ts = _utcnow_iso()
        with self.findings_path.open("a", encoding="utf-8") as fh:
            for finding in result.findings:
                finding_id = str(uuid.uuid4())
                ids.append(finding_id)
                record = {
                    "finding_id": finding_id,
                    "fingerprint": fingerprint(finding),
                    "timestamp": ts,
                    "repo": f"{owner}/{repo}",
                    "pr_number": pr_number,
                    "model": result.model,
                    "finding": finding.model_dump(mode="json"),
                }
                fh.write(json.dumps(record) + "\n")
        log.info("feedback_logged", count=len(ids), repo=f"{owner}/{repo}", pr=pr_number)
        return ids

    def record_outcome(self, finding_id: str, outcome: Outcome, note: str = "") -> None:
        """Record a human outcome for a posted finding."""
        if outcome not in ("accepted", "dismissed", "edited"):
            raise ValueError(f"invalid outcome: {outcome!r}")
        record = {
            "finding_id": finding_id,
            "outcome": outcome,
            "note": note,
            "timestamp": _utcnow_iso(),
        }
        with self.outcomes_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info("feedback_outcome_recorded", finding_id=finding_id, outcome=outcome)

    def load_findings(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.findings_path)

    def load_outcomes(self) -> dict[str, dict[str, Any]]:
        """Map finding_id -> latest outcome record."""
        outcomes: dict[str, dict[str, Any]] = {}
        for record in _read_jsonl(self.outcomes_path):
            outcomes[record["finding_id"]] = record
        return outcomes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
