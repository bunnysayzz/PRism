"""
Export feedback to fine-tune training data.

Joins findings.jsonl with outcomes.jsonl and emits JSONL in prompt/completion
format for the LoRA retraining loop:

    {"prompt": "<system>...</system>\\n<user>...code context...</user>",
     "completion": "<assistant>{...findings JSON...}</assistant>"}

- accepted  -> positive example: reproduce the finding JSON.
- dismissed -> negative example: completion is {"findings": []}
               (teaches the model what NOT to flag).
- edited    -> positive example using the human-edited note as the finding
               text when provided.

Usage:
    python -m prism.feedback.export --dir feedback_data --out train.jsonl
    python -m prism.feedback.export --dir feedback_data --out train.jsonl --min-outcomes 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prism.feedback.store import FeedbackStore
from prism.logging import get_logger

log = get_logger(__name__)


def _prompt_for(record: dict[str, Any]) -> str:
    finding = record["finding"]
    return (
        "You are a senior staff engineer reviewing a pull request diff. "
        "Report only real, defensible issues as strict JSON.\n"
        f"File: {finding['path']}, line {finding['line']}\n"
        f"Category under review: {finding['category']}\n"
        f"Code context: {finding.get('explanation', '')[:500]}"
    )


def _completion_for(record: dict[str, Any], outcome: dict[str, Any]) -> str:
    kind = outcome["outcome"]
    if kind == "dismissed":
        return '{"findings": []}'
    finding = dict(record["finding"])
    if kind == "edited" and outcome.get("note"):
        finding["explanation"] = outcome["note"]
    return json.dumps({"findings": [finding]})


def export_training_data(store: FeedbackStore, min_outcomes: int = 1) -> list[dict[str, str]]:
    findings = store.load_findings()
    outcomes = store.load_outcomes()
    pairs: list[dict[str, str]] = []
    for record in findings:
        outcome = outcomes.get(record["finding_id"])
        if outcome is None:
            continue
        pairs.append(
            {
                "prompt": _prompt_for(record),
                "completion": _completion_for(record, outcome),
            }
        )
    if len(pairs) < min_outcomes:
        log.warning(
            "export_below_minimum",
            pairs=len(pairs),
            min_outcomes=min_outcomes,
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export feedback to fine-tune JSONL.")
    parser.add_argument("--dir", default="feedback_data", help="Feedback store directory")
    parser.add_argument("--out", default="train.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--min-outcomes",
        type=int,
        default=1,
        help="Warn if fewer labeled pairs than this",
    )
    args = parser.parse_args()

    store = FeedbackStore(args.dir)
    pairs = export_training_data(store, min_outcomes=args.min_outcomes)
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair) + "\n")
    print(f"exported {len(pairs)} training pairs -> {out_path}")


if __name__ == "__main__":
    main()
