"""
Golden eval runner.

Loads eval/golden.jsonl — one JSON object per line:
    {"path": ..., "patch": ..., "stub_findings": [...], "expected": [...]}

- stub_findings: what the model under test would output (Finding-shaped dicts).
- expected: ground-truth issues [{"line":..,"category":..,"severity":..}].

Pipeline per case (real engine code paths, no GitHub calls):
  1. backend.analyze()  (--backend stub replays stub_findings; vllm/api/hf_endpoint
     run a real model and ignore stub_findings)
  2. apply_confidence_gate() with a verdict stub (or the real verifier backend)
  3. validate_finding_lines() against commentable_lines(patch)

Matching: predicted matches expected iff category equal and |line diff| <= 2.
Metrics: precision, recall, F1, severity-weighted F1.
Exits non-zero when F1 < --min-f1 (this is the CI eval gate).

Usage:
    python -m prism.eval.runner --golden eval/golden.example.jsonl
    python -m prism.eval.runner --golden eval/golden.jsonl --backend vllm --min-f1 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism.config import get_settings
from prism.diff.hunks import commentable_lines
from prism.eval.metrics import SEVERITY_WEIGHTS as _SEVERITY_WEIGHT
from prism.logging import get_logger, setup_logging
from prism.review import prompts
from prism.review.backends import (
    ReviewBackend,
    StubBackend,
    build_backend,
)
from prism.review.engine import apply_confidence_gate, validate_finding_lines
from prism.review.schemas import Finding

log = get_logger(__name__)

_LINE_TOLERANCE = 2


@dataclass
class Expected:
    line: int
    category: str
    severity: str


@dataclass
class CaseResult:
    name: str
    tp: int
    fp: int
    fn: int
    wtp: float
    wfp: float
    wfn: float
    predicted: int
    expected: int


class VerdictStubBackend(ReviewBackend):
    """
    Deterministic verifier for eval: verdicts keyed by finding title.

    verdicts: {title_substring: (is_real, revised_confidence)}.
    Unknown titles -> ("false_positive", 0.0) — fail closed, like production.
    """

    name = "verdict-stub"

    def __init__(self, verdicts: dict[str, tuple[bool, float]] | None = None) -> None:
        self.verdicts = verdicts or {}

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        lowered = user_prompt.lower()
        for title, (is_real, conf) in self.verdicts.items():
            if title.lower() in lowered:
                verdict = "real" if is_real else "false_positive"
                return (
                    json.dumps(
                        {
                            "verdict": verdict,
                            "revised_confidence": conf,
                            "reason": "stub verdict",
                        }
                    ),
                    {"backend": self.name},
                )
        return (
            json.dumps(
                {"verdict": "false_positive", "revised_confidence": 0.0, "reason": "unknown"}
            ),
            {"backend": self.name},
        )


def _match(
    predicted: list[Finding], expected: list[Expected]
) -> tuple[int, int, int, float, float, float]:
    matched_expected: set[int] = set()
    tp = 0
    wtp = 0.0
    wfp = 0.0
    for finding in predicted:
        hit: int | None = None
        for i, exp in enumerate(expected):
            if i in matched_expected:
                continue
            if exp.category == finding.category and abs(exp.line - finding.line) <= _LINE_TOLERANCE:
                hit = i
                break
        weight = _SEVERITY_WEIGHT.get(finding.severity.value, 1.0)
        if hit is not None:
            matched_expected.add(hit)
            tp += 1
            wtp += weight
        else:
            wfp += weight
    fn = len(expected) - len(matched_expected)
    wfn = sum(
        _SEVERITY_WEIGHT.get(e.severity, 1.0)
        for i, e in enumerate(expected)
        if i not in matched_expected
    )
    fp = len(predicted) - tp
    return tp, fp, fn, wtp, wfp, wfn


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


async def run_case_pipeline(
    case: dict[str, Any],
    primary: ReviewBackend,
    verifier: ReviewBackend,
    settings: Any,
) -> list[Finding]:
    """
    Run the real review pipeline for one golden case and return kept findings.

    Shared by the golden runner and the benchmark harness: backend.analyze()
    -> confidence gate -> line validation against the patch.
    """
    path = case["path"]
    patch = case["patch"]

    user_prompt = prompts.build_user_prompt(
        path=path,
        language=prompts.guess_language(path),
        hunk_context=patch,
    )
    findings, _ = await primary.analyze(prompts.SYSTEM_PROMPT, user_prompt)
    findings = [f.model_copy(update={"path": path}) for f in findings]

    findings = await apply_confidence_gate(
        findings,
        verifier,
        {path: patch},
        settings.confidence_auto_post,
        settings.confidence_verify,
    )
    return validate_finding_lines(findings, {path: commentable_lines(patch)})


async def evaluate_case(
    case: dict[str, Any],
    primary: ReviewBackend,
    verifier: ReviewBackend,
    settings: Any,
) -> CaseResult:
    expected = [Expected(**e) for e in case.get("expected", [])]
    findings = await run_case_pipeline(case, primary, verifier, settings)

    tp, fp, fn, wtp, wfp, wfn = _match(findings, expected)
    return CaseResult(
        name=case["path"],
        tp=tp,
        fp=fp,
        fn=fn,
        wtp=wtp,
        wfp=wfp,
        wfn=wfn,
        predicted=len(findings),
        expected=len(expected),
    )


async def run_eval(
    golden_path: Path,
    backend_name: str,
    min_f1: float,
    verdicts: dict[str, tuple[bool, float]] | None = None,
) -> int:
    settings = get_settings()
    cases = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        print("no golden cases found")
        return 2

    totals = {"tp": 0, "fp": 0, "fn": 0, "wtp": 0.0, "wfp": 0.0, "wfn": 0.0}
    print(f"{'case':<40} {'P':>6} {'R':>6} {'F1':>6}  pred/exp")
    print("-" * 72)
    for case in cases:
        if backend_name == "stub":
            stub_findings = [Finding.model_validate(f) for f in case.get("stub_findings", [])]
            primary: ReviewBackend = StubBackend(canned={"__all__": stub_findings})
            verifier: ReviewBackend = VerdictStubBackend(verdicts)
        else:
            primary = build_backend(settings)
            verifier = build_backend(settings, for_verification=True)
        result = await evaluate_case(case, primary, verifier, settings)
        p, r, f1 = _prf(result.tp, result.fp, result.fn)
        print(
            f"{result.name:<40} {p:>6.2f} {r:>6.2f} {f1:>6.2f}  "
            f"{result.predicted}/{result.expected}"
        )
        for key in ("tp", "fp", "fn", "wtp", "wfp", "wfn"):
            totals[key] += getattr(result, key)

    p, r, f1 = _prf(totals["tp"], totals["fp"], totals["fn"])
    wp, wr, wf1 = _prf(totals["wtp"], totals["wfp"], totals["wfn"])
    print("-" * 72)
    print(f"micro  precision={p:.3f} recall={r:.3f} F1={f1:.3f}")
    print(f"severity-weighted  precision={wp:.3f} recall={wr:.3f} F1={wf1:.3f}")
    print(f"gate: min-f1={min_f1:.2f} -> {'PASS' if f1 >= min_f1 else 'FAIL'}")
    return 0 if f1 >= min_f1 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden eval for PRism.")
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--backend", default="stub", choices=["stub", "vllm", "api", "hf_endpoint"])
    parser.add_argument("--min-f1", type=float, default=0.70)
    args = parser.parse_args()

    setup_logging("WARNING", force=True)
    sys.exit(_run(args))


def _run(args: Any) -> int:
    import asyncio

    return asyncio.run(run_eval(Path(args.golden), args.backend, args.min_f1))


if __name__ == "__main__":
    main()
