"""
Benchmark harness for PRism.

Runs a golden set through the *real* review pipeline
(``run_case_pipeline``: backend -> confidence gate -> line validation) for
two backends and compares them:

- **candidate**: the backend under test — ``stub`` replays each case's
  ``stub_findings``; ``vllm`` / ``api`` / ``hf_endpoint`` run a real model.
- **baseline**: ``HeuristicBaseline``, a dumb-but-honest reference that
  flags obvious patterns (``eval(``, ``TODO``, bare ``except:``, ...) in
  added diff lines. The candidate is expected to beat it.

Outputs a JSON report and a markdown summary table under ``eval/reports/``,
plus a PASS/FAIL verdict against configurable thresholds.

Usage:
    python -m prism.eval.harness --golden eval/golden.jsonl --backend stub
    python -m prism.eval.harness --golden eval/golden.jsonl --backend vllm \\
        --min-f1 0.75 --min-category-f1 security=0.85
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prism.config import Settings, get_settings
from prism.diff.hunks import parse_hunks
from prism.eval.golden import validate_golden
from prism.eval.metrics import (
    AggregateReport,
    ExpectedIssue,
    LatencyStats,
    calibration_bins,
    classification_metrics,
    expected_calibration_error,
    false_positive_rate,
    false_positives_per_case,
    latency_stats,
    match_findings,
    per_category_metrics,
    per_severity_metrics,
    severity_weighted_prf,
)
from prism.eval.runner import (
    VerdictStubBackend,
    run_case_pipeline,
)
from prism.logging import get_logger, setup_logging
from prism.review.backends import ReviewBackend, StubBackend, build_backend
from prism.review.schemas import Finding

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Heuristic baseline backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rule:
    """One regex heuristic: pattern -> finding template."""

    name: str
    pattern: re.Pattern[str]
    category: str
    severity: str
    confidence: float
    title: str
    explanation: str
    suggestion: str


_RULES: tuple[_Rule, ...] = (
    _Rule(
        name="eval",
        pattern=re.compile(r"\beval\s*\("),
        category="security",
        severity="critical",
        confidence=0.85,
        title="Use of eval() on untrusted input",
        explanation=(
            "eval() executes arbitrary code. If the argument derives from user "
            "input this is remote code execution; use ast.literal_eval or a "
            "proper parser instead."
        ),
        suggestion="ast.literal_eval(value)  # or a real parser, never eval()",
    ),
    _Rule(
        name="exec",
        pattern=re.compile(r"\bexec\s*\("),
        category="security",
        severity="critical",
        confidence=0.85,
        title="Use of exec() on untrusted input",
        explanation=(
            "exec() executes arbitrary code with the caller's privileges. "
            "Untrusted input here is remote code execution."
        ),
        suggestion="# refactor: dispatch table of allowed callables instead of exec()",
    ),
    _Rule(
        name="pickle",
        pattern=re.compile(r"\bpickle\.loads?\s*\("),
        category="security",
        severity="critical",
        confidence=0.85,
        title="pickle deserialization of untrusted data",
        explanation=(
            "Unpickling untrusted bytes is remote code execution by design "
            "(arbitrary __reduce__ payloads). Use a safe format like JSON."
        ),
        suggestion="json.loads(blob)  # never pickle.loads() on untrusted data",
    ),
    _Rule(
        name="sqli",
        pattern=re.compile(r"(?i)f[\"'].*\bselect\b"),
        category="security",
        severity="critical",
        confidence=0.85,
        title="Probable SQL injection via f-string query",
        explanation=(
            "An f-string builds a SQL query with interpolated values. Unless every "
            "interpolation is provably safe this is SQL injection; use parameterized "
            "queries."
        ),
        suggestion='cursor.execute("SELECT ... WHERE name = %s", (name,))',
    ),
    _Rule(
        name="shell_true",
        pattern=re.compile(r"shell\s*=\s*True"),
        category="security",
        severity="high",
        confidence=0.85,
        title="subprocess with shell=True",
        explanation=(
            "shell=True passes the command through /bin/sh. Any interpolated "
            "variable becomes command injection; pass an argv list instead."
        ),
        suggestion='subprocess.run(["ffmpeg", "-i", src], check=True)  # no shell',
    ),
    _Rule(
        name="os_system",
        pattern=re.compile(r"\bos\.system\s*\("),
        category="security",
        severity="high",
        confidence=0.85,
        title="os.system() with interpolated command",
        explanation=(
            "os.system() always goes through the shell; interpolated arguments "
            "are command injection. Use subprocess with an argv list."
        ),
        suggestion="subprocess.run([...], check=True)  # argv list, no shell",
    ),
    _Rule(
        name="hardcoded_secret",
        pattern=re.compile(
            r"(?i)(sk-live-[0-9a-z]+|AKIA[0-9A-Z]{16}|xox[bap]-[0-9a-z-]+"
            r"|(api[_-]?key|secret|passwd|password)\s*=\s*[\"'][^\"']{4,}[\"'])"
        ),
        category="security",
        severity="critical",
        confidence=0.80,
        title="Hardcoded secret in source",
        explanation=(
            "A secret committed to source is exposed to every repo reader and "
            "lives forever in git history. Load it from the environment or a "
            "secret manager, then rotate it."
        ),
        suggestion='API_KEY = os.environ["API_KEY"]  # then rotate the leaked key',
    ),
    _Rule(
        name="weak_hash",
        pattern=re.compile(r"\bhashlib\.(md5|sha1)\s*\("),
        category="security",
        severity="medium",
        confidence=0.80,
        title="Weak hash function (MD5/SHA1)",
        explanation=(
            "MD5/SHA1 are broken for collision resistance and far too fast for "
            "password hashing (GPU brute force). Use bcrypt/argon2/scrypt for "
            "passwords, SHA-256+ for integrity."
        ),
        suggestion="hashlib.sha256(...)  # or bcrypt for password storage",
    ),
    _Rule(
        name="bare_except",
        pattern=re.compile(r"^\s*except\s*:\s*(#.*)?$"),
        category="correctness",
        severity="medium",
        confidence=0.75,
        title="Bare except: swallows everything",
        explanation=(
            "A bare except: also catches KeyboardInterrupt and SystemExit and "
            "masks real bugs. Catch the specific exception you expect."
        ),
        suggestion="except ValueError:  # narrow to what you actually expect",
    ),
    _Rule(
        name="todo",
        pattern=re.compile(r"(?i)\b(TODO|FIXME|XXX|HACK)\b"),
        category="style",
        severity="low",
        confidence=0.70,
        title="TODO/FIXME left in the diff",
        explanation=(
            "An unresolved TODO in a merged diff is a forgotten promise. Either "
            "resolve it now or file a tracked issue and reference it."
        ),
        suggestion="# TODO(#123): ...  or resolve before merging",
    ),
)


class HeuristicBaseline(ReviewBackend):
    """
    Dumb-but-honest baseline: regex heuristics over *added* diff lines.

    Extracts the diff from the review prompt, walks added lines with their
    new-file line numbers, and emits one finding per matching rule. It has
    no model behind it — its score is the floor the real backend must beat.
    """

    name = "heuristic-baseline"

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        path = self._extract_path(user_prompt)
        diff = self._extract_diff(user_prompt)
        findings: list[dict[str, Any]] = []
        for new_line, code in self._added_lines(diff):
            for rule in _RULES:
                if rule.pattern.search(code):
                    findings.append(
                        {
                            "path": path,
                            "line": new_line,
                            "severity": rule.severity,
                            "category": rule.category,
                            "title": f"{rule.title} [{rule.name}]",
                            "explanation": rule.explanation,
                            "suggestion": rule.suggestion,
                            "confidence": rule.confidence,
                            "language_hint": "python",
                        }
                    )
        payload = {"findings": findings}
        return json.dumps(payload), {"backend": self.name}

    @staticmethod
    def _extract_path(user_prompt: str) -> str:
        match = re.search(r"changes to `([^`]+)`", user_prompt)
        return match.group(1) if match else "unknown.py"

    @staticmethod
    def _extract_diff(user_prompt: str) -> str:
        """Pull the ```diff block out of the review prompt."""
        start = user_prompt.find("```diff")
        if start == -1:
            return user_prompt
        start += len("```diff")
        end = user_prompt.find("```", start)
        return user_prompt[start:end] if end != -1 else user_prompt[start:]

    @staticmethod
    def _added_lines(diff: str) -> list[tuple[int, str]]:
        """(new-file line number, code) for every added diff line."""
        added: list[tuple[int, str]] = []
        for hunk in parse_hunks(diff):
            new_line = hunk.new_start
            for raw in hunk.lines:
                if raw.startswith("+") and not raw.startswith("+++"):
                    added.append((new_line, raw[1:]))
                    new_line += 1
                elif raw.startswith("-") and not raw.startswith("---"):
                    continue
                else:
                    new_line += 1
        return added


class AcceptAllVerifier(ReviewBackend):
    """
    Verifier used *only* for the heuristic baseline.

    The baseline's confidences are hand-set constants, not model
    uncertainties, so a real second-pass verdict would be meaningless. This
    verifier accepts every finding and echoes its first-pass confidence
    (parsed from the verify prompt) so calibration stays measurable.
    """

    name = "accept-all-verifier"

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        match = re.search(r"First-pass confidence: ([0-9.]+)", user_prompt)
        confidence = float(match.group(1)) if match else 0.8
        verdict = {
            "verdict": "real",
            "revised_confidence": confidence,
            "reason": "baseline heuristics bypass second-pass verification",
        }
        return json.dumps(verdict), {"backend": self.name}


# ---------------------------------------------------------------------------
# Harness config / report
# ---------------------------------------------------------------------------


@dataclass
class HarnessThresholds:
    """PASS/FAIL gates for the benchmark verdict."""

    min_micro_f1: float = 0.70
    min_weighted_f1: float = 0.70
    min_category_f1: dict[str, float] = field(
        default_factory=lambda: {"security": 0.80, "correctness": 0.70}
    )
    max_false_positive_rate: float = 0.35
    max_calibration_error: float = 0.25
    require_beats_baseline: bool = True


@dataclass
class HarnessConfig:
    """Harness run configuration."""

    golden_path: Path
    report_dir: Path = Path("eval/reports")
    backend: str = "stub"  # stub | vllm | api | hf_endpoint
    thresholds: HarnessThresholds = field(default_factory=HarnessThresholds)
    line_tolerance: int = 2
    include_baseline: bool = True


@dataclass
class _CaseScore:
    """Per-case pipeline output plus match outcome."""

    case_id: str
    path: str
    findings: list[Finding]
    expected: list[ExpectedIssue]
    tp: int
    fp: int
    fn: int
    latency_s: float


@dataclass
class VerdictCheck:
    """One named gate: threshold, actual value, pass/fail."""

    name: str
    threshold: str
    actual: float | None
    passed: bool
    skipped: bool = False


@dataclass
class BenchmarkReport:
    """Full benchmark outcome: metrics for candidate + baseline, verdict."""

    generated_at: str
    golden_path: str
    backend: str
    n_cases: int
    candidate: AggregateReport
    baseline: AggregateReport | None
    checks: list[VerdictCheck]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form of the report."""
        return {
            "generated_at": self.generated_at,
            "golden_path": self.golden_path,
            "backend": self.backend,
            "n_cases": self.n_cases,
            "verdict": {
                "passed": self.passed,
                "checks": [
                    {
                        "name": c.name,
                        "threshold": c.threshold,
                        "actual": c.actual,
                        "passed": c.passed,
                        "skipped": c.skipped,
                    }
                    for c in self.checks
                ],
            },
            "candidate": _aggregate_to_dict(self.candidate),
            "baseline": _aggregate_to_dict(self.baseline) if self.baseline else None,
        }


def _metrics_to_dict(m: Any) -> dict[str, Any]:
    return {
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
    }


def _aggregate_to_dict(agg: AggregateReport) -> dict[str, Any]:
    lat: LatencyStats = agg.latency
    return {
        "n_cases": agg.n_cases,
        "micro": _metrics_to_dict(agg.micro),
        "weighted_f1": agg.weighted_f1,
        "per_category": {
            cat: {**_metrics_to_dict(m), "support": m.tp + m.fn}
            for cat, m in agg.per_category.items()
        },
        "per_severity": {
            sev: {**_metrics_to_dict(m), "support": m.tp + m.fn}
            for sev, m in agg.per_severity.items()
        },
        "false_positive_rate": agg.false_positive_rate,
        "false_positives_per_case": agg.false_positives_per_case,
        "calibration_error": agg.calibration_error,
        "calibration_table": [
            {
                "bin": [b.bin_start, b.bin_end],
                "count": b.count,
                "mean_confidence": b.mean_confidence,
                "empirical_precision": b.empirical_precision,
            }
            for b in agg.calibration_table
        ],
        "latency": {
            "count": lat.count,
            "mean_s": lat.mean,
            "min_s": lat.min,
            "max_s": lat.max,
            "p50_s": lat.p50,
            "p95_s": lat.p95,
        },
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class GoldenValidationError(ValueError):
    """Raised when the golden file fails validate_golden()."""


class BenchmarkHarness:
    """
    Run a golden set against the candidate backend and the heuristic
    baseline, then score with the metric library.

    The candidate goes through the real review pipeline per case
    (``run_case_pipeline``); the baseline goes through the same pipeline
    with ``AcceptAllVerifier`` (documented above).
    """

    def __init__(self, config: HarnessConfig, settings: Settings | None = None) -> None:
        self.config = config
        self.settings = settings or get_settings()

    async def run(self) -> BenchmarkReport:
        """Execute the benchmark; returns the scored report (not yet saved)."""
        errors = validate_golden(self.config.golden_path)
        if errors:
            raise GoldenValidationError(
                f"{self.config.golden_path}: {len(errors)} validation error(s):\n"
                + "\n".join(errors[:10])
            )
        cases = [
            json.loads(line)
            for line in self.config.golden_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not cases:
            raise GoldenValidationError(f"no golden cases in {self.config.golden_path}")

        candidate = await self._run_backend(cases, self.config.backend, baseline=False)
        baseline = (
            await self._run_backend(cases, "heuristic", baseline=True)
            if self.config.include_baseline
            else None
        )
        checks = self._verdict(candidate, baseline)
        return BenchmarkReport(
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            golden_path=str(self.config.golden_path),
            backend=self.config.backend,
            n_cases=len(cases),
            candidate=candidate,
            baseline=baseline,
            checks=checks,
            passed=all(c.passed for c in checks if not c.skipped),
        )

    async def _run_backend(
        self, cases: list[dict[str, Any]], backend_name: str, baseline: bool
    ) -> AggregateReport:
        """Score one backend across all cases."""
        scores: list[_CaseScore] = []
        shared_primary: ReviewBackend | None = None
        shared_verifier: ReviewBackend | None = None
        if backend_name != "stub" and not baseline:
            # Real backends are built once and reused across cases.
            shared_primary = build_backend(self.settings)
            shared_verifier = build_backend(self.settings, for_verification=True)
        for case in cases:
            primary, verifier = self._backends_for_case(
                case, backend_name, baseline, shared_primary, shared_verifier
            )
            scores.append(await self._score_case(case, primary, verifier))
        return self._summarize(scores)

    def _backends_for_case(
        self,
        case: dict[str, Any],
        backend_name: str,
        baseline: bool,
        shared_primary: ReviewBackend | None,
        shared_verifier: ReviewBackend | None,
    ) -> tuple[ReviewBackend, ReviewBackend]:
        if baseline:
            return HeuristicBaseline(), AcceptAllVerifier()
        if backend_name == "stub":
            stub_findings = [Finding.model_validate(f) for f in case.get("stub_findings", [])]
            return StubBackend(canned={"__all__": stub_findings}), VerdictStubBackend()
        assert shared_primary is not None and shared_verifier is not None
        return shared_primary, shared_verifier

    async def _score_case(
        self,
        case: dict[str, Any],
        primary: ReviewBackend,
        verifier: ReviewBackend,
    ) -> _CaseScore:
        start = time.perf_counter()
        findings = await run_case_pipeline(case, primary, verifier, self.settings)
        latency = time.perf_counter() - start
        expected = [ExpectedIssue(**e) for e in case.get("expected", [])]
        match = match_findings(findings, expected, self.config.line_tolerance)
        tp = sum(match.predicted_is_tp)
        return _CaseScore(
            case_id=str(case.get("id", case.get("path", "?"))),
            path=str(case.get("path", "?")),
            findings=findings,
            expected=expected,
            tp=tp,
            fp=len(findings) - tp,
            fn=len(expected) - sum(match.expected_matched),
            latency_s=latency,
        )

    def _summarize(self, scores: list[_CaseScore]) -> AggregateReport:
        findings: list[Finding] = [f for s in scores for f in s.findings]
        expected: list[ExpectedIssue] = [e for s in scores for e in s.expected]
        # Re-match globally is unnecessary: per-case matches are disjoint, so
        # we can rebuild a global MatchResult by concatenation.
        from prism.eval.metrics import MatchResult

        pred_tp: list[bool] = []
        exp_hit: list[bool] = []
        pairs: list[tuple[int, int]] = []
        pred_off = 0
        exp_off = 0
        for s in scores:
            m = match_findings(s.findings, s.expected, self.config.line_tolerance)
            pred_tp.extend(m.predicted_is_tp)
            exp_hit.extend(m.expected_matched)
            pairs.extend((pi + pred_off, ei + exp_off) for pi, ei in m.pairs)
            pred_off += len(s.findings)
            exp_off += len(s.expected)
        match = MatchResult(
            predicted_is_tp=tuple(pred_tp),
            expected_matched=tuple(exp_hit),
            pairs=tuple(pairs),
        )
        tp = sum(pred_tp)
        fp = len(findings) - tp
        fn = len(expected) - sum(exp_hit)
        confidences = [f.confidence for f in findings]
        correct = list(pred_tp)
        _, _, weighted_f1 = severity_weighted_prf(findings, expected, match)
        return AggregateReport(
            n_cases=len(scores),
            micro=classification_metrics(tp, fp, fn),
            weighted_f1=weighted_f1,
            per_category=per_category_metrics(findings, expected, match),
            per_severity=per_severity_metrics(findings, expected, match),
            false_positive_rate=false_positive_rate(tp, fp),
            false_positives_per_case=false_positives_per_case(fp, len(scores)),
            calibration_error=expected_calibration_error(confidences, correct),
            calibration_table=calibration_bins(confidences, correct),
            latency=latency_stats([s.latency_s for s in scores]),
        )

    def _verdict(
        self, candidate: AggregateReport, baseline: AggregateReport | None
    ) -> list[VerdictCheck]:
        t = self.config.thresholds
        checks = [
            VerdictCheck(
                name="micro F1",
                threshold=f">= {t.min_micro_f1:.2f}",
                actual=candidate.micro.f1,
                passed=candidate.micro.f1 >= t.min_micro_f1,
            ),
            VerdictCheck(
                name="severity-weighted F1",
                threshold=f">= {t.min_weighted_f1:.2f}",
                actual=candidate.weighted_f1,
                passed=candidate.weighted_f1 >= t.min_weighted_f1,
            ),
        ]
        for category, minimum in sorted(t.min_category_f1.items()):
            metrics = candidate.per_category.get(category)
            support = (metrics.tp + metrics.fn) if metrics else 0
            if metrics is None or support == 0:
                checks.append(
                    VerdictCheck(
                        name=f"{category} F1",
                        threshold=f">= {minimum:.2f}",
                        actual=None,
                        passed=True,
                        skipped=True,
                    )
                )
            else:
                checks.append(
                    VerdictCheck(
                        name=f"{category} F1",
                        threshold=f">= {minimum:.2f} (support={support})",
                        actual=metrics.f1,
                        passed=metrics.f1 >= minimum,
                    )
                )
        checks.append(
            VerdictCheck(
                name="false positive rate",
                threshold=f"<= {t.max_false_positive_rate:.2f}",
                actual=candidate.false_positive_rate,
                passed=candidate.false_positive_rate <= t.max_false_positive_rate,
            )
        )
        checks.append(
            VerdictCheck(
                name="calibration ECE",
                threshold=f"<= {t.max_calibration_error:.2f}",
                actual=candidate.calibration_error,
                passed=candidate.calibration_error <= t.max_calibration_error,
            )
        )
        if baseline is not None and t.require_beats_baseline:
            checks.append(
                VerdictCheck(
                    name="beats heuristic baseline (micro F1)",
                    threshold=f">= baseline {baseline.micro.f1:.3f}",
                    actual=candidate.micro.f1,
                    passed=candidate.micro.f1 >= baseline.micro.f1,
                )
            )
        return checks

    def write_reports(self, report: BenchmarkReport) -> tuple[Path, Path]:
        """
        Write the JSON report and markdown summary under the report dir.

        Returns (json_path, markdown_path).
        """
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        stamp = report.generated_at.replace(":", "").replace("+", "")
        json_path = self.config.report_dir / f"benchmark-{stamp}.json"
        md_path = self.config.report_dir / f"benchmark-{stamp}.md"
        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        log.info("benchmark_reports_written", json=str(json_path), md=str(md_path))
        return json_path, md_path


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def render_markdown(report: BenchmarkReport) -> str:
    """Render the benchmark report as a markdown summary table."""
    c = report.candidate
    verdict = "✅ PASS" if report.passed else "❌ FAIL"
    lines = [
        f"# PRism benchmark — {report.generated_at}",
        "",
        f"golden: `{report.golden_path}` ({report.n_cases} cases) · backend: `{report.backend}`",
        "",
        f"## Verdict: {verdict}",
        "",
        "| check | actual | threshold | status |",
        "|---|---|---|---|",
    ]
    for check in report.checks:
        status = "⏭️ skip" if check.skipped else ("✅" if check.passed else "❌")
        lines.append(f"| {check.name} | {_fmt(check.actual)} | {check.threshold} | {status} |")
    lines += [
        "",
        f"## Candidate: `{report.backend}`",
        "",
        "| category | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for category, m in c.per_category.items():
        lines.append(
            f"| {category} | {_fmt(m.precision)} | {_fmt(m.recall)} | "
            f"{_fmt(m.f1)} | {m.tp + m.fn} |"
        )
    lines += [
        "",
        f"- micro: P={_fmt(c.micro.precision)} R={_fmt(c.micro.recall)} "
        f"F1={_fmt(c.micro.f1)} (tp={c.micro.tp} fp={c.micro.fp} fn={c.micro.fn})",
        f"- severity-weighted F1: {_fmt(c.weighted_f1)}",
        f"- false positive rate: {_fmt(c.false_positive_rate)} "
        f"({_fmt(c.false_positives_per_case, 2)} FP/case)",
        f"- calibration ECE: {_fmt(c.calibration_error)}",
        f"- latency: p50={_fmt(c.latency.p50)}s p95={_fmt(c.latency.p95)}s "
        f"max={_fmt(c.latency.max)}s",
    ]
    if report.baseline is not None:
        b = report.baseline
        delta = c.micro.f1 - b.micro.f1
        lines += [
            "",
            "## Baseline: `heuristic-baseline`",
            "",
            f"- micro F1: {_fmt(b.micro.f1)} (Δ {_fmt(delta, 3)} vs candidate)",
            f"- severity-weighted F1: {_fmt(b.weighted_f1)}",
            f"- false positive rate: {_fmt(b.false_positive_rate)}",
        ]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_category_thresholds(raw: str) -> dict[str, float]:
    """Parse 'security=0.8,correctness=0.7' into a dict."""
    thresholds: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        thresholds[name.strip()] = float(value)
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="PRism benchmark harness.")
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--backend", default="stub", choices=["stub", "vllm", "api", "hf_endpoint"])
    parser.add_argument("--min-f1", type=float, default=0.70)
    parser.add_argument("--min-weighted-f1", type=float, default=0.70)
    parser.add_argument("--min-category-f1", default="security=0.80,correctness=0.70")
    parser.add_argument("--max-fpr", type=float, default=0.35)
    parser.add_argument("--max-ece", type=float, default=0.25)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-beats-baseline", action="store_true")
    parser.add_argument("--report-dir", default="eval/reports")
    args = parser.parse_args()

    setup_logging("WARNING", force=True)
    sys.exit(_run(args))


def _run(args: Any) -> int:
    import asyncio

    return asyncio.run(_arun(args))


async def _arun(args: Any) -> int:
    thresholds = HarnessThresholds(
        min_micro_f1=args.min_f1,
        min_weighted_f1=args.min_weighted_f1,
        min_category_f1=_parse_category_thresholds(args.min_category_f1),
        max_false_positive_rate=args.max_fpr,
        max_calibration_error=args.max_ece,
        require_beats_baseline=not args.no_beats_baseline,
    )
    config = HarnessConfig(
        golden_path=Path(args.golden),
        report_dir=Path(args.report_dir),
        backend=args.backend,
        thresholds=thresholds,
        include_baseline=not args.no_baseline,
    )
    harness = BenchmarkHarness(config)
    try:
        report = await harness.run()
    except GoldenValidationError as exc:
        print(f"golden validation failed:\n{exc}")
        return 2
    json_path, md_path = harness.write_reports(report)
    print(render_markdown(report))
    print(f"reports: {json_path} {md_path}")
    print(f"gate: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    main()
