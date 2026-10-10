"""Tests for the eval metric library, benchmark harness, and golden validation."""

from pathlib import Path

import pytest

from prism.eval.golden import CATEGORIES, validate_golden
from prism.eval.harness import (
    AcceptAllVerifier,
    BenchmarkHarness,
    HarnessConfig,
    HarnessThresholds,
    HeuristicBaseline,
    render_markdown,
)
from prism.eval.metrics import (
    ExpectedIssue,
    calibration_bins,
    classification_metrics,
    expected_calibration_error,
    false_positive_rate,
    false_positives_per_case,
    latency_stats,
    match_findings,
    per_category_metrics,
    per_severity_metrics,
    prf,
    severity_weighted_prf,
)
from prism.eval.runner import run_eval
from prism.review import prompts
from prism.review.engine import make_finding
from prism.review.schemas import Severity

REPO_ROOT = Path(__file__).resolve().parent.parent


def _exp(line: int, category: str = "security", severity: str = "high") -> ExpectedIssue:
    return ExpectedIssue(line=line, category=category, severity=severity)


# ---------------------------------------------------------------------------
# prf
# ---------------------------------------------------------------------------


def test_prf_basic():
    # tp=2, fp=1, fn=1 -> P=2/3, R=2/3, F1=2/3
    p, r, f1 = prf(2, 1, 1)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_prf_empty_conventions():
    # Nothing predicted and nothing expected: vacuously perfect.
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0)


def test_prf_no_predictions():
    p, r, f1 = prf(0, 0, 3)
    assert p == 1.0
    assert r == 0.0
    assert f1 == 0.0


def test_classification_metrics_bundles_counts():
    m = classification_metrics(2, 1, 1)
    assert (m.tp, m.fp, m.fn) == (2, 1, 1)
    assert m.f1 == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def test_match_within_tolerance():
    pred = [make_finding(line=12, category="security")]
    match = match_findings(pred, [_exp(10, "security")])
    assert match.predicted_is_tp == (True,)
    assert match.expected_matched == (True,)


def test_match_outside_tolerance():
    pred = [make_finding(line=13, category="security")]
    match = match_findings(pred, [_exp(10, "security")])
    assert match.predicted_is_tp == (False,)
    assert match.expected_matched == (False,)


def test_match_category_mismatch():
    pred = [make_finding(line=10, category="performance")]
    match = match_findings(pred, [_exp(10, "security")])
    assert match.predicted_is_tp == (False,)
    assert match.expected_matched == (False,)


def test_match_duplicate_predictions_single_tp():
    pred = [
        make_finding(line=10, category="security"),
        make_finding(line=11, category="security"),
    ]
    match = match_findings(pred, [_exp(10, "security")])
    # First prediction claims the expected issue; the duplicate is an FP.
    assert match.predicted_is_tp == (True, False)
    assert match.expected_matched == (True,)
    assert match.pairs == ((0, 0),)


def test_match_custom_tolerance():
    pred = [make_finding(line=15, category="security")]
    assert match_findings(pred, [_exp(10, "security")], line_tolerance=5).predicted_is_tp == (True,)
    assert match_findings(pred, [_exp(10, "security")], line_tolerance=4).predicted_is_tp == (
        False,
    )


# ---------------------------------------------------------------------------
# per-category / per-severity / weighted
# ---------------------------------------------------------------------------


def test_per_category_metrics():
    pred = [
        make_finding(line=10, category="security"),
        make_finding(line=20, category="security"),
        make_finding(line=30, category="performance"),
    ]
    exp = [_exp(10, "security"), _exp(12, "security"), _exp(31, "performance")]
    by_cat = per_category_metrics(pred, exp, match_findings(pred, exp))
    sec = by_cat["security"]
    assert (sec.tp, sec.fp, sec.fn) == (1, 1, 1)
    assert sec.f1 == pytest.approx(0.5)
    perf = by_cat["performance"]
    assert (perf.tp, perf.fp, perf.fn) == (1, 0, 0)
    assert perf.f1 == pytest.approx(1.0)


def test_per_severity_metrics():
    pred = [
        make_finding(line=10, category="security", severity=Severity.critical),
        make_finding(line=20, category="security", severity=Severity.low),
    ]
    exp = [_exp(10, "security", "critical"), _exp(50, "security", "high")]
    by_sev = per_severity_metrics(pred, exp, match_findings(pred, exp))
    assert (by_sev["critical"].tp, by_sev["critical"].fp, by_sev["critical"].fn) == (1, 0, 0)
    assert (by_sev["low"].tp, by_sev["low"].fp, by_sev["low"].fn) == (0, 1, 0)
    assert (by_sev["high"].tp, by_sev["high"].fp, by_sev["high"].fn) == (0, 0, 1)


def test_severity_weighted_prf():
    # TP critical (w=4), FP low (w=1), FN high (w=3):
    # P=4/5=0.8, R=4/7, F1=2/3.
    pred = [
        make_finding(line=10, category="security", severity=Severity.critical),
        make_finding(line=99, category="security", severity=Severity.low),
    ]
    exp = [_exp(10, "security", "critical"), _exp(50, "security", "high")]
    p, r, f1 = severity_weighted_prf(pred, exp, match_findings(pred, exp))
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(4 / 7)
    assert f1 == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# false positives
# ---------------------------------------------------------------------------


def test_false_positive_rate():
    assert false_positive_rate(3, 1) == pytest.approx(0.25)
    assert false_positive_rate(0, 0) == 0.0


def test_false_positives_per_case():
    assert false_positives_per_case(7, 2) == pytest.approx(3.5)
    assert false_positives_per_case(0, 0) == 0.0


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_calibration_perfect():
    # Mean confidence 0.75 in the single occupied bucket, precision 0.75.
    ece = expected_calibration_error([0.9, 0.9, 0.6, 0.6], [True, True, True, False], n_bins=2)
    assert ece == pytest.approx(0.0)


def test_calibration_error_hand_computed():
    # Bin [0, .5): conf 0.0, precision 0.5 -> gap .5, weight .5 -> .25
    # Bin [.5, 1]: conf 1.0, precision 0.5 -> gap .5, weight .5 -> .25
    ece = expected_calibration_error([1.0, 1.0, 0.0, 0.0], [True, False, True, False], n_bins=2)
    assert ece == pytest.approx(0.5)


def test_calibration_bins_table():
    bins = calibration_bins([0.9, 0.1], [True, False], n_bins=2)
    assert [b.count for b in bins] == [1, 1]
    assert bins[0].empirical_precision == pytest.approx(0.0)
    assert bins[1].empirical_precision == pytest.approx(1.0)
    assert bins[1].mean_confidence == pytest.approx(0.9)


def test_calibration_empty():
    assert expected_calibration_error([], []) == 0.0


def test_calibration_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        calibration_bins([0.5], [True, False])


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------


def test_latency_stats():
    s = latency_stats([1.0, 2.0, 3.0, 4.0])
    assert s.count == 4
    assert s.mean == pytest.approx(2.5)
    assert s.min == pytest.approx(1.0)
    assert s.max == pytest.approx(4.0)
    assert s.p50 == pytest.approx(2.5)
    # rank = 3 * 0.95 = 2.85 -> 3.0 + 0.85 * (4.0 - 3.0)
    assert s.p95 == pytest.approx(3.85)


def test_latency_stats_empty():
    s = latency_stats([])
    assert s.count == 0
    assert s.mean == s.p50 == s.p95 == 0.0


# ---------------------------------------------------------------------------
# HeuristicBaseline
# ---------------------------------------------------------------------------


def _prompt_for(patch: str, path: str = "a.py") -> tuple[str, str]:
    from prism.review.prompts import SYSTEM_PROMPT

    user = prompts.build_user_prompt(
        path=path, language=prompts.guess_language(path), hunk_context=patch
    )
    return SYSTEM_PROMPT, user


async def test_baseline_flags_eval_and_todo():
    patch = (
        "@@ -1,2 +1,4 @@\n"
        " def f(x):\n"
        "+    result = eval(x)\n"
        "+    # TODO: handle errors\n"
        "     return result\n"
    )
    system, user = _prompt_for(patch)
    findings, _ = await HeuristicBaseline().analyze(system, user)
    by_line = {f.line: f for f in findings}
    assert set(by_line) == {2, 3}
    assert by_line[2].category == "security"
    assert by_line[2].severity == Severity.critical
    assert by_line[3].category == "style"


async def test_baseline_ignores_context_and_removed_lines():
    patch = "@@ -1,4 +1,3 @@\n def f(x):\n-    result = eval(x)\n     y = eval(x)\n+    return y\n"
    system, user = _prompt_for(patch)
    findings, _ = await HeuristicBaseline().analyze(system, user)
    assert findings == []  # eval( only in context/removed lines


async def test_baseline_flags_shell_and_bare_except():
    patch = (
        "@@ -1,2 +1,4 @@\n"
        " def f(cmd):\n"
        "+    subprocess.run(cmd, shell=True)\n"
        "+    try:\n"
        "+        int(cmd)\n"
        "+    except:\n"
        "+        pass\n"
    )
    system, user = _prompt_for(patch)
    findings, _ = await HeuristicBaseline().analyze(system, user)
    cats = {(f.line, f.category) for f in findings}
    assert (2, "security") in cats
    assert (5, "correctness") in cats


async def test_accept_all_verifier_echoes_confidence():
    text, _ = await AcceptAllVerifier().complete(
        "system", "First-pass confidence: 0.72\nTitle: whatever"
    )
    import json

    verdict = json.loads(text)
    assert verdict["verdict"] == "real"
    assert verdict["revised_confidence"] == pytest.approx(0.72)


# ---------------------------------------------------------------------------
# golden validation
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _good_case(case_id: str = "c1") -> str:
    import json

    return json.dumps(
        {
            "id": case_id,
            "path": "a.py",
            "patch": "@@ -1,2 +1,3 @@\n def f():\n+    x = eval(y)\n     return x\n",
            "expected": [{"line": 2, "category": "security", "severity": "critical"}],
            "stub_findings": [
                {
                    "path": "a.py",
                    "line": 2,
                    "severity": "critical",
                    "category": "security",
                    "title": "eval usage",
                    "explanation": "eval on untrusted input is remote code execution",
                    "confidence": 0.9,
                }
            ],
        }
    )


def test_validate_golden_good(tmp_path):
    assert validate_golden(_write(tmp_path, "g.jsonl", [_good_case()])) == []


def test_validate_golden_bad_category(tmp_path):
    import json

    case = json.loads(_good_case())
    case["expected"][0]["category"] = "vibes"
    errors = validate_golden(_write(tmp_path, "g.jsonl", [json.dumps(case)]))
    assert any("category" in e for e in errors)


def test_validate_golden_line_out_of_patch(tmp_path):
    import json

    case = json.loads(_good_case())
    case["expected"][0]["line"] = 999
    errors = validate_golden(_write(tmp_path, "g.jsonl", [json.dumps(case)]))
    assert any("not commentable" in e for e in errors)


def test_validate_golden_missing_key(tmp_path):
    import json

    case = json.loads(_good_case())
    del case["expected"]
    errors = validate_golden(_write(tmp_path, "g.jsonl", [json.dumps(case)]))
    assert any("expected" in e for e in errors)


def test_validate_golden_duplicate_id(tmp_path):
    errors = validate_golden(_write(tmp_path, "g.jsonl", [_good_case("c1"), _good_case("c1")]))
    assert any("duplicate id" in e for e in errors)


def test_validate_golden_bad_json(tmp_path):
    errors = validate_golden(_write(tmp_path, "g.jsonl", ["{not json"]))
    assert any("invalid JSON" in e for e in errors)


def test_validate_golden_bad_severity_and_stub(tmp_path):
    import json

    case = json.loads(_good_case())
    case["expected"][0]["severity"] = "extreme"
    case["stub_findings"][0]["line"] = "two"  # invalid Finding
    errors = validate_golden(_write(tmp_path, "g.jsonl", [json.dumps(case)]))
    assert any("severity" in e for e in errors)
    assert any("stub_findings" in e for e in errors)


def test_validate_golden_missing_file(tmp_path):
    errors = validate_golden(tmp_path / "nope.jsonl")
    assert len(errors) == 1 and "not found" in errors[0]


def test_validate_golden_categories_are_known_vocabulary():
    assert set(CATEGORIES) >= {"security", "correctness", "performance", "style"}


# ---------------------------------------------------------------------------
# harness end-to-end
# ---------------------------------------------------------------------------


def _mini_golden() -> list[str]:
    import json

    def finding(path: str, line: int, category: str, severity: str, conf: float) -> dict:
        return {
            "path": path,
            "line": line,
            "severity": severity,
            "category": category,
            "title": f"{category} issue",
            "explanation": "a genuine issue described at sufficient length",
            "confidence": conf,
        }

    cases = [
        {
            "id": "m-eval",
            "path": "a.py",
            "patch": "@@ -1,2 +1,3 @@\n def f(x):\n+    return eval(x)\n     return 1\n",
            "expected": [{"line": 2, "category": "security", "severity": "critical"}],
            "stub_findings": [finding("a.py", 2, "security", "critical", 0.95)],
        },
        {
            # Off-by-one: the heuristic baseline has no rule for this, so it
            # misses it while the stub replay catches it.
            "id": "m-offbyone",
            "path": "b.py",
            "patch": (
                "@@ -5,3 +5,3 @@\n def g(items):\n-    return items[1:]\n"
                "+    return items[2:]\n     return None\n"
            ),
            "expected": [{"line": 6, "category": "correctness", "severity": "medium"}],
            "stub_findings": [finding("b.py", 6, "correctness", "medium", 0.90)],
        },
        {
            "id": "m-clean",
            "path": "c.py",
            "patch": "@@ -1,1 +1,2 @@\n x = 1\n+y = 2\n",
            "expected": [],
            "stub_findings": [],
        },
    ]
    return [json.dumps(c) for c in cases]


def _harness(tmp_path: Path, thresholds: HarnessThresholds | None = None) -> BenchmarkHarness:
    golden = _write(tmp_path, "mini.jsonl", _mini_golden())
    config = HarnessConfig(
        golden_path=golden,
        report_dir=tmp_path / "reports",
        backend="stub",
        thresholds=thresholds or HarnessThresholds(min_category_f1={}),
        include_baseline=True,
    )
    return BenchmarkHarness(config)


async def test_harness_end_to_end_pass(tmp_path):
    harness = _harness(tmp_path)
    report = await harness.run()
    assert report.passed
    assert report.n_cases == 3
    assert report.candidate.micro.f1 == pytest.approx(1.0)
    assert report.baseline is not None
    assert report.baseline.micro.f1 < report.candidate.micro.f1  # baseline is beatable

    json_path, md_path = harness.write_reports(report)
    assert json_path.is_file() and md_path.is_file()
    import json as _json

    data = _json.loads(json_path.read_text(encoding="utf-8"))
    assert data["verdict"]["passed"] is True
    assert data["candidate"]["per_category"]["security"]["f1"] == pytest.approx(1.0)
    md = md_path.read_text(encoding="utf-8")
    assert "## Verdict: ✅ PASS" in md
    assert "| security |" in md
    assert render_markdown(report) == md


async def test_harness_verdict_fail(tmp_path):
    thresholds = HarnessThresholds(
        min_micro_f1=0.99,  # stub replays nothing -> F1 0.0
        min_category_f1={},
        require_beats_baseline=False,
    )
    import json

    case = json.loads(_mini_golden()[0])
    case["stub_findings"] = []  # candidate finds nothing
    golden = _write(tmp_path, "mini.jsonl", [json.dumps(case)])
    config = HarnessConfig(
        golden_path=golden,
        report_dir=tmp_path / "reports",
        thresholds=thresholds,
        include_baseline=False,
    )
    report = await BenchmarkHarness(config).run()
    assert not report.passed
    assert report.candidate.micro.f1 == pytest.approx(0.0)
    assert any(not c.passed and not c.skipped for c in report.checks)


async def test_harness_rejects_invalid_golden(tmp_path):
    from prism.eval.harness import GoldenValidationError

    bad = _write(tmp_path, "bad.jsonl", ["{nope"])
    config = HarnessConfig(golden_path=bad, report_dir=tmp_path / "reports")
    with pytest.raises(GoldenValidationError):
        await BenchmarkHarness(config).run()


# ---------------------------------------------------------------------------
# runner regression: existing CLI behavior unchanged
# ---------------------------------------------------------------------------


async def test_runner_still_passes_example_golden():
    # The pre-existing runner entry point keeps working after the refactor.
    rc = await run_eval(REPO_ROOT / "eval" / "golden.example.jsonl", "stub", 0.0)
    assert rc == 0
