"""Tests for the GitHub review payload shape and markdown rendering."""

import pytest

from prism.github import reviews
from prism.review.engine import make_finding
from prism.review.schemas import Severity


def test_build_review_payload_shape():
    findings = [
        make_finding(path="a.py", line=12, severity=Severity.critical, title="SQLi"),
        make_finding(path="b.py", line=3, severity=Severity.low, title="Nit"),
    ]
    payload = reviews.build_review_payload("abc123sha", "summary text", findings)

    assert payload["commit_id"] == "abc123sha"
    assert payload["event"] == "COMMENT"
    assert payload["body"] == "summary text"
    assert len(payload["comments"]) == 2

    c0 = payload["comments"][0]
    assert c0["path"] == "a.py"
    assert c0["line"] == 12
    assert c0["side"] == "RIGHT"  # new-file side, exact casing
    assert "<!-- PRism -->" in c0["body"]
    assert "<!-- PRism fp:" in c0["body"]  # dedup fingerprint embedded


def test_render_finding_body_contents():
    f = make_finding(
        severity=Severity.high,
        category="security",
        title="XSS",
        explanation="Unescaped user input is rendered as HTML.",
        suggestion="<div>{escape(name)}</div>",
        confidence=0.81,
    )
    body = reviews.render_finding_body(f)
    assert "🟠" not in body
    assert "**HIGH**" in body
    assert "`security`" in body
    assert "```" in body and "escape(name)" in body
    assert "_confidence: 0.81_" in body


def test_render_summary_counts():
    findings = [
        make_finding(severity=Severity.critical),
        make_finding(severity=Severity.critical),
        make_finding(severity=Severity.low),
    ]
    summary = reviews.render_summary("o", "r", 1, findings, ["lock.json (ignored)"], "vllm")
    assert "## PRism review" in summary
    assert "critical: 2" in summary
    assert "low: 1" in summary
    assert "lock.json" in summary


def test_render_summary_no_findings():
    summary = reviews.render_summary("o", "r", 1, [], [], "vllm")
    assert "no issues found" in summary
    assert "looks clean" not in summary


def test_render_summary_backend_errors_no_findings():
    summary = reviews.render_summary("o", "r", 1, [], [], "vllm", ["a.py: connection refused"])
    assert "Review incomplete" in summary
    assert "not a clean bill of health" in summary
    assert "a.py: connection refused" in summary


def test_render_summary_backend_errors_with_findings():
    findings = [make_finding(severity=Severity.high)]
    summary = reviews.render_summary("o", "r", 1, findings, [], "vllm", ["b.py: timeout"])
    assert "high: 1" in summary
    assert "were not reviewed" in summary


def test_render_finding_body_short_code_suggestion_uses_suggestion_fence():
    f = make_finding(
        suggestion="name = escape(name)",
        explanation="Unescaped user input is rendered as HTML.",
    )
    body = reviews.render_finding_body(f)
    assert "```suggestion" in body
    assert "name = escape(name)" in body


def test_render_finding_body_long_suggestion_keeps_language_fence():
    from prism.review.schemas import Finding

    suggestion = "\n".join(f"x{i} = compute({i});" for i in range(13))
    f = Finding(
        path="a.py",
        line=1,
        severity="medium",
        category="correctness",
        title="Long fix needed here",
        explanation="The replacement spans many lines of code here.",
        suggestion=suggestion,
        language_hint="python",
        confidence=0.9,
    )
    body = reviews.render_finding_body(f)
    assert "```suggestion" not in body
    assert "```python" in body
    assert suggestion in body


def test_render_finding_body_prose_suggestion_keeps_language_fence():
    from prism.review.schemas import Finding

    f = Finding(
        path="a.py",
        line=1,
        severity="low",
        category="style",
        title="Style nit picked up",
        explanation="This block could be more readable overall.",
        suggestion="Consider extracting this logic into a small helper function "
        "to improve readability and keep the enclosing block short.",
        confidence=0.8,
    )
    body = reviews.render_finding_body(f)
    assert "```suggestion" not in body
    assert "Suggested fix:" in body


def test_render_finding_body_empty_suggestion_no_block():
    f = make_finding(suggestion="", explanation="Unescaped user input is rendered as HTML.")
    body = reviews.render_finding_body(f)
    assert "Suggested fix:" not in body
    assert "```" not in body


def test_render_summary_with_pr_summary():
    summary = reviews.render_summary(
        "o", "r", 1, [], [], "vllm", pr_summary="Adds retries to the webhook handler."
    )
    assert "<details><summary>PR summary</summary>" in summary
    assert "Adds retries to the webhook handler." in summary
    # Placed right after the bot marker, before the review heading.
    assert summary.index("PR summary") < summary.index("## PRism review")
    assert summary.index("<!-- PRism -->") < summary.index("PR summary")


def test_render_summary_without_pr_summary():
    summary = reviews.render_summary("o", "r", 1, [], [], "vllm")
    assert "PR summary" not in summary


def test_render_summary_with_none_pr_summary():
    summary = reviews.render_summary("o", "r", 1, [], [], "vllm", pr_summary=None)
    assert "PR summary" not in summary


# --- nitpick / actionable split ---


def _mk(severity, confidence, path="a.py", line=1, title="Test finding title"):
    return make_finding(path=path, line=line, severity=severity, confidence=confidence, title=title)


def test_split_actionable_high_critical_always_actionable():
    for sev in (Severity.high, Severity.critical):
        actionable, nitpicks = reviews.split_actionable([_mk(sev, 0.1)], 0.9)
        assert len(actionable) == 1 and not nitpicks


def test_split_actionable_confidence_bar_boundary():
    # >= bar is actionable, just below is a nitpick.
    actionable, _ = reviews.split_actionable([_mk(Severity.medium, 0.9)], 0.9)
    assert len(actionable) == 1
    _, nitpicks = reviews.split_actionable([_mk(Severity.medium, 0.899)], 0.9)
    assert len(nitpicks) == 1


def test_split_actionable_low_severity_high_confidence_is_actionable():
    actionable, nitpicks = reviews.split_actionable([_mk(Severity.low, 1.0)], 0.9)
    assert len(actionable) == 1 and not nitpicks


def test_split_actionable_custom_bar():
    actionable, nitpicks = reviews.split_actionable([_mk(Severity.medium, 0.85)], 0.8)
    assert len(actionable) == 1 and not nitpicks
    _, nitpicks = reviews.split_actionable([_mk(Severity.medium, 0.75)], 0.8)
    assert len(nitpicks) == 1


def test_split_actionable_mixed():
    findings = [
        _mk(Severity.high, 0.5),
        _mk(Severity.low, 0.5),
        _mk(Severity.medium, 0.95),
    ]
    actionable, nitpicks = reviews.split_actionable(findings, 0.9)
    assert len(actionable) == 2
    assert len(nitpicks) == 1
    assert nitpicks[0].severity == Severity.low


# --- merge score / verdict ---


def test_compute_merge_score_zero_findings_is_one():
    assert reviews.compute_merge_score([]) == 1.0


def test_compute_merge_score_weights():
    assert reviews.compute_merge_score([_mk(Severity.low, 1.0)]) == pytest.approx(0.95)
    assert reviews.compute_merge_score([_mk(Severity.medium, 0.5)]) == pytest.approx(0.925)
    assert reviews.compute_merge_score([_mk(Severity.high, 1.0)]) == pytest.approx(0.65)
    assert reviews.compute_merge_score([_mk(Severity.critical, 1.0)]) == pytest.approx(0.4)


def test_compute_merge_score_clamped():
    assert reviews.compute_merge_score([_mk(Severity.critical, 1.0)] * 3) == 0.0
    assert 0.0 <= reviews.compute_merge_score([_mk(Severity.low, 0.1)]) <= 1.0


def test_merge_verdict_thresholds():
    assert reviews.merge_verdict(1.0) == "Safe to merge"
    assert reviews.merge_verdict(0.85) == "Safe to merge"
    assert reviews.merge_verdict(0.849) == "Needs careful review"
    assert reviews.merge_verdict(0.5) == "Needs careful review"
    assert reviews.merge_verdict(0.499) == "Changes requested"
    assert reviews.merge_verdict(0.0) == "Changes requested"


# --- effort ---


def test_estimate_effort_format():
    assert reviews.estimate_effort(0, []) == "1/5 · ~5 min"
    assert reviews.estimate_effort(4, []) == "2/5 · ~10 min"
    assert reviews.estimate_effort(8, [_mk(Severity.high, 0.9)]) == "4/5 · ~20 min"


def test_estimate_effort_capped_at_five():
    assert reviews.estimate_effort(100, [_mk(Severity.critical, 1.0)] * 10) == "5/5 · ~25 min"


# --- summary rendering of new blocks ---


def test_render_summary_nitpick_details_only():
    nitpicks = [
        make_finding(
            path="style.py",
            line=7,
            severity=Severity.low,
            confidence=0.5,
            title="Long line",
            explanation="This line is a bit too long for readability.",
        )
    ]
    summary = reviews.render_summary("o", "r", 1, [], [], "vllm", nitpicks=nitpicks)
    assert "Nitpick comments (1)" in summary
    assert "low severity/confidence, safe to ignore" in summary
    assert "**style.py:7**" in summary
    assert "Long line" in summary


def test_nitpicks_never_inline():
    nitpicks = [_mk(Severity.low, 0.5, path="n.py", line=2, title="Nitpick title")]
    actionable, nitpicks_out = reviews.split_actionable(nitpicks, 0.9)
    assert not actionable
    payload = reviews.build_review_payload("sha", "body", actionable)
    assert payload["comments"] == []
    for c in payload["comments"]:
        assert "Nitpick title" not in c["body"]
    # And the summary lists them only in the collapsed block.
    summary = reviews.render_summary("o", "r", 1, actionable, [], "vllm", nitpicks=nitpicks_out)
    assert "Nitpick comments (1)" in summary


def test_render_summary_verdict_line_with_score():
    summary = reviews.render_summary("o", "r", 1, [], [], "vllm")
    assert "**Verdict:** Safe to merge (score 1.00)" in summary


def test_render_summary_verdict_line_not_shown_when_incomplete():
    summary = reviews.render_summary("o", "r", 1, [], [], "vllm", ["a.py: boom"])
    assert "Review incomplete" in summary
    assert "**Verdict:**" not in summary


def test_render_summary_verdict_and_effort_in_pr_summary_block_without_text():
    summary = reviews.render_summary(
        "o", "r", 1, [], [], "vllm", verdict="Needs careful review", effort="2/5 · ~10 min"
    )
    assert "<details><summary>PR summary</summary>" in summary
    assert "**Verdict:** Needs careful review" in summary
    assert "**Estimated effort:** 2/5 · ~10 min" in summary


def test_render_agent_prompt_format():
    findings = [
        make_finding(
            path="db.py",
            line=12,
            severity=Severity.high,
            category="security",
            title="SQL injection",
            explanation="User input flows into the query unsanitized. This is bad.",
            suggestion="use params",
        )
    ]
    block = reviews.render_agent_prompt(findings)
    assert "<details><summary>Prompt for AI agents</summary>" in block
    assert "```text" in block
    assert "db.py:12 [HIGH/security] SQL injection" in block
    assert "Suggested fix: use params" in block


def test_render_agent_prompt_empty_suggestion_says_none():
    block = reviews.render_agent_prompt([_mk(Severity.medium, 0.95)])
    assert "Suggested fix: none" in block


def test_render_summary_appends_agent_prompt_when_findings_exist():
    summary = reviews.render_summary("o", "r", 1, [_mk(Severity.high, 0.9)], [], "vllm")
    assert "<details><summary>Prompt for AI agents</summary>" in summary
    assert "a.py:1 [HIGH/correctness]" in summary


def test_render_summary_no_agent_prompt_without_findings():
    summary = reviews.render_summary("o", "r", 1, [], [], "vllm")
    assert "Prompt for AI agents" not in summary


# --- payload event + labels ---


def test_build_review_payload_default_event_comment():
    payload = reviews.build_review_payload("sha", "body", [])
    assert payload["event"] == "COMMENT"


def test_build_review_payload_approve_event():
    payload = reviews.build_review_payload("sha", "body", [], event="APPROVE")
    assert payload["event"] == "APPROVE"


class _FakeLabelClient:
    def __init__(self, fail=False):
        self.posts: list[tuple[str, dict]] = []
        self.fail = fail

    async def post(self, url, json=None):
        if self.fail:
            raise RuntimeError("403: no permission")
        self.posts.append((url, json))
        return {}


@pytest.mark.asyncio
async def test_post_pr_labels_safe_to_merge():
    client = _FakeLabelClient()
    await reviews.post_pr_labels(client, "o", "r", 42, "Safe to merge")
    assert client.posts == [("/repos/o/r/issues/42/labels", {"labels": ["prism: safe-to-merge"]})]


@pytest.mark.asyncio
async def test_post_pr_labels_other_verdicts_map_to_careful():
    for verdict in ("Needs careful review", "Changes requested"):
        client = _FakeLabelClient()
        await reviews.post_pr_labels(client, "o", "r", 42, verdict)
        assert client.posts[0][1] == {"labels": ["prism: needs-careful-review"]}


@pytest.mark.asyncio
async def test_post_pr_labels_never_raises():
    client = _FakeLabelClient(fail=True)
    await reviews.post_pr_labels(client, "o", "r", 42, "Safe to merge")  # must not raise
