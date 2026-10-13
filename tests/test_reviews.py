"""Tests for the GitHub review payload shape and markdown rendering."""

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
