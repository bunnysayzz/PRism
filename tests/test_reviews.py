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
    assert "🟠" in body
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
    assert "## 🤖 PR Review" in summary
    assert "critical: 2" in summary
    assert "low: 1" in summary
    assert "lock.json" in summary


def test_render_summary_no_findings():
    summary = reviews.render_summary("o", "r", 1, [], [], "vllm")
    assert "looks clean" in summary
