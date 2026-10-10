"""Tests for finding dedup (fingerprint + existing-comment matching)."""

from prism.review.dedup import (
    dedupe_against_existing,
    embed_fingerprint,
    extract_fingerprints,
    fingerprint,
)
from prism.review.engine import make_finding


def test_fingerprint_stable():
    f1 = make_finding(title="SQL injection", explanation="Tainted input reaches the query.")
    f2 = make_finding(title="SQL injection", explanation="Tainted input reaches the query.")
    assert fingerprint(f1) == fingerprint(f2)
    assert len(fingerprint(f1)) == 16


def test_fingerprint_ignores_line_shift():
    # Same issue, line moved after a rebase -> still the same finding.
    f1 = make_finding(line=10, title="SQL injection")
    f2 = make_finding(line=14, title="SQL injection")
    assert fingerprint(f1) == fingerprint(f2)


def test_fingerprint_differs_across_issues():
    f1 = make_finding(title="SQL injection")
    f2 = make_finding(title="Missing await")
    assert fingerprint(f1) != fingerprint(f2)


def test_embed_and_extract_roundtrip():
    f = make_finding()
    body = embed_fingerprint("some comment", fingerprint(f))
    assert extract_fingerprints(body) == {fingerprint(f)}


def test_dedupe_drops_already_posted():
    posted = make_finding(title="SQL injection")
    new_same = make_finding(title="SQL injection", line=99)  # shifted line
    new_other = make_finding(title="Hardcoded secret")
    existing = [{"body": embed_fingerprint("old comment", fingerprint(posted))}]
    kept = dedupe_against_existing([new_same, new_other], existing)
    assert [f.title for f in kept] == ["Hardcoded secret"]


def test_dedupe_no_existing_keeps_all():
    findings = [make_finding(title="Issue A"), make_finding(title="Issue B")]
    assert len(dedupe_against_existing(findings, [])) == 2


def test_dedupe_ignores_non_bot_comments():
    findings = [make_finding(title="Issue A")]
    existing = [{"body": "a human comment without any marker"}]
    assert len(dedupe_against_existing(findings, existing)) == 1
