"""Tests for deterministic secret scanning over diffs."""

import logging

import pytest

from prism.review.schemas import Severity
from prism.review.secrets import SECRET_PATTERNS, scan_patches


def _added_patch(added_line: str, new_start: int = 10) -> tuple[str, str]:
    """A patch with one added line between two context lines."""
    patch = f"@@ -{new_start},2 +{new_start},3 @@\n context before\n+{added_line}\n context after\n"
    return ("src/config.py", patch)


# (name, realistic added line) - one case per SECRET_PATTERNS entry.
PATTERN_CASES = [
    ("AWS access key ID", 'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"'),
    ("GitHub token", 'GITHUB_TOKEN = "ghp_abcdefghij1234567890"'),
    ("generic API key", 'api_key = "sk-abcdef1234567890xyz"'),
    ("private key", "-----BEGIN RSA PRIVATE KEY-----"),
    (
        "database connection string",
        'DATABASE_URL = "postgres://admin:s3cret-pw@db.internal:5432/app"',
    ),
    ("Slack token", 'SLACK_TOKEN = "xoxb-1234567890-abcdefghij"'),
    ("Stripe live key", 'STRIPE_KEY = "sk_live_51HabcDEFghiJKLmnopqr"'),
]


def test_every_pattern_has_a_case():
    pattern_names = [name for name, _, _ in SECRET_PATTERNS]
    assert sorted(pattern_names) == sorted(name for name, _ in PATTERN_CASES)


@pytest.mark.parametrize("name,added_line", PATTERN_CASES)
def test_each_pattern_fires_with_correct_line_number(name, added_line):
    findings = scan_patches([_added_patch(added_line, new_start=10)])
    assert len(findings) == 1
    finding = findings[0]
    assert finding.path == "src/config.py"
    assert finding.line == 11  # context at 10, added line at 11
    assert finding.severity == Severity.critical
    assert finding.category == "security"
    assert finding.title == f"Exposed secret: {name}"
    assert len(finding.explanation) >= 10
    lowered = finding.explanation.lower()
    assert "rotate" in lowered or "revoke" in lowered or "roll" in lowered
    assert finding.suggestion == ""
    assert finding.confidence == 1.0


def test_line_numbers_track_multiple_hunks():
    patch = (
        "@@ -1,2 +1,3 @@\n"
        " context\n"
        "+clean = 1\n"
        " more context\n"
        "@@ -20,2 +21,3 @@\n"
        " context\n"
        '+AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\n'
        " more context\n"
    )
    findings = scan_patches([("src/a.py", patch)])
    assert len(findings) == 1
    assert findings[0].line == 22


def test_removed_lines_do_not_fire():
    patch = (
        '@@ -1,3 +1,2 @@\n-AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\n context\n more context\n'
    )
    assert scan_patches([("src/a.py", patch)]) == []


def test_context_lines_do_not_fire():
    patch = (
        '@@ -1,3 +1,3 @@\n AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\n context\n more context\n'
    )
    assert scan_patches([("src/a.py", patch)]) == []


def test_clean_diff_returns_empty():
    patch = "@@ -1,2 +1,3 @@\n def f(x):\n+    return x * 2\n # done\n"
    assert scan_patches([("src/a.py", patch)]) == []


def test_multiple_secrets_all_reported():
    findings = scan_patches(
        [
            _added_patch('AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"', new_start=1),
            _added_patch('STRIPE_KEY = "sk_live_51HabcDEFghiJKLmnopqr"', new_start=5),
        ]
    )
    assert len(findings) == 2
    assert {(f.path, f.line, f.title) for f in findings} == {
        ("src/config.py", 2, "Exposed secret: AWS access key ID"),
        ("src/config.py", 6, "Exposed secret: Stripe live key"),
    }


def test_duplicate_hits_deduplicated():
    # Same (name, path, line) reported twice must collapse to one finding.
    patch = _added_patch('AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"', new_start=10)
    findings = scan_patches([patch, patch])
    assert len(findings) == 1


def test_secret_values_never_logged(caplog):
    secret = "AKIAIOSFODNN7EXAMPLE"
    with caplog.at_level(logging.INFO, logger="prism.review.secrets"):
        scan_patches([_added_patch(f'KEY = "{secret}"')])
    for record in caplog.records:
        assert secret not in record.getMessage()
        assert "ghp_" not in record.getMessage()


def test_pattern_shapes():
    # SECRET_PATTERNS is a list of (name, compiled regex, title).
    for name, pattern, title in SECRET_PATTERNS:
        assert isinstance(name, str) and name
        assert hasattr(pattern, "search")
        assert title == f"Exposed secret: {name}"
