"""
Deterministic secret scanning over unified diffs.

Runs BEFORE any model call: cheap, deterministic, and safe to run on every
pull request. Only added lines are scanned, so secrets that already existed
(and were removed or left untouched) never fire.

No network access, no model involved. Logs carry only counts and file
paths - never matched secret values, never line content.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from prism.review.schemas import Finding, Severity

log = logging.getLogger(__name__)

SECRET_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "AWS access key ID",
        re.compile(r"AKIA[0-9A-Z]{16}"),
        "Exposed secret: AWS access key ID",
    ),
    (
        "GitHub token",
        re.compile(r"(?:ghp_|gho_|ghu_|github_pat_)[A-Za-z0-9_]{10,}"),
        "Exposed secret: GitHub token",
    ),
    (
        "generic API key",
        re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][^'\"]{16,}['\"]"),
        "Exposed secret: generic API key",
    ),
    (
        "private key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        "Exposed secret: private key",
    ),
    (
        "database connection string",
        re.compile(r"(?i)(?:mongodb|postgres|mysql|redis)://[^/\s:]+:[^/\s@]+@"),
        "Exposed secret: database connection string",
    ),
    (
        "Slack token",
        re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
        "Exposed secret: Slack token",
    ),
    (
        "Stripe live key",
        re.compile(r"sk_live_[A-Za-z0-9]{16,}"),
        "Exposed secret: Stripe live key",
    ),
]

_ROTATION_ADVICE: dict[str, str] = {
    "AWS access key ID": (
        "Revoke the key in AWS IAM immediately, audit recent usage, and load "
        "it from a secret store or environment variable instead of the diff."
    ),
    "GitHub token": (
        "Revoke the token in GitHub settings right away and replace it with a "
        "secret injected at deploy time, never committed."
    ),
    "generic API key": (
        "Rotate the key with the provider, then move it out of the codebase "
        "into a secret manager or environment variable."
    ),
    "private key": (
        "Treat this key as compromised: generate a new keypair, revoke the "
        "old public key everywhere, and never commit private key material."
    ),
    "database connection string": (
        "Rotate the database credentials immediately and move the connection "
        "string to a secret store or environment variable."
    ),
    "Slack token": (
        "Revoke the token in the Slack app admin console and re-issue it "
        "through a secret store, not the repository."
    ),
    "Stripe live key": (
        "Roll the key in the Stripe dashboard immediately, check the audit "
        "log for misuse, and store it in a secret manager."
    ),
}

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_MIN_EXPLANATION_LEN = 10


def _explanation(name: str) -> str:
    """Why + rotation advice for a detected secret of the given name."""
    return (
        f"Hardcoded {name} added in this diff. Anyone with read access to the "
        f"repository can exfiltrate it and impersonate the owner. "
        f"{_ROTATION_ADVICE[name]}"
    )


def _iter_added_lines(patch: str) -> Iterator[tuple[int, str]]:
    """
    Yield (new-file line number, line content) for added lines only.

    Removed lines and context lines are skipped; so are the "+++"/"---" file
    headers. Lines are only yielded once a valid hunk header established the
    new-file line numbering.
    """
    new_line = 0
    for raw in patch.splitlines():
        if raw.startswith("@@"):
            match = _HUNK_HEADER_RE.match(raw)
            if match:
                new_line = int(match.group(1))
            continue
        if raw.startswith("++") or raw.startswith("---"):
            # "+++ b/<path>" / "--- a/<path>" headers: never a finding.
            # ("+" prefix only, not "++": an added line whose content starts
            # with "+" is out of scope by design.)
            continue
        if raw.startswith("+"):
            if new_line > 0:
                yield new_line, raw[1:]
            new_line += 1
        elif raw.startswith(" "):
            new_line += 1
        # "-" removed lines and "\\ No newline at end of file" markers carry
        # no new-file line number, so they are ignored.


def scan_patches(patches: list[tuple[str, str]]) -> list[Finding]:
    """
    Scan added lines of unified-diff patches for hardcoded secrets.

    Each patch is (path, unified diff). Returns one critical security
    Finding per hit, with confidence 1.0. Identical (name, path, line) hits
    are deduplicated. Logs only counts and paths, never matched values.
    """
    findings: list[Finding] = []
    seen: set[tuple[str, str, int]] = set()
    for path, patch in patches:
        count = 0
        for line_no, content in _iter_added_lines(patch):
            for name, pattern, title in SECRET_PATTERNS:
                if not pattern.search(content):
                    continue
                key = (name, path, line_no)
                if key in seen:
                    continue
                seen.add(key)
                explanation = _explanation(name)
                assert len(explanation) >= _MIN_EXPLANATION_LEN
                findings.append(
                    Finding(
                        path=path,
                        line=line_no,
                        severity=Severity.critical,
                        category="security",
                        title=title,
                        explanation=explanation,
                        suggestion="",
                        confidence=1.0,
                    )
                )
                count += 1
        log.info("secrets scan: %d finding(s) in %s", count, path)
    return findings
