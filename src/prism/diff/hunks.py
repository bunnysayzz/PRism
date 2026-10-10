"""
Unified-diff hunk parsing.

GitHub returns each changed file's `patch` as a unified diff. Review comments
must anchor to lines that are part of the diff (on the new-file side), so we
compute the set of commentable new-file line numbers per file.

Hunk header format:
    @@ -<old_start>[,<old_count>] +<new_start>[,<new_count>] @@
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]  # raw diff lines (with +/-/space prefixes)


def parse_hunks(patch: str | None) -> list[Hunk]:
    """Parse a unified diff patch into hunks. Returns [] for None/empty patch."""
    if not patch:
        return []
    hunks: list[Hunk] = []
    current: list[str] = []
    header: re.Match[str] | None = None
    meta: tuple[int, int, int, int] | None = None

    def flush() -> None:
        nonlocal header, meta, current
        if header is not None and meta is not None:
            old_start, old_count, new_start, new_count = meta
            hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=tuple(current),
                )
            )
        header, meta, current = None, None, []

    for raw_line in patch.splitlines():
        m = _HUNK_RE.match(raw_line)
        if m:
            flush()
            header = m
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            meta = (old_start, old_count, new_start, new_count)
        elif header is not None:
            # "\ No newline at end of file" markers are not diff content.
            if not raw_line.startswith("\\"):
                current.append(raw_line)
    flush()
    return hunks


def commentable_lines(patch: str | None) -> set[int]:
    """
    New-file line numbers that are part of the diff (added + context lines).

    GitHub only accepts review comments on lines within the diff. Deleted
    lines ("-") have no new-file line number and are excluded.
    """
    commentable: set[int] = set()
    for hunk in parse_hunks(patch):
        new_line = hunk.new_start
        for line in hunk.lines:
            if line.startswith("+"):
                commentable.add(new_line)
                new_line += 1
            elif line.startswith("-"):
                continue  # no new-file line number
            else:
                # context line (starts with a space, or empty edge cases)
                commentable.add(new_line)
                new_line += 1
    return commentable


def extract_hunk_context(patch: str | None, line: int, radius: int = 6) -> str:
    """
    Return the hunk text surrounding `line` (new-file numbering), giving the
    model the local diff context for that finding. Empty string if not found.
    """
    for hunk in parse_hunks(patch):
        new_line = hunk.new_start
        span: list[tuple[int | None, str]] = []  # (new-file line or None, raw)
        for raw in hunk.lines:
            if raw.startswith("+"):
                span.append((new_line, raw))
                new_line += 1
            elif raw.startswith("-"):
                span.append((None, raw))
            else:
                span.append((new_line, raw))
                new_line += 1
        numbered = [ln for ln, _ in span if ln is not None]
        if numbered and min(numbered) <= line <= max(numbered):
            idx = next(i for i, (ln, _) in enumerate(span) if ln == line)
            lo, hi = max(0, idx - radius), idx + radius + 1
            return "\n".join(raw for _, raw in span[lo:hi])
    return ""
