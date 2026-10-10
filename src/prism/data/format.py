"""
SFT instruction formatting: (system, user, assistant) triples.

- ``user``: repo/PR/file context + the diff with new-file line numbers.
  The human reviewer comment is deliberately NOT included — it is the label
  source, and it will not exist at inference time.
- ``assistant``: strict JSON ``{"findings": [Finding, ...]}`` matching
  ``prism.review.schemas.Finding``.

Labels come from two sources:
  - handcrafted ``finding`` overrides on the sample (validated strictly —
    curated data must be exact), or
  - :func:`heuristic_finding_from_comment`, a weak labeler that maps the
    reviewer comment onto the Finding schema (title = first line, category
    from keyword matching, suggestion = first fenced code block, fixed
    confidence 0.6). Heuristic labels are noisy by nature; prefer handcrafted
    labels for eval-quality data.

Also provides a tokenizer-free length check: ``estimate_tokens`` uses the
~4-chars-per-token BPE heuristic so overlong examples can be filtered without
a transformers dependency.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from prism.data.models import CleanSample, TrainingRecord
from prism.diff.hunks import parse_hunks
from prism.logging import get_logger
from prism.review.schemas import Finding, Severity

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are PRism, a senior staff engineer reviewing pull request diffs. "
    "Report only real, defensible issues — no nitpicks, no praise, no summaries. "
    "Output STRICT JSON with this exact shape:\n"
    '{"findings": [{"path": "<repo-relative file path>", '
    '"line": <new-file line number>, '
    '"severity": "<critical|high|medium|low>", '
    '"category": "<security|correctness|performance|concurrency|style|testing|...>", '
    '"title": "<short title>", "explanation": "<why this is a problem>", '
    '"suggestion": "<fixed code snippet, may be empty>", '
    '"confidence": <0.0-1.0>, "language_hint": "<code fence language>"}]}\n'
    'If there is nothing worth flagging, output {"findings": []}.'
)

# avg chars per token for BPE tokenizers on code/English mix.
_CHARS_PER_TOKEN = 4

_FENCE_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

# (category, keywords) — checked in order; first match wins.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "security",
        (
            "sql injection",
            "xss",
            "csrf",
            "ssrf",
            "injection",
            "vulnerab",
            "secret",
            "password",
            "credential",
            "auth",
            "token",
            "encrypt",
            "sanitiz",
            "escape",
            "cve",
        ),
    ),
    (
        "concurrency",
        (
            "race condition",
            "deadlock",
            "data race",
            "goroutine",
            "thread-safe",
            "mutex",
            "atomic",
        ),
    ),
    (
        "performance",
        (
            "n+1",
            "slow",
            "o(n",
            "perf",
            "latency",
            "cache",
            "memory leak",
            "blocking",
            "inefficien",
        ),
    ),
    (
        "correctness",
        (
            "bug",
            "wrong",
            "incorrect",
            "off-by-one",
            "off by one",
            "edge case",
            "null",
            "none",
            "undefined",
            "exception",
            "crash",
            "panic",
            "unhandled",
        ),
    ),
    ("testing", ("test", "coverage", "flaky", "mock")),
    ("style", ("lint", "style", "naming", "readab", "pep8", "unused")),
    ("documentation", ("docstring", "readme", "document")),
]


def numbered_diff(patch: str) -> str:
    """
    Render a unified diff with new-file line numbers in the left column.

    Deleted lines ("-") have no new-file line number and get a blank gutter.
    """
    lines: list[str] = []
    for hunk in parse_hunks(patch):
        new_line = hunk.new_start
        for raw in hunk.lines:
            if raw.startswith("+"):
                lines.append(f"{new_line:>6} | {raw}")
                new_line += 1
            elif raw.startswith("-"):
                lines.append(f"{'':>6} | {raw}")
            else:  # context line
                lines.append(f"{new_line:>6} | {raw}")
                new_line += 1
    return "\n".join(lines)


def build_user_prompt(sample: CleanSample) -> str:
    """Build the model input: context header + numbered diff (no reviewer comment)."""
    return (
        f"Repository: {sample.repo} — PR #{sample.pr_number}: {sample.pr_title}\n"
        f"File: {sample.path} (language: {sample.language or 'unknown'})\n\n"
        "Review the diff below. The left column shows new-file line numbers "
        '("-" rows are deletions with no new-file line number).\n\n'
        "```diff\n"
        f"{numbered_diff(sample.patch)}\n"
        "```"
    )


def heuristic_category(text: str) -> str:
    """Guess the finding category from keyword matching; 'general' fallback."""
    lowered = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return category
    return "general"


def extract_suggestion(body: str) -> str:
    """Extract the first fenced code block from a comment body; '' if none."""
    match = _FENCE_RE.search(body)
    return match.group(1).strip() if match else ""


def heuristic_finding_from_comment(sample: CleanSample) -> Finding:
    """
    Weak label: map a reviewer comment onto the Finding schema.

    Title = first non-empty line (<=160 chars); explanation = full body;
    suggestion = first fenced code block; severity = high for security,
    medium otherwise; confidence fixed at 0.6 (weak label marker).
    """
    body = sample.comment_body.strip()
    first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), body[:160])
    title = first_line[:160] or "Reviewer flagged an issue"
    category = heuristic_category(f"{title}\n{body}")
    severity = Severity.high if category == "security" else Severity.medium
    return Finding(
        path=sample.path,
        line=sample.comment_line,
        severity=severity,
        category=category,
        title=title,
        explanation=body[:2000],
        suggestion=extract_suggestion(body),
        confidence=0.6,
        language_hint=sample.language,
    )


def finding_for_sample(sample: CleanSample) -> tuple[Finding, str]:
    """
    Resolve the label for a sample.

    Returns ``(finding, source)`` where source is "handcrafted" or
    "heuristic". Handcrafted overrides are validated strictly — curated data
    must match the schema exactly.
    """
    if sample.finding is not None:
        try:
            return Finding.model_validate(sample.finding), "handcrafted"
        except Exception as exc:
            raise ValueError(f"invalid handcrafted finding for {sample.sample_id}: {exc}") from exc
    return heuristic_finding_from_comment(sample), "heuristic"


def build_record(sample: CleanSample) -> TrainingRecord:
    """Build one SFT (system, user, assistant) triple from a clean sample."""
    finding, source = finding_for_sample(sample)
    assistant = json.dumps({"findings": [finding.model_dump(mode="json")]}, indent=2)
    return TrainingRecord(
        system=SYSTEM_PROMPT,
        user=build_user_prompt(sample),
        assistant=assistant,
        meta={
            "sample_id": sample.sample_id,
            "repo": sample.repo,
            "pr_number": sample.pr_number,
            "path": sample.path,
            "language": sample.language,
            "comment_id": sample.comment_id,
            "source": source,
        },
    )


def build_records(samples: list[CleanSample]) -> list[TrainingRecord]:
    return [build_record(s) for s in samples]


# -- length check (no tokenizer dependency) --------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate via the ~4-chars-per-token BPE heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def record_token_estimate(record: TrainingRecord) -> int:
    return (
        estimate_tokens(record.system)
        + estimate_tokens(record.user)
        + estimate_tokens(record.assistant)
    )


def filter_by_length(
    records: list[TrainingRecord], max_tokens: int
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    """Split records into (kept, dropped) by estimated total token length."""
    kept: list[TrainingRecord] = []
    dropped: list[TrainingRecord] = []
    for record in records:
        (kept if record_token_estimate(record) <= max_tokens else dropped).append(record)
    log.info("length_filter", kept=len(kept), dropped=len(dropped), max_toks=max_tokens)
    return kept, dropped


def write_jsonl(path: str | Path, records: list[TrainingRecord]) -> None:
    """Write training records as JSONL (one object per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.model_dump(mode="json")) + "\n")
    log.info("records_written", path=str(p), count=len(records))


def read_jsonl(path: str | Path) -> list[TrainingRecord]:
    """Read training records back from JSONL."""
    records: list[TrainingRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(TrainingRecord.model_validate(json.loads(line)))
    return records
