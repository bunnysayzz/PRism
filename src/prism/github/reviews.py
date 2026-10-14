"""
Posting pull-request reviews via the GitHub REST API.

Endpoint: POST /repos/{owner}/{repo}/pulls/{pr_number}/reviews
Payload shape (exact):
    {
        "commit_id": "<head sha>",
        "event": "COMMENT" | "APPROVE",  # COMMENT posts without approving
        "body": "<summary markdown>",
        "comments": [
            {"path": "...", "line": <new-file line>, "side": "RIGHT", "body": "..."},
            ...
        ],
    }

`line` is a line number in the NEW file version (the PR head), and
`side: "RIGHT"` anchors the comment to the new-file side of the diff.
GitHub rejects comments on lines that are not part of the diff - the engine
validates every finding line against commentable_lines() before posting.

Only *actionable* findings become inline comments (see split_actionable);
nitpicks render as a collapsed list inside the summary body.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from prism.logging import get_logger
from prism.review.dedup import embed_fingerprint, fingerprint
from prism.review.schemas import Severity

if TYPE_CHECKING:
    from prism.github.client import GitHubClient
    from prism.review.schemas import Finding

log = get_logger(__name__)

_BOT_MARKER = "<!-- PRism -->"

# Max lines for a suggestion to qualify as a drop-in ```suggestion block.
_SUGGESTION_MAX_LINES = 12

# Severities that always post as inline comments, regardless of confidence.
_ACTIONABLE_SEVERITIES = (Severity.high, Severity.critical)

# Merge-confidence score: expected impact of one finding of this severity at
# confidence 1.0. Score = 1 - sum(weight * confidence), clamped to [0, 1].
_SEV_SCORE_WEIGHTS: dict[Severity, float] = {
    Severity.low: 0.05,
    Severity.medium: 0.15,
    Severity.high: 0.35,
    Severity.critical: 0.60,
}


def _looks_like_code(text: str) -> bool:
    """True when the text reads as code rather than prose.

    Prose suggestions get the language-fenced treatment; code-shaped ones
    are candidates for GitHub's one-click ```suggestion blocks.
    """
    return bool(re.search(r"[{}()\[\]=;<>]", text))


def _is_drop_in_suggestion(suggestion: str) -> bool:
    """Heuristic: is the suggestion a drop-in replacement for the flagged lines?"""
    if not suggestion or not suggestion.strip():
        return False
    lines = suggestion.splitlines()
    if len(lines) > _SUGGESTION_MAX_LINES:
        return False
    if "```" in suggestion:
        return False  # would break the fence
    return _looks_like_code(suggestion)


def _one_liner(text: str, limit: int = 200) -> str:
    """Collapse text to a single line, truncated to `limit` chars."""
    line = " ".join(text.split())
    if len(line) <= limit:
        return line
    return line[: limit - 1].rstrip() + "..."


def split_actionable(
    findings: list[Finding], verify_threshold: float
) -> tuple[list[Finding], list[Finding]]:
    """Split findings into (actionable, nitpicks).

    Actionable = severity high/critical, or confidence >= the bar. The caller
    passes the auto-post confidence bar (e.g. cfg's confidence_auto_post);
    the parameter keeps the name from the original spec.
    """
    actionable: list[Finding] = []
    nitpicks: list[Finding] = []
    for f in findings:
        if f.severity in _ACTIONABLE_SEVERITIES or f.confidence >= verify_threshold:
            actionable.append(f)
        else:
            nitpicks.append(f)
    return actionable, nitpicks


def compute_merge_score(findings: list[Finding]) -> float:
    """Merge-confidence score in [0.0, 1.0].

    Starts at 1.0; each finding subtracts sev_weight * confidence.
    Zero findings -> 1.0.
    """
    score = 1.0
    for f in findings:
        score -= _SEV_SCORE_WEIGHTS.get(f.severity, 0.15) * f.confidence
    return max(0.0, min(1.0, score))


def merge_verdict(score: float) -> str:
    """Human verdict for a merge-confidence score."""
    if score >= 0.85:
        return "Safe to merge"
    if score >= 0.5:
        return "Needs careful review"
    return "Changes requested"


def estimate_effort(num_files: int, findings: list[Finding]) -> str:
    """Rough review/fix effort on a 1-5 scale, e.g. "2/5 · ~10 min".

    Level derives from file count plus severity-weighted finding count
    (rank: low=0, medium=1, high=2, critical=3); minutes = level * 5.
    """
    weighted = sum(f.severity.rank() for f in findings)
    level = min(5, max(1, 1 + num_files // 4 + weighted // 2))
    return f"{level}/5 · ~{level * 5} min"


def render_agent_prompt(findings: list[Finding]) -> str:
    """Machine-readable restatement of findings for coding agents.

    No model calls - purely renders from the Finding schema.
    """
    lines = ["<details><summary>Prompt for AI agents</summary>", "", "```text"]
    for f in findings:
        lines.append(
            f"{f.path}:{f.line} [{f.severity.value.upper()}/{f.category}] "
            f"{f.title} — {_one_liner(f.explanation, 280)}"
        )
        fix = f.suggestion.strip() if f.suggestion else ""
        lines.append(f"Suggested fix: {fix if fix else 'none'}")
        lines.append("")
    lines += ["```", "", "</details>", ""]
    return "\n".join(lines)


def render_finding_body(finding: Finding) -> str:
    """Render one finding as review-comment markdown."""
    parts = [
        _BOT_MARKER,
        f"**{finding.severity.value.upper()}** · `{finding.category}`",
        "",
        f"**{finding.title}**",
        "",
        finding.explanation,
    ]
    if finding.suggestion:
        if _is_drop_in_suggestion(finding.suggestion):
            # GitHub renders a one-click "Commit suggestion" button.
            parts += ["", "Suggested fix:", "", "```suggestion", finding.suggestion, "```"]
        else:
            fence = f"```{finding.language_hint or ''}".rstrip()
            parts += ["", "Suggested fix:", "", fence, finding.suggestion, "```"]
    parts += ["", f"_confidence: {finding.confidence:.2f}_"]
    return "\n".join(parts)


def render_summary(
    owner: str,
    repo: str,
    pr_number: int,
    findings: list[Finding],
    skipped_files: list[str],
    model: str,
    backend_errors: list[str] | None = None,
    pr_summary: str | None = None,
    nitpicks: list[Finding] | None = None,
    verdict: str | None = None,
    effort: str | None = None,
) -> str:
    errors = backend_errors or []
    nitpick_list = nitpicks or []
    all_findings = [*findings, *nitpick_list]
    score = compute_merge_score(all_findings)
    verdict_text = verdict or merge_verdict(score)
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
    lines = [
        _BOT_MARKER,
    ]
    if pr_summary or verdict or effort:
        # Verdict + effort ride in the top PR-summary block even when the
        # model produced no walkthrough text.
        lines += [
            "<details><summary>PR summary</summary>",
            "",
            f"**Verdict:** {verdict_text} (score {score:.2f})",
        ]
        if effort:
            lines += [f"**Estimated effort:** {effort}"]
        if pr_summary:
            lines += ["", pr_summary]
        lines += ["", "</details>", ""]
    lines += [
        "## PRism review",
        "",
    ]
    if errors and not findings:
        # The model never ran: say so loudly. Never claim "looks clean"
        # when the backend failed - that lie is worse than no review.
        lines.append(
            f"**Review incomplete** - PRism could not finish this review: "
            f"the backend failed on {len(errors)} file(s), so this is not "
            "a clean bill of health:"
        )
        lines.append("")
        for err in errors[:5]:
            lines.append(f"- `{err}`")
        if len(errors) > 5:
            lines.append(f"- ...and {len(errors) - 5} more")
    else:
        lines.append(
            f"PRism reviewed this pull request. Found **{len(findings)}** issue(s): "
            + (
                ", ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
                if counts
                else "none - no issues found"
            )
        )
        lines.append("")
        lines.append(f"**Verdict:** {verdict_text} (score {score:.2f})")
        if errors:
            lines += [
                "",
                f"_Note: the backend failed on {len(errors)} file(s); "
                "those files were not reviewed._",
            ]
    lines.append("")
    if nitpick_list:
        # Nitpicks never become inline comments - collapsed list only.
        lines += [
            f"<details><summary>Nitpick comments ({len(nitpick_list)}) — "
            "low severity/confidence, safe to ignore</summary>",
            "",
        ]
        for f in nitpick_list:
            lines.append(f"- **{f.path}:{f.line}** — {f.title}: {_one_liner(f.explanation)}")
        lines += ["", "</details>", ""]
    if all_findings:
        lines += [render_agent_prompt(all_findings)]
    if skipped_files:
        lines += [
            "<details><summary>Skipped files</summary>",
            "",
            *[f"- `{p}`" for p in skipped_files],
            "",
            "</details>",
            "",
        ]
    return "\n".join(lines)


def build_review_payload(
    commit_sha: str,
    summary: str,
    findings: list[Finding],
    event: str = "COMMENT",
) -> dict[str, Any]:
    """Build the exact JSON payload for POST .../pulls/{n}/reviews.

    event: "COMMENT" posts without approving; "APPROVE" approves the PR.
    """
    return {
        "commit_id": commit_sha,
        "event": event,
        "body": summary,
        "comments": [
            {
                "path": f.path,
                "line": f.line,
                "side": "RIGHT",
                # Fingerprint embedded as hidden HTML comment for dedup.
                "body": embed_fingerprint(render_finding_body(f), fingerprint(f)),
            }
            for f in findings
        ],
    }


async def post_review(
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
    commit_sha: str,
    summary: str,
    findings: list[Finding],
    event: str = "COMMENT",
) -> dict[str, Any]:
    """Post the review; returns the created review resource."""
    payload = build_review_payload(commit_sha, summary, findings, event=event)
    log.info(
        "posting_review",
        repo=f"{owner}/{repo}",
        pr=pr_number,
        findings=len(findings),
        event=event,
    )
    result: dict[str, Any] = await client.post(
        f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews", json=payload
    )
    return result


async def post_pr_labels(
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
    verdict: str,
) -> None:
    """Tag the PR with a prism verdict label.

    "Safe to merge" -> prism: safe-to-merge; anything else
    ("Needs careful review", "Changes requested") ->
    prism: needs-careful-review. Never raises: the Issues labels endpoint
    may be unavailable if the app lacks permission.
    """
    label = "prism: safe-to-merge" if verdict == "Safe to merge" else "prism: needs-careful-review"
    try:
        await client.post(
            f"/repos/{owner}/{repo}/issues/{pr_number}/labels",
            json={"labels": [label]},
        )
    except Exception as exc:
        log.warning("labels_failed", repo=f"{owner}/{repo}", pr=pr_number, error=str(exc))


async def fetch_existing_review_comments(
    client: GitHubClient, owner: str, repo: str, pr_number: int
) -> list[dict[str, Any]]:
    """
    Fetch existing review comments on the PR (for dedup).

    GET /repos/{owner}/{repo}/pulls/{pr_number}/comments - only returns
    comments made by our bot (filtered by the HTML marker).
    """
    comments = await client.paginate(f"/repos/{owner}/{repo}/pulls/{pr_number}/comments")
    return [c for c in comments if _BOT_MARKER in (c.get("body") or "")]


async def get_pr_head_sha(client: GitHubClient, owner: str, repo: str, pr_number: int) -> str:
    pr: dict[str, Any] = await client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
    sha: str = pr["head"]["sha"]
    return sha
