"""
Posting pull-request reviews via the GitHub REST API.

Endpoint: POST /repos/{owner}/{repo}/pulls/{pr_number}/reviews
Payload shape (exact):
    {
        "commit_id": "<head sha>",
        "event": "COMMENT",          # posts without approving/requesting changes
        "body": "<summary markdown>",
        "comments": [
            {"path": "...", "line": <new-file line>, "side": "RIGHT", "body": "..."},
            ...
        ],
    }

`line` is a line number in the NEW file version (the PR head), and
`side: "RIGHT"` anchors the comment to the new-file side of the diff.
GitHub rejects comments on lines that are not part of the diff — the engine
validates every finding line against commentable_lines() before posting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prism.logging import get_logger
from prism.review.dedup import embed_fingerprint, fingerprint

if TYPE_CHECKING:
    from prism.github.client import GitHubClient
    from prism.review.schemas import Finding

log = get_logger(__name__)

_SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
}

_BOT_MARKER = "<!-- PRism -->"


def render_finding_body(finding: Finding) -> str:
    """Render one finding as review-comment markdown."""
    emoji = _SEVERITY_EMOJI.get(finding.severity.value, "⚪")
    parts = [
        _BOT_MARKER,
        f"{emoji} **{finding.severity.value.upper()}** · `{finding.category}`",
        "",
        f"**{finding.title}**",
        "",
        finding.explanation,
    ]
    if finding.suggestion:
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
) -> str:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
    lines = [
        _BOT_MARKER,
        "## 🤖 PR Review",
        "",
        f"Reviewed with `{model}`. Found **{len(findings)}** issue(s): "
        + (
            ", ".join(f"{_SEVERITY_EMOJI.get(s, '')} {s}: {n}" for s, n in sorted(counts.items()))
            if counts
            else "none — looks clean! 🎉"
        ),
        "",
    ]
    if skipped_files:
        lines += [
            "<details><summary>Skipped files</summary>",
            "",
            *[f"- `{p}`" for p in skipped_files],
            "",
            "</details>",
            "",
        ]
    lines.append(
        "_React 👍/👎 or reply to a comment to train the reviewer — "
        "your feedback feeds the retraining loop._"
    )
    return "\n".join(lines)


def build_review_payload(
    commit_sha: str,
    summary: str,
    findings: list[Finding],
) -> dict[str, Any]:
    """Build the exact JSON payload for POST .../pulls/{n}/reviews."""
    return {
        "commit_id": commit_sha,
        "event": "COMMENT",
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
) -> dict[str, Any]:
    """Post the review; returns the created review resource."""
    payload = build_review_payload(commit_sha, summary, findings)
    log.info(
        "posting_review",
        repo=f"{owner}/{repo}",
        pr=pr_number,
        findings=len(findings),
    )
    result: dict[str, Any] = await client.post(
        f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews", json=payload
    )
    return result


async def fetch_existing_review_comments(
    client: GitHubClient, owner: str, repo: str, pr_number: int
) -> list[dict[str, Any]]:
    """
    Fetch existing review comments on the PR (for dedup).

    GET /repos/{owner}/{repo}/pulls/{pr_number}/comments — only returns
    comments made by our bot (filtered by the HTML marker).
    """
    comments = await client.paginate(f"/repos/{owner}/{repo}/pulls/{pr_number}/comments")
    return [c for c in comments if _BOT_MARKER in (c.get("body") or "")]


async def get_pr_head_sha(client: GitHubClient, owner: str, repo: str, pr_number: int) -> str:
    pr: dict[str, Any] = await client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
    sha: str = pr["head"]["sha"]
    return sha
