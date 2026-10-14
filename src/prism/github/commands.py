"""
`@prism fix` command handler.

When a user comments `@prism fix` on a PR, we collect the bot's own review
comments that contain drop-in ```suggestion blocks and apply them as a single
commit on the PR branch via the git database API (blobs -> tree -> commit ->
ref update).

Permission note: pushing commits needs the GitHub App to have
**Contents: Read & write**. The app currently ships with read-only contents
access, so the mutating sequence is wrapped to degrade gracefully: on
403/404 we post an explanatory comment telling the requester how to grant
the permission instead of crashing.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

from prism.github.client import GitHubAPIError, GitHubClient
from prism.github.reviews import _is_drop_in_suggestion, fetch_existing_review_comments
from prism.github.webhooks import FixCommand
from prism.logging import get_logger

log = get_logger(__name__)

#: Cap on how many suggested files one `@prism fix` applies. Keeps the
#: single commit reviewable and the webhook inside the serverless timeout.
MAX_FIX_FILES = 5

#: GitHub statuses that mean "the app is not allowed to do this" (as opposed
#: to a transient failure a retry could fix).
_PERMISSION_STATUSES = frozenset({403, 404})

_DEGRADED_MESSAGE = (
    "@{requester} I couldn't apply fixes - the PRism GitHub App needs the "
    "**Contents: Read & write** permission. An admin can grant it under the "
    "app's Permissions settings, then re-run `@prism fix`."
)

_SUGGESTION_FENCE_RE = re.compile(r"```suggestion\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class FixTarget:
    """One applicable suggestion: file, 1-indexed line, replacement lines."""

    path: str
    line: int
    suggestion: str


def extract_suggestion(body: str) -> str | None:
    """Pull the suggestion text out of a ```suggestion fence, if present."""
    if not body:
        return None
    match = _SUGGESTION_FENCE_RE.search(body)
    if not match:
        return None
    return match.group(1).strip("\n")


def extract_applicable_suggestions(
    comments: list[dict[str, Any]], *, limit: int = MAX_FIX_FILES
) -> list[FixTarget]:
    """Filter review comments down to applicable drop-in suggestions.

    Keeps only comments that carry a path + line, whose fenced suggestion
    passes the drop-in heuristic. Dedupes on (path, line) and caps at
    ``limit`` files.
    """
    targets: list[FixTarget] = []
    seen: set[tuple[str, int]] = set()
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        path = comment.get("path")
        suggestion = extract_suggestion(str(comment.get("body") or ""))
        try:
            line = int(comment["line"]) if comment.get("line") is not None else 0
        except (TypeError, ValueError):
            continue
        if not path or not isinstance(path, str) or line < 1 or not suggestion:
            continue
        if not _is_drop_in_suggestion(suggestion):
            continue
        key = (path, line)
        if key in seen:
            continue
        seen.add(key)
        targets.append(FixTarget(path=path, line=line, suggestion=suggestion))
        if len(targets) >= limit:
            break
    return targets


async def handle_fix_command(client: GitHubClient, cmd: FixCommand) -> dict[str, Any]:
    """Apply the bot's drop-in suggestions to the PR branch. Never raises.

    Returns one of:
      {"status": "ok", "applied": [paths]}
      {"status": "noop", "reason": ...}
      {"status": "degraded", "reason": "missing contents:write"}
      {"status": "error", "reason": ...}
    """
    try:
        return await _handle_fix_command(client, cmd)
    except Exception as exc:  # noqa: BLE001 - command must report, never crash
        log.error(
            "fix_command_failed",
            repo=f"{cmd.owner}/{cmd.repo}",
            pr=cmd.pr_number,
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}


async def _handle_fix_command(client: GitHubClient, cmd: FixCommand) -> dict[str, Any]:
    owner, repo, pr_number = cmd.owner, cmd.repo, cmd.pr_number
    try:
        pr = await client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
        head_sha = str(pr["head"]["sha"])
        branch = str(pr["head"]["ref"])
        comments = await fetch_existing_review_comments(client, owner, repo, pr_number)
        targets = extract_applicable_suggestions(comments)
        if not targets:
            log.info("fix_command_noop", repo=f"{owner}/{repo}", pr=pr_number)
            return {"status": "noop", "reason": "no applicable drop-in suggestions"}

        applied = await _apply_targets(client, owner, repo, targets, head_sha, branch)
        if not applied:
            return {"status": "noop", "reason": "no suggestions could be applied"}
    except GitHubAPIError as exc:
        if exc.status in _PERMISSION_STATUSES:
            await _post_degraded_comment(client, cmd)
            return {"status": "degraded", "reason": "missing contents:write"}
        raise

    await _post_summary_comment(client, cmd, applied)
    log.info(
        "fix_command_applied",
        repo=f"{owner}/{repo}",
        pr=pr_number,
        files=len(applied),
    )
    return {"status": "ok", "applied": applied}


async def _apply_targets(
    client: GitHubClient,
    owner: str,
    repo: str,
    targets: list[FixTarget],
    head_sha: str,
    branch: str,
) -> list[str]:
    """Apply each target file best-effort; commit and push the successes."""
    entries: list[dict[str, Any]] = []
    applied: list[str] = []
    for target in targets:
        try:
            blob_sha = await _target_to_blob(client, owner, repo, target, head_sha)
        except GitHubAPIError as exc:
            if exc.status in _PERMISSION_STATUSES:
                raise  # permission problem -> whole command degrades
            log.warning(
                "fix_command_file_skipped",
                path=target.path,
                error_type=type(exc).__name__,
                status=exc.status,
            )
            continue
        except Exception as exc:  # noqa: BLE001 - best-effort per file
            log.warning(
                "fix_command_file_skipped",
                path=target.path,
                error_type=type(exc).__name__,
            )
            continue
        entries.append({"path": target.path, "mode": "100644", "type": "blob", "sha": blob_sha})
        applied.append(target.path)

    if not entries:
        return []

    # Single commit for all applied files: base tree -> new tree -> commit.
    base_commit = await client.get(f"/repos/{owner}/{repo}/git/commits/{head_sha}")
    base_tree_sha = str(base_commit["tree"]["sha"])
    new_tree = await client.post(
        f"/repos/{owner}/{repo}/git/trees",
        json={"base_tree": base_tree_sha, "tree": entries},
    )
    new_commit = await client.post(
        f"/repos/{owner}/{repo}/git/commits",
        json={
            "message": "@prism fix: apply review suggestions",
            "tree": str(new_tree["sha"]),
            "parents": [head_sha],
        },
    )
    new_sha = str(new_commit["sha"])
    await client.request(
        "PATCH",
        f"/repos/{owner}/{repo}/git/refs/heads/{branch}",
        json={"sha": new_sha},
    )
    return applied


async def _target_to_blob(
    client: GitHubClient,
    owner: str,
    repo: str,
    target: FixTarget,
    head_sha: str,
) -> str:
    """Fetch a file at head_sha, replace the flagged line, store as a blob."""
    data = await client.get(
        f"/repos/{owner}/{repo}/contents/{target.path}",
        params={"ref": head_sha},
    )
    raw = base64.b64decode(str(data["content"]))
    lines = raw.decode("utf-8").split("\n")
    if target.line > len(lines):
        raise ValueError(f"line {target.line} out of range ({len(lines)} lines)")
    lines[target.line - 1 : target.line] = target.suggestion.splitlines()
    new_content = base64.b64encode("\n".join(lines).encode("utf-8")).decode("ascii")
    blob = await client.post(
        f"/repos/{owner}/{repo}/git/blobs",
        json={"content": new_content, "encoding": "base64"},
    )
    return str(blob["sha"])


async def _post_degraded_comment(client: GitHubClient, cmd: FixCommand) -> None:
    body = _DEGRADED_MESSAGE.format(requester=cmd.requester)
    await client.post(
        f"/repos/{cmd.owner}/{cmd.repo}/issues/{cmd.pr_number}/comments",
        json={"body": body},
    )
    log.info("fix_command_degraded", repo=f"{cmd.owner}/{cmd.repo}", pr=cmd.pr_number)


async def _post_summary_comment(client: GitHubClient, cmd: FixCommand, applied: list[str]) -> None:
    lines = ["`@prism fix` applied suggestions to:"]
    lines.extend(f"- `{path}`" for path in applied)
    await client.post(
        f"/repos/{cmd.owner}/{cmd.repo}/issues/{cmd.pr_number}/comments",
        json={"body": "\n".join(lines)},
    )
