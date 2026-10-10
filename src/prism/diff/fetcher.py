"""
PR file fetching and filtering.

- Lists changed files via GET /repos/{o}/{r}/pulls/{n}/files (paginated).
- Filters with ignore globs, binary detection, and size caps.
- Fetches raw file content (base + head) so the review prompt gets
  surrounding code beyond the diff hunks.
"""

from __future__ import annotations

import base64
import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prism.logging import get_logger

if TYPE_CHECKING:
    from prism.github.client import GitHubClient

log = get_logger(__name__)


@dataclass
class ChangedFile:
    filename: str
    status: str  # added | removed | modified | renamed
    additions: int
    deletions: int
    patch: str | None
    previous_filename: str | None = None
    head_excerpt: str = ""  # surrounding code from the new version
    base_excerpt: str = ""  # surrounding code from the old version


def _matches_glob(path: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path.lstrip("/"), pat.lstrip("/")):
            return True
    return False


def _looks_binary(patch: str | None, filename: str) -> bool:
    if patch and "Binary files" in patch:
        return True
    binary_exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp4",
        ".mov",
        ".pyc",
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".dat",
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
    }
    return any(filename.lower().endswith(ext) for ext in binary_exts)


def should_review_file(
    filename: str,
    status: str,
    patch: str | None,
    *,
    ignore_globs: list[str],
    max_file_bytes: int,
    patch_bytes: int = 0,
) -> tuple[bool, str]:
    """
    Decide whether a file should be reviewed.

    Returns (True, "") or (False, reason). Reasons feed the "skipped files"
    section of the review summary.
    """
    if status == "removed":
        return False, "file removed"
    if _looks_binary(patch, filename):
        return False, "binary file"
    if _matches_glob(filename, ignore_globs):
        return False, "matches ignore glob"
    if not patch:
        return False, "no textual diff"
    if patch_bytes and patch_bytes > max_file_bytes:
        return False, f"patch exceeds {max_file_bytes} bytes"
    return True, ""


async def list_pr_files(
    client: GitHubClient, owner: str, repo: str, pr_number: int
) -> list[dict[str, Any]]:
    """GET /repos/{o}/{r}/pulls/{n}/files with pagination."""
    return await client.paginate(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")


async def _fetch_blob(client: GitHubClient, owner: str, repo: str, path: str, ref: str) -> str:
    """
    Fetch a file's content at a ref via GET /repos/{o}/{r}/contents/{path}?ref=...
    Returns decoded text; empty string on failure or non-UTF8 content.
    """
    try:
        data = await client.get(f"/repos/{owner}/{repo}/contents/{path}", params={"ref": ref})
        if isinstance(data, dict) and data.get("encoding") == "base64":
            raw = base64.b64decode(data["content"])
            return raw.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 — context is best-effort
        log.warning("blob_fetch_failed", path=path, ref=ref[:12], error=str(exc)[:200])
    return ""


def _excerpt_around(text: str, line: int, radius: int = 40) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    idx = max(0, min(line - 1, len(lines) - 1))
    lo, hi = max(0, idx - radius), idx + radius + 1
    numbered = [f"{i + 1:>5} | {text}" for i, text in enumerate(lines[lo:hi], start=lo + 1)]
    return "\n".join(numbered)


async def enrich_with_context(
    client: GitHubClient,
    owner: str,
    repo: str,
    base_sha: str,
    head_sha: str,
    changed: ChangedFile,
    focus_line: int | None = None,
) -> ChangedFile:
    """
    Attach base/head file excerpts around the first changed line, so the
    review prompt sees surrounding code beyond the diff hunk.
    """
    path = changed.filename
    head_text = await _fetch_blob(client, owner, repo, path, head_sha)
    base_text = ""
    if changed.status != "added":
        base_ref_path = changed.previous_filename or path
        base_text = await _fetch_blob(client, owner, repo, base_ref_path, base_sha)
    anchor = focus_line or 1
    changed.head_excerpt = _excerpt_around(head_text, anchor)
    changed.base_excerpt = _excerpt_around(base_text, anchor)
    return changed


def to_changed_file(entry: dict[str, Any]) -> ChangedFile:
    return ChangedFile(
        filename=entry["filename"],
        status=entry.get("status", "modified"),
        additions=int(entry.get("additions", 0)),
        deletions=int(entry.get("deletions", 0)),
        patch=entry.get("patch"),
        previous_filename=entry.get("previous_filename"),
    )
