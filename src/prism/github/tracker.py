"""Issue-tracker ticket context (Linear / Jira) for PR reviews.

Security contract: the tracker ``api_token`` is read from ``cfg["tracker"]``
and is NEVER logged, NEVER included in exceptions, and NEVER echoed into
returned strings. Only the tracker provider name and the ticket ref are
logged.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from prism.logging import get_logger

log = get_logger(__name__)

#: Jira/Linear-style ticket keys: 2-10 uppercase letters, dash, digits.
TICKET_RE = re.compile(r"\b([A-Z]{2,10}-\d+)\b")

#: Combined scan for ticket keys and GitHub issue refs, in text order.
_REF_RE = re.compile(r"\b([A-Z]{2,10}-\d+)\b|#(\d+)")

_DESCRIPTION_MAX_CHARS = 1500


def parse_ticket_refs(text: str) -> list[str]:
    """Extract ticket refs from text, deduped, in order of appearance.

    Matches Jira/Linear-style keys (``PROJ-123``) and GitHub issue refs
    (``#45``, returned as ``"#45"`` strings). Lowercase keys are ignored.
    """
    refs: list[str] = []
    seen: set[str] = set()
    for match in _REF_RE.finditer(text or ""):
        ref = match.group(1) or f"#{match.group(2)}"
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


async def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON over a fresh client and return the decoded body.

    Test seam: monkeypatch this function to stub the ticket API.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body


async def _get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    """GET JSON over a fresh client and return the decoded body.

    Test seam: monkeypatch this function to stub the ticket API.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body


def _format_ticket(ref: str, title: str, description: str) -> str:
    """Render one ticket as review context. Never includes any token."""
    description = description[:_DESCRIPTION_MAX_CHARS]
    return f"Ticket {ref}: {title}\nAcceptance criteria / description:\n{description}"


def _jira_description_text(value: Any) -> str:
    """Best-effort plain text from a Jira ADF (rich text) description."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    texts: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            text = node.get("text")
            if isinstance(text, str):
                texts.append(text)
            content = node.get("content")
            if isinstance(content, list):
                for child in content:
                    _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(value)
    return "\n".join(texts)


async def _fetch_linear(token: str, ref: str) -> str | None:
    """Fetch a Linear issue by identifier."""
    data = await _post_json(
        "https://api.linear.app/graphql",
        {"Authorization": token, "Content-Type": "application/json"},
        {
            "query": "query($id:String!){issue(id:$id){identifier title description}}",
            "variables": {"id": ref},
        },
    )
    issue = (data.get("data") or {}).get("issue") or {}
    if not issue:
        return None
    return _format_ticket(
        str(issue.get("identifier") or ref),
        str(issue.get("title") or ""),
        str(issue.get("description") or ""),
    )


async def _fetch_jira(token: str, ref: str, base_url: str) -> str | None:
    """Fetch a Jira issue by key."""
    base = base_url.rstrip("/")
    if not base:
        return None
    data = await _get_json(f"{base}/rest/api/3/issue/{ref}", {"Authorization": f"Bearer {token}"})
    fields = data.get("fields") or {}
    key = str(data.get("key") or ref)
    title = str(fields.get("summary") or "")
    if not data.get("key") and not title:
        return None
    return _format_ticket(key, title, _jira_description_text(fields.get("description")))


async def fetch_ticket_context(
    client: Any, owner: str, repo: str, pr_number: int, cfg: dict[str, Any]
) -> str | None:
    """Fetch ticket context for a PR's referenced tickets (Linear/Jira).

    Scans the PR branch name and body for ticket refs, then pulls the first
    tracker-style ref from the configured provider. Never raises; returns
    ``None`` when there are no refs, no usable tracker config, or any error.
    Never logs the API token or ticket contents (only provider + ref).
    """
    try:
        pr = await client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
        head = pr.get("head") or {}
        refs = parse_ticket_refs(f"{head.get('ref') or ''} {pr.get('body') or ''}")
        if not refs:
            return None
        tracker = cfg.get("tracker") or {}
        if not isinstance(tracker, dict):
            return None
        provider = str(tracker.get("provider") or "").lower()
        token = str(tracker.get("api_token") or "")
        if not token or provider not in ("linear", "jira"):
            return None
        ref = next((r for r in refs if not r.startswith("#")), None)
        if ref is None:
            return None
        log.info("ticket_fetch", provider=provider, ref=ref)
        if provider == "linear":
            return await _fetch_linear(token, ref)
        return await _fetch_jira(token, ref, str(tracker.get("base_url") or ""))
    except Exception as exc:  # noqa: BLE001 - contract: never raises
        # Only the exception type is logged; the token and ticket contents
        # never appear in logs or exceptions.
        log.info("ticket_fetch_failed", error=type(exc).__name__)
        return None
