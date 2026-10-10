"""
Webhook signature verification and event dispatch.

GitHub signs webhook bodies with HMAC-SHA256 using the webhook secret:
    X-Hub-Signature-256: sha256=<hex digest of body>
We recompute and compare with hmac.compare_digest (constant time).
"""

from __future__ import annotations

import hmac as hmac_lib
from dataclasses import dataclass
from typing import Any

from prism.logging import get_logger

log = get_logger(__name__)

SIGNATURE_HEADER = "X-Hub-Signature-256"
EVENT_HEADER = "X-GitHub-Event"

# (event, action) pairs we act on.
_WATCHED = {("pull_request", "opened"), ("pull_request", "synchronize")}


def verify_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    """
    Verify the webhook HMAC-SHA256 signature.

    Returns False for missing/empty secrets, missing headers, malformed
    headers, or digest mismatch. Never raises on attacker-controlled input.
    """
    if not secret or not signature_header:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac_lib.new(secret.encode("utf-8"), body, "sha256").hexdigest()
    actual = signature_header[len("sha256=") :]
    return hmac_lib.compare_digest(expected, actual)


@dataclass(frozen=True)
class ReviewJob:
    """A PR that needs reviewing, extracted from a webhook event."""

    installation_id: int
    owner: str
    repo: str
    pr_number: int


def parse_event(headers: dict[str, str], payload: dict[str, Any]) -> ReviewJob | None:
    """
    Dispatch on X-GitHub-Event + action.

    Returns a ReviewJob for pull_request opened/synchronize, else None
    (ignored cleanly — pings, closed PRs, reviews, etc.).
    """
    # Normalize header lookup (case-insensitive).
    lowered = {k.lower(): v for k, v in headers.items()}
    event = lowered.get(EVENT_HEADER.lower())
    if not event:
        log.warning("webhook_missing_event_header")
        return None

    action = payload.get("action")
    if (event, action) not in _WATCHED:
        # NB: `event` is structlog's reserved first arg name — use gh_event.
        log.info("webhook_event_ignored", gh_event=event, action=action)
        return None

    try:
        installation_id = int(payload["installation"]["id"])
        owner = payload["repository"]["owner"]["login"]
        repo = payload["repository"]["name"]
        pr_number = int(payload["pull_request"]["number"])
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("webhook_malformed_payload", error=str(exc))
        return None

    return ReviewJob(installation_id=installation_id, owner=owner, repo=repo, pr_number=pr_number)
