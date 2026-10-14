"""
arq worker: executes `review_pr` jobs off the queue.

Job: review_pr(ctx, installation_id, owner, repo, pr_number)
  1. Mint an installation token for the installation.
  2. Run the review engine.
  3. Post the review (summary + inline comments) to the PR.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from prism.config import (
    get_settings,
    is_repo_allowed,
    load_repo_config,
    merge_repo_config,
)
from prism.github import reviews as gh_reviews
from prism.github.auth import get_installation_token
from prism.github.client import GitHubClient
from prism.github.repoconfig import fetch_repo_file_config
from prism.logging import get_logger, setup_logging
from prism.review.engine import review_pull_request

log = get_logger(__name__)


async def review_pr(
    ctx: dict[str, Any],
    installation_id: int,
    owner: str,
    repo: str,
    pr_number: int,
) -> dict[str, Any]:
    settings = get_settings()
    setup_logging(settings.log_level)
    cfg = load_repo_config()

    if not is_repo_allowed(cfg, owner, repo):
        log.info("repo_not_allowlisted", repo=f"{owner}/{repo}", pr=pr_number)
        return {"status": "skipped", "reason": "repo not in allowlist"}

    token = await get_installation_token(settings, installation_id)
    async with GitHubClient(settings, installation_token=token) as client:
        repo_file_cfg = await fetch_repo_file_config(client, owner, repo)
        cfg = merge_repo_config(load_repo_config(), repo_file_cfg)

        # Ticket context is built by another workstream; tolerate its absence.
        ticket_context: Any = None
        try:
            tracker = importlib.import_module("prism.github.tracker")
            ticket_context = await tracker.fetch_ticket_context(client, owner, repo, pr_number, cfg)
        except Exception:
            ticket_context = None

        engine_kwargs: dict[str, Any] = {}
        if "ticket_context" in inspect.signature(review_pull_request).parameters:
            engine_kwargs["ticket_context"] = ticket_context
        result = await review_pull_request(
            settings, cfg, client, owner, repo, pr_number, **engine_kwargs
        )

        auto_post_bar = float(cfg.get("confidence_auto_post") or 0.9)
        actionable, nitpicks = gh_reviews.split_actionable(result.findings, auto_post_bar)
        score = gh_reviews.compute_merge_score(result.findings)
        verdict = gh_reviews.merge_verdict(score)
        if result.backend_errors and verdict == "Safe to merge":
            # Never bless an incomplete review as safe to merge.
            verdict = "Needs careful review"
        result.merge_score = score
        result.verdict = verdict
        num_files = len({f.path for f in result.findings})
        effort = gh_reviews.estimate_effort(num_files, result.findings)

        # Analytics is built by another workstream; never let it break the review.
        try:
            analytics = importlib.import_module("prism.review.analytics")
            recorded = analytics.record_review(
                owner,
                repo,
                pr_number,
                {
                    "findings": len(actionable),
                    "nitpicks": len(nitpicks),
                    "merge_score": score,
                    "verdict": verdict,
                    "backend_errors": len(result.backend_errors),
                },
            )
            if inspect.isawaitable(recorded):
                await recorded
        except Exception:
            pass

        summary = gh_reviews.render_summary(
            owner,
            repo,
            pr_number,
            actionable,
            result.skipped_files,
            result.model,
            result.backend_errors,
            pr_summary=result.pr_summary,
            nitpicks=nitpicks,
            verdict=verdict,
            effort=effort,
        )
        head_sha = await gh_reviews.get_pr_head_sha(client, owner, repo, pr_number)

        # Approve only a truly clean PR; otherwise COMMENT.
        clean = (
            not actionable
            and not nitpicks
            and not result.backend_errors
            and not result.skipped_files
        )
        event = "APPROVE" if clean else "COMMENT"
        # Always post the summary so the PR shows the bot ran; inline comments
        # only for actionable findings (empty comments list is valid).
        await gh_reviews.post_review(
            client, owner, repo, pr_number, head_sha, summary, actionable, event=event
        )
        await gh_reviews.post_pr_labels(client, owner, repo, pr_number, verdict)

    log.info(
        "review_job_done",
        repo=f"{owner}/{repo}",
        pr=pr_number,
        findings=len(result.findings),
        verdict=verdict,
        merge_score=score,
    )
    return {
        "status": "ok",
        "findings": len(result.findings),
        "verdict": verdict,
        "merge_score": score,
    }


class WorkerSettings:
    functions = [review_pr]
    redis_settings = None  # set from env in create_worker()

    @staticmethod
    async def on_startup(ctx: dict[str, Any]) -> None:
        setup_logging(get_settings().log_level)
        log.info("worker_startup")


def create_worker() -> Any:
    """Build an arq Worker with Redis configured from the environment."""
    from arq import Worker
    from arq.connections import RedisSettings

    settings = get_settings()
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL must be set to run the worker")
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    return Worker(
        functions=WorkerSettings.functions,
        redis_settings=redis_settings,
        on_startup=WorkerSettings.on_startup,
    )


if __name__ == "__main__":
    import asyncio

    worker = create_worker()
    asyncio.run(worker.async_run())
