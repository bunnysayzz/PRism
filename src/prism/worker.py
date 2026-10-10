"""
arq worker: executes `review_pr` jobs off the queue.

Job: review_pr(ctx, installation_id, owner, repo, pr_number)
  1. Mint an installation token for the installation.
  2. Run the review engine.
  3. Post the review (summary + inline comments) to the PR.
"""

from __future__ import annotations

from typing import Any

from prism.config import get_settings, is_repo_allowed, load_repo_config
from prism.github import reviews as gh_reviews
from prism.github.auth import get_installation_token
from prism.github.client import GitHubClient
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
        result = await review_pull_request(settings, cfg, client, owner, repo, pr_number)
        head_sha = await gh_reviews.get_pr_head_sha(client, owner, repo, pr_number)
        summary = gh_reviews.render_summary(
            owner, repo, pr_number, result.findings, result.skipped_files, result.model
        )
        # Always post the summary so the PR shows the bot ran; inline comments
        # only when there are findings (empty comments list is valid).
        await gh_reviews.post_review(
            client, owner, repo, pr_number, head_sha, summary, result.findings
        )

    log.info(
        "review_job_done",
        repo=f"{owner}/{repo}",
        pr=pr_number,
        findings=len(result.findings),
    )
    return {"status": "ok", "findings": len(result.findings)}


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
