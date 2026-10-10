"""
Async GitHub REST collector for PR review training samples.

Collects, from PUBLIC repositories:
  - PR metadata (number, title),
  - per-file unified diffs (``patch``),
  - inline review comments anchored to a diff line (path + line).

Output: ``data/raw/raw.jsonl`` — one :class:`RawSample` JSON object per
(PR, file, comment) triple. PRs with no inline review comments are skipped
(the PR list endpoint exposes a ``review_comments`` count, so this costs no
extra API calls).

Auth: ``GH_TOKEN`` env var — a classic PAT with the ``public_repo`` scope, or
a fine-grained token with read access to public repositories. The token is
never logged or written to disk.

Rate limits: every response's ``X-RateLimit-Remaining`` /
``X-RateLimit-Reset`` headers are honored — the client sleeps proactively when
the budget runs low, and backs off until reset on 403/429 (respecting
``Retry-After``).

Idempotency: a checkpoint file records every processed ``(repo, pr)`` pair,
so re-runs skip already-collected PRs and resume where they left off.
``load_raw_samples()`` additionally dedupes by ``sample_id`` on read, so a
crash between the JSONL append and the checkpoint write cannot corrupt the
dataset.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from prism.data.models import RawSample
from prism.logging import get_logger

log = get_logger(__name__)

API_BASE = "https://api.github.com"
_ACCEPT = "application/vnd.github+json"
_API_VERSION = "2022-11-28"
_PER_PAGE = 100

# Sleep proactively when fewer than this many core requests remain.
_RATE_LIMIT_LOW_WATERMARK = 100
_MAX_ATTEMPTS = 5


class CollectorError(RuntimeError):
    """Fatal collector error (bad config, auth failure, unrecoverable API error)."""


@dataclass
class CollectorConfig:
    repos: list[str] = field(default_factory=list)
    max_prs_per_repo: int = 200
    out_dir: Path = Path("data/raw")
    checkpoint_path: Path = Path("data/raw/checkpoint.json")
    concurrency: int = 3
    min_review_comments: int = 1


@dataclass
class CollectionStats:
    repos_done: int = 0
    prs_seen: int = 0
    prs_collected: int = 0
    samples_written: int = 0
    prs_skipped_no_comments: int = 0
    prs_skipped_checkpoint: int = 0


class Checkpoint:
    """Tracks processed (repo, pr) pairs so collection is idempotent."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.processed: set[str] = set()
        self._load()

    @staticmethod
    def pair(repo: str, pr_number: int) -> str:
        return f"{repo}#{pr_number}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("checkpoint_load_failed", path=str(self.path), error=str(exc)[:200])
            return
        raw = data.get("processed", [])
        self.processed = {str(p) for p in raw}
        log.info("checkpoint_loaded", processed=len(self.processed), path=str(self.path))

    def mark(self, repo: str, pr_number: int) -> None:
        self.processed.add(self.pair(repo, pr_number))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        payload = {"version": 1, "processed": sorted(self.processed)}
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def _split_repo(repo: str) -> tuple[str, str]:
    parts = repo.split("/")
    if len(parts) != 2 or not all(p.strip() for p in parts):
        raise CollectorError(f"invalid repo {repo!r}: expected 'owner/name'")
    return parts[0].strip(), parts[1].strip()


def _rate_limit_exhausted(resp: httpx.Response) -> bool:
    return resp.status_code == 403 and resp.headers.get("x-ratelimit-remaining") == "0"


async def _sleep_until_reset(headers: httpx.Headers) -> None:
    """Sleep until the rate-limit window resets (honors Retry-After)."""
    retry_after = headers.get("retry-after")
    wait_s: float | None = None
    if retry_after and retry_after.isdigit():
        wait_s = float(retry_after)
    else:
        reset = headers.get("x-ratelimit-reset")
        if reset and reset.isdigit():
            wait_s = max(1.0, float(reset) - time.time() + 2.0)
    if wait_s is None:
        wait_s = 60.0
    wait_s = min(wait_s, 3600.0)
    log.warning("github_rate_limit_sleep", sleeping_s=round(wait_s, 1))
    await asyncio.sleep(wait_s)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_raw_samples(path: str | Path) -> list[RawSample]:
    """
    Read raw.jsonl, skipping malformed lines; dedupe by sample_id (keep first).

    The dedupe guards against duplicates from a crash between the JSONL
    append and the checkpoint write during collection.
    """
    p = Path(path)
    if not p.exists():
        raise CollectorError(f"raw input not found: {p}")
    samples: list[RawSample] = []
    seen: set[str] = set()
    dupes = 0
    skipped = 0
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            sample = RawSample.model_validate(json.loads(line))
        except Exception as exc:  # noqa: BLE001 — malformed line: skip, don't fail
            skipped += 1
            log.warning("raw_line_skipped", line=lineno, error=str(exc)[:200])
            continue
        if sample.sample_id in seen:
            dupes += 1
            continue
        seen.add(sample.sample_id)
        samples.append(sample)
    log.info(
        "raw_loaded",
        path=str(p),
        samples=len(samples),
        duplicates_skipped=dupes,
        lines_skipped=skipped,
    )
    return samples


class GitHubCollector:
    """
    Async GitHub REST client for PR review training data.

    Pass a custom ``transport`` (e.g. ``httpx.MockTransport``) for tests.
    Use :meth:`from_env` in production to read ``GH_TOKEN``.
    """

    def __init__(self, token: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not token or not token.strip():
            raise CollectorError("GitHub token is empty.")
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            transport=transport,
            timeout=httpx.Timeout(30.0),
            headers={
                "Accept": _ACCEPT,
                "X-GitHub-Api-Version": _API_VERSION,
                "User-Agent": "PRism-data-collector/0.1.0",
                "Authorization": f"Bearer {token.strip()}",
            },
        )

    @classmethod
    def from_env(cls, *, transport: httpx.AsyncBaseTransport | None = None) -> GitHubCollector:
        token = os.environ.get("GH_TOKEN", "").strip()
        if not token:
            raise CollectorError(
                "GH_TOKEN env var is not set. Create a classic PAT with the "
                "'public_repo' scope (or a fine-grained token with read access "
                "to public repositories) and export GH_TOKEN=<token>."
            )
        return cls(token, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GitHubCollector:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- core request machinery -------------------------------------------

    async def _request(
        self, method: str, path: str, *, params: dict[str, Any] | None = None
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = await self._client.request(method, path, params=params)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                await asyncio.sleep(min(2.0**attempt, 30.0))
                continue

            await self._maybe_throttle(resp.headers)

            if _rate_limit_exhausted(resp) or resp.status_code == 429:
                await _sleep_until_reset(resp.headers)
                continue
            if resp.status_code in {500, 502, 503, 504}:
                last_error = CollectorError(f"{method} {path} -> {resp.status_code} (retryable)")
                await asyncio.sleep(min(2.0**attempt, 30.0))
                continue
            if resp.status_code == 401:
                raise CollectorError("GitHub auth failed (401): check GH_TOKEN value and expiry.")
            if resp.status_code == 404:
                raise CollectorError(
                    f"GitHub {method} {path} -> 404: repo/PR not found or not public."
                )
            if resp.status_code >= 400:
                raise CollectorError(
                    f"GitHub {method} {path} -> {resp.status_code}: {resp.text[:300]}"
                )
            return resp.json() if resp.content else None

        raise CollectorError(f"{method} {path} failed after {_MAX_ATTEMPTS} attempts: {last_error}")

    async def _maybe_throttle(self, headers: httpx.Headers) -> None:
        """Proactively sleep when the remaining rate-limit budget runs low."""
        remaining = headers.get("x-ratelimit-remaining")
        low = (
            remaining is not None
            and remaining.isdigit()
            and int(remaining) < _RATE_LIMIT_LOW_WATERMARK
        )
        if low:
            await _sleep_until_reset(headers)

    async def _paginate(self, path: str, *, params: dict[str, Any] | None = None) -> list[Any]:
        items: list[Any] = []
        page = 1
        while True:
            query = dict(params or {})
            query.update({"per_page": _PER_PAGE, "page": page})
            batch = await self._request("GET", path, params=query)
            if not isinstance(batch, list) or not batch:
                break
            items.extend(batch)
            if len(batch) < _PER_PAGE:
                break
            page += 1
        return items

    # -- collection ---------------------------------------------------------

    async def collect_pr_list(self, repo: str) -> list[dict[str, Any]]:
        """List closed PRs, most recently updated first."""
        owner, name = _split_repo(repo)
        items = await self._paginate(
            f"/repos/{owner}/{name}/pulls",
            params={"state": "closed", "sort": "updated", "direction": "desc"},
        )
        return [i for i in items if isinstance(i, dict)]

    async def collect_pr(self, repo: str, pr: dict[str, Any]) -> list[RawSample]:
        """
        Collect one PR's file patches + anchored inline review comments.

        Returns one RawSample per comment that anchors to a diff line of a
        file we have a patch for. Comments without a patch (oversized files)
        or without an anchor line are skipped here; the cleaner drops any
        stragglers.
        """
        owner, name = _split_repo(repo)
        pr_number = int(pr["number"])
        title = str(pr.get("title") or "")
        files, comments = await asyncio.gather(
            self._paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/files"),
            self._paginate(f"/repos/{owner}/{name}/pulls/{pr_number}/comments"),
        )
        patches: dict[str, str] = {}
        for entry in files:
            if not isinstance(entry, dict):
                continue
            filename = entry.get("filename")
            patch = entry.get("patch")
            if isinstance(filename, str) and isinstance(patch, str) and patch:
                patches[filename] = patch

        samples: list[RawSample] = []
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            comment_id = _optional_int(comment.get("id"))
            path = comment.get("path")
            line = _optional_int(comment.get("line"))
            if line is None:
                line = _optional_int(comment.get("original_line"))
            body = comment.get("body")
            if (
                comment_id is None
                or not isinstance(path, str)
                or not path
                or line is None
                or not isinstance(body, str)
                or not body.strip()
            ):
                continue
            patch = patches.get(path)
            if patch is None:
                continue
            user = comment.get("user") or {}
            user = user if isinstance(user, dict) else {}
            samples.append(
                RawSample(
                    sample_id=f"{repo}#{pr_number}#{path}#c{comment_id}",
                    repo=repo,
                    pr_number=pr_number,
                    pr_title=title,
                    path=path,
                    patch=patch,
                    comment_id=comment_id,
                    comment_author=str(user.get("login") or ""),
                    comment_author_type=str(user.get("type") or "User"),
                    comment_body=body,
                    comment_line=line,
                    comment_original_line=_optional_int(comment.get("original_line")),
                    comment_side=str(comment.get("side") or "RIGHT"),
                )
            )
        log.info("pr_collected", repo=repo, pr=pr_number, samples=len(samples))
        return samples

    async def run(self, config: CollectorConfig) -> CollectionStats:
        """
        Collect all configured repos into ``<out_dir>/raw.jsonl``.

        Idempotent: PRs already in the checkpoint are skipped; the checkpoint
        is saved after every PR so a crash loses at most one PR's progress.
        """
        if not config.repos:
            raise CollectorError("no repos configured")
        checkpoint = Checkpoint(config.checkpoint_path)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = config.out_dir / "raw.jsonl"
        stats = CollectionStats()
        sem = asyncio.Semaphore(max(1, config.concurrency))

        async with self:
            for repo in config.repos:
                _split_repo(repo)  # fail fast on bad input
                prs = await self.collect_pr_list(repo)
                stats.prs_seen += len(prs)
                with_comments = [
                    pr
                    for pr in prs
                    if (_optional_int(pr.get("review_comments")) or 0) >= config.min_review_comments
                ]
                stats.prs_skipped_no_comments += len(prs) - len(with_comments)
                todo = [
                    pr
                    for pr in with_comments[: config.max_prs_per_repo]
                    if Checkpoint.pair(repo, int(pr["number"])) not in checkpoint.processed
                ]
                capped = with_comments[: config.max_prs_per_repo]
                stats.prs_skipped_checkpoint += len(capped) - len(todo)

                async def _one(target_repo: str, pr: dict[str, Any]) -> list[RawSample]:
                    async with sem:
                        return await self.collect_pr(target_repo, pr)

                results = await asyncio.gather(*(_one(repo, pr) for pr in todo))
                with raw_path.open("a", encoding="utf-8") as fh:
                    for pr, samples in zip(todo, results, strict=True):
                        for sample in samples:
                            fh.write(json.dumps(sample.model_dump(mode="json")) + "\n")
                            stats.samples_written += 1
                        stats.prs_collected += 1
                        checkpoint.mark(repo, int(pr["number"]))
                        checkpoint.save()  # crash-safe: at most one PR is re-collected
                stats.repos_done += 1
                log.info("repo_done", repo=repo, prs=len(todo))

        log.info(
            "collection_done",
            repos=stats.repos_done,
            prs_collected=stats.prs_collected,
            samples=stats.samples_written,
            skipped_no_comments=stats.prs_skipped_no_comments,
            skipped_checkpoint=stats.prs_skipped_checkpoint,
        )
        return stats
