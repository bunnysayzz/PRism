"""
FastAPI app: GitHub webhook receiver.

- POST /webhooks/github — rate-limited (in-memory token bucket, per
  installation), HMAC signature verified, event parsed, then enqueued as
  `review_pr` on the arq queue. When REDIS_URL is unset (dev mode), the job
  runs inline as a background task instead of going through Redis.
- GET /healthz — liveness check (always 200 when the process is up).
- GET /readyz — readiness check: settings validity, backend probe, Redis ping.
  Returns 503 with per-check details when not ready.

Every response carries an ``X-Request-ID`` header (client-supplied value is
echoed, otherwise a generated one); it is also bound into the structlog
context so all logs for the request carry it.

NOTE: the webhook rate limiter is single-process / in-memory. For multi-replica
deployments, move it to Redis (future work).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import structlog
from fastapi import BackgroundTasks, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse

from prism.config import Settings, get_settings
from prism.github.webhooks import (
    EVENT_HEADER,
    SIGNATURE_HEADER,
    parse_event,
    verify_signature,
)
from prism.logging import get_logger, setup_logging
from prism.worker import review_pr

log = get_logger(__name__)

_settings = get_settings()
setup_logging(_settings.log_level, log_format=_settings.log_format)


# ---------------------------------------------------------------------------
# Rate limiting: in-memory token bucket, per installation.
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Token bucket: ``capacity`` tokens max, refilled at ``refill_per_sec``."""

    def __init__(self, capacity: int, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.refill_per_sec = refill_per_sec
        self._tokens = float(capacity)
        self._last = time.monotonic()

    def consume(self) -> tuple[bool, float]:
        """Try to take one token.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after_seconds`` is
        0.0 when allowed, otherwise how long until one token is available.
        """
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True, 0.0
        return False, (1.0 - self._tokens) / self.refill_per_sec


# Single-process state. Key: "installation:<id>" or "ip:<client>" fallback.
_buckets: dict[str, _TokenBucket] = {}
_MAX_BUCKETS = 10_000


def _check_rate_limit(key: str, settings: Settings) -> float:
    """Return 0.0 if the request is allowed, else seconds until it may retry."""
    per_min = settings.rate_limit_webhook_per_min
    bucket = _buckets.get(key)
    if bucket is None or bucket.capacity != float(per_min):
        bucket = _TokenBucket(capacity=per_min, refill_per_sec=per_min / 60.0)
        _buckets[key] = bucket
        if len(_buckets) > _MAX_BUCKETS:
            # Prune the stalest bucket (least recently refilled).
            oldest = min(_buckets, key=lambda k: _buckets[k]._last)
            del _buckets[oldest]
    allowed, retry_after = bucket.consume()
    return 0.0 if allowed else retry_after


def _rate_limit_key(payload: dict[str, Any], request: Request) -> str:
    """Per-installation key; falls back to client IP for unparsable bodies."""
    try:
        installation_id = payload.get("installation", {}).get("id")
        if installation_id:
            return f"installation:{installation_id}"
    except AttributeError:
        pass
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


# ---------------------------------------------------------------------------
# Readiness probes
# ---------------------------------------------------------------------------


def _check_settings(settings: Settings) -> dict[str, Any]:
    """Settings are valid only if the app can actually authenticate + verify."""
    problems: list[str] = []
    if settings.github_app_id <= 0:
        problems.append("GITHUB_APP_ID is not set")
    if not settings.github_webhook_secret:
        problems.append("GITHUB_WEBHOOK_SECRET is not set")
    if not settings.github_private_key_path:
        problems.append("GITHUB_PRIVATE_KEY_PATH is not set")
    elif not Path(settings.github_private_key_path).exists():
        problems.append("GITHUB_PRIVATE_KEY_PATH does not exist")
    return {"ok": not problems, "problems": problems}


async def _check_backend(settings: Settings) -> dict[str, Any]:
    """Lightweight probe of the review backend. Never leaks credentials."""
    backend = settings.review_backend
    if backend == "stub":
        return {
            "ok": True,
            "backend": backend,
            "probed": False,
            "detail": "stub backend; probe skipped",
        }
    if backend == "vllm":
        url = settings.vllm_base_url.rstrip("/") + "/health"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(url)
            ok = resp.status_code == 200
            return {
                "ok": ok,
                "backend": backend,
                "probed": True,
                "detail": f"GET {url} -> {resp.status_code}",
            }
        except Exception as exc:  # network errors only; never includes secrets
            return {
                "ok": False,
                "backend": backend,
                "probed": True,
                "detail": f"probe failed: {type(exc).__name__}",
            }
    return {
        "ok": True,
        "backend": backend,
        "probed": False,
        "detail": "no probe implemented for this backend",
    }


async def _check_redis(redis_url: str) -> dict[str, Any]:
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(redis_url, socket_connect_timeout=2.0, socket_timeout=2.0)
        try:
            await client.ping()
        finally:
            # aclose exists at runtime (redis-py >= 5.0.1); missing from its type stubs
            await client.aclose()  # type: ignore[attr-defined]
        return {"ok": True, "probed": True, "detail": "PING ok"}
    except Exception as exc:
        return {"ok": False, "probed": True, "detail": f"ping failed: {type(exc).__name__}"}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="PRism", version="0.1.0")

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: 200 as long as the process is up."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Readiness: 200 only when settings, backend, and Redis are all OK."""
        settings = get_settings()
        checks: dict[str, Any] = {
            "settings": _check_settings(settings),
            "backend": await _check_backend(settings),
        }
        if settings.redis_url:
            checks["redis"] = await _check_redis(settings.redis_url)
        ready = all(bool(check.get("ok", False)) for check in checks.values())
        if not ready:
            log.warning("not_ready", checks=checks)
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "checks": checks},
            status_code=200 if ready else 503,
        )

    @app.post("/webhooks/github")
    async def github_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        response: Response,
        x_hub_signature_256: str | None = Header(default=None),
        x_github_event: str | None = Header(default=None),
    ) -> JSONResponse:
        settings = get_settings()
        body = await request.body()
        client_host = request.client.host if request.client else "unknown"

        # Rate limit BEFORE signature verification (cheap DoS shield). The key
        # is per-installation so one noisy repo can't starve the others.
        try:
            preview: dict[str, Any] = json.loads(body) if body else {}
        except json.JSONDecodeError:
            preview = {}
        retry_after = _check_rate_limit(_rate_limit_key(preview, request), settings)
        if retry_after > 0:
            log.warning("webhook_rate_limited", client=client_host)
            return JSONResponse(
                {"error": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        # Signature failure logs context only — never the secret or the digest.
        if not verify_signature(settings.github_webhook_secret, body, x_hub_signature_256):
            log.warning(
                "webhook_bad_signature",
                gh_event=x_github_event or "unknown",
                client=client_host,
            )
            return JSONResponse({"error": "invalid signature"}, status_code=401)

        try:
            payload: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        job = parse_event(
            {SIGNATURE_HEADER: x_hub_signature_256 or "", EVENT_HEADER: x_github_event or ""},
            payload,
        )
        if job is None:
            return JSONResponse({"status": "ignored"})

        log.info(
            "webhook_review_enqueued",
            repo=f"{job.owner}/{job.repo}",
            pr=job.pr_number,
            installation_id=job.installation_id,
        )
        if settings.redis_url:
            await _enqueue_redis(settings.redis_url, job)
        else:
            # Dev mode: no Redis — run inline in the background.
            log.info("dev_inline_mode")
            background_tasks.add_task(
                review_pr, {}, job.installation_id, job.owner, job.repo, job.pr_number
            )
        return JSONResponse({"status": "enqueued"})

    return app


async def _enqueue_redis(redis_url: str, job: Any) -> None:
    from arq import create_pool
    from arq.connections import RedisSettings

    pool = await create_pool(RedisSettings.from_dsn(redis_url))
    try:
        await pool.enqueue_job("review_pr", job.installation_id, job.owner, job.repo, job.pr_number)
    finally:
        # aclose exists at runtime (arq >= 0.26); missing from arq's type stubs
        await pool.aclose()  # type: ignore[attr-defined]


app = create_app()


def run() -> None:
    """Entrypoint for `PRism` console script."""
    import uvicorn

    uvicorn.run("prism.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
