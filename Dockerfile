# Multi-stage: build deps in one layer, ship a slim runtime.
#
# Base image is pinned by digest (multi-arch manifest list for
# python:3.12-slim-bookworm, refreshed 2026-10-06). The digest moves whenever
# the tag is rebuilt for security updates — re-pin periodically:
#   crane digest python:3.12-slim-bookworm   # or: docker buildx imagetools inspect
# Keeping the tag alongside the digest documents which release line we track.

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Non-root user; app-owned dirs (feedback store, key/config mounts).
RUN groupadd -r bot && useradd -r -g bot bot \
    && mkdir -p /app/feedback_data /app/keys \
    && chown -R bot:bot /app

# Runtime gets only the installed package — no build tools, no source tree,
# no .env / keys / config.yaml (those are mounted or passed at run time).
COPY --from=builder /install /usr/local
COPY config.example.yaml ./config.example.yaml

USER bot

EXPOSE 8000

# Liveness probe: the process is up and serving HTTP.
# (No curl in slim images — stdlib urllib keeps the layer minimal.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5).getcode() == 200 else 1)"

# Default: run the API. Override command to `worker` for the arq worker.
# docker compose handles this; see docker-compose.yml.
CMD ["uvicorn", "prism.main:app", "--host", "0.0.0.0", "--port", "8000"]
