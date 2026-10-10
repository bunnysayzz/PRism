#!/usr/bin/env bash
# Heroku web boot for PRism.
# - Binds Heroku's $PORT (assigned dynamically per dyno).
# - Materializes the GitHub App private key from the GITHUB_PRIVATE_KEY_PEM
#   config var into an ephemeral file, because the app reads the key from a
#   path (GITHUB_PRIVATE_KEY_PATH) and Heroku has no secret-file mounts.
# - Requires FALLBACK_API_KEY: with no adapter, HF endpoint, or API key, the
#   model router silently falls back to the dev-only stub backend and posts
#   placeholder reviews to real PRs. Fail loudly here instead.
set -euo pipefail

: "${PORT:?PORT is not set (Heroku provides it on every dyno)}"
: "${GITHUB_PRIVATE_KEY_PEM:?Set the GITHUB_PRIVATE_KEY_PEM config var with the App private key}"
: "${FALLBACK_API_KEY:?Set FALLBACK_API_KEY (an OpenAI-compatible API key) — without it the bot posts stub placeholder reviews}"

KEY_PATH=/tmp/prism-private-key.pem
printf '%s' "$GITHUB_PRIVATE_KEY_PEM" > "$KEY_PATH"
chmod 600 "$KEY_PATH"
export GITHUB_PRIVATE_KEY_PATH="$KEY_PATH"

exec uvicorn prism.main:app --host 0.0.0.0 --port "$PORT"
