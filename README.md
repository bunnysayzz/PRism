# PRism 🤖

PRism is an **autonomous PR review bot** delivered as a **GitHub App**. It reviews every
pull request with your fine-tuned code-review LLM, gates findings through a
confidence system with a stronger-model second pass, and posts inline review
comments — while every human reaction feeds a retraining loop that makes the
reviewer smarter over time.

## Architecture

```
                        ┌────────────────────────────── REVIEW PATH ──────────────────────────────┐
                        │                                                                       │
  GitHub PR             │   POST /webhooks/github                                               │
  opened /              │   ├─ token-bucket rate limit (per installation, 30/min)               │
  synchronize           │   ├─ HMAC-SHA256 signature verify                                     │
        │               │   └─ parse event → ReviewJob                                          │
        ▼               │             │                                                         │
  ┌──────────┐    ┌─────┴──────┐      │                                                         │
  │ GitHub   │───▶│  FastAPI   │──────┼──▶ arq queue (Redis) ──▶ worker: review_pr               │
  │ webhooks │    │  /healthz  │      │    (dev: REDIS_URL unset → runs inline, no Redis)        │
  └──────────┘    │  /readyz   │      │              │                                          │
                  └────────────┘      │              ├─ mint installation token (App JWT)        │
                                      │              ├─ fetch PR files → filter → per-file LLM  │
                                      │              ▼                                         │
                                      │       ┌──────────────┐                                 │
                                      │       │ CONFIDENCE   │  ≥ 0.80  → post directly        │
                                      │       │ GATE         │  0.55–0.80 → verify via stronger │
                                      │       │              │    model (fail closed on errors) │
                                      │       │              │  < 0.55  → drop                   │
                                      │       └──────┬───────┘                                 │
                                      │              ▼                                         │
                                      │       line validation → severity/category filters →     │
                                      │       dedup vs existing bot comments (fingerprint)     │
                                      │              │                                         │
                                      │              ▼                                         │
  ┌──────────┐                        │    POST /repos/{o}/{r}/pulls/{n}/reviews                │
  │ GitHub   │◀───────────────────────┴──  {event: "COMMENT", summary + inline comments}       │
  │ PR page  │                                                                               │
  └──────────┘                                                                               │

  ┌────────────────────────────── DATA / TRAINING PATH ─────────────────────────────┐
  │  every posted finding → feedback_data/findings.jsonl                            │
  │  human 👍/👎/edits → outcomes.jsonl → export → train.jsonl                       │
  │        │                                                                        │
  │        ▼                                                                        │
  │  fine-tune LoRA adapter ──(MLflow tracked)──▶ vLLM --enable-lora                │
  │        │                                         ▲                              │
  │        └──────── eval gate (golden set, F1 ≥ 0.70 blocks deploy) ────────────────┘
  └──────────────────────────────────────────────────────────────────────────────────┘
```

### Services

| Service | What it is | Port |
|---|---|---|
| `api` | FastAPI webhook receiver + health endpoints | 8000 |
| `worker` | arq worker running `review_pr` jobs | — |
| `redis` | Job queue (arq) | 6379 |
| `mlflow` | Experiment tracking for training/eval | 5000 |
| `vllm` | Model serving with LoRA adapter (`--profile serving`, GPU) | 8000→8001 |

## Quickstart

### GitHub App setup

1. Go to **github.com/settings/apps → New GitHub App** (or your org's settings).
2. Name it (e.g. `PRism`), set a homepage URL.
3. **Webhook URL:** `https://<your-host>/webhooks/github` — must be publicly reachable
   (use [smee.io](https://smee.io) or ngrok for local dev).
4. **Webhook secret:** generate one (`openssl rand -hex 32`) and save it as `GITHUB_WEBHOOK_SECRET`.
5. **Permissions** (Repository):
   - **Pull requests:** Read & write (read diffs, post reviews)
   - **Contents:** Read (fetch file context around diffs)
   - **Metadata:** Read (mandatory for all Apps)
6. **Subscribe to events:** check **Pull request** and **Pull request review**.
7. Create the App, then **generate a private key** — download the `.pem` file.
   Save it as `private-key.pem` next to `.env` (it's gitignored; never commit it).
8. **Install the App** on the repos you want reviewed. Note the **App ID** (numeric).

### Run with Docker (recommended)

```bash
cp .env.example .env
cp config.example.yaml config.yaml
# edit .env: GITHUB_APP_ID, GITHUB_PRIVATE_KEY_PATH=./private-key.pem,
#            GITHUB_WEBHOOK_SECRET, backend settings...

docker compose up --build
# api    → http://localhost:8000  (POST /webhooks/github, GET /healthz, GET /readyz)
# worker → arq worker consuming the Redis queue
# mlflow → http://localhost:5000
```

### Run locally (dev-inline mode, no Redis)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # leave REDIS_URL unset → webhook runs the review in-process
uvicorn prism.main:app --reload
```

Point your webhook forwarder (smee/ngrok) at `http://localhost:8000/webhooks/github`,
open a PR on an installed repo, and watch the review land.

### Serve the fine-tuned model (GPU host)

```bash
# 1. train the LoRA adapter (see "Feedback → retrain loop" below)
# 2. start everything including vLLM:
VLLM_BASE_URL=http://vllm:8000 PRISM_ADAPTER_DIR=/path/to/adapter \
  docker compose --profile serving up --build
```

This starts the `vllm` service (`vllm/vllm-openai`) with
`--enable-lora --lora-modules code-review=/adapter`, exposing the
OpenAI-compatible API the app already talks to. Needs
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
on the host. The profile is off by default so CPU-only machines skip it.

## Configuration reference

All settings are env vars (see `.env.example`); `config.yaml` holds behavior
config. New/prod-related vars are marked ★.

| Variable | Default | What it does |
|---|---|---|
| `GITHUB_APP_ID` | — | Numeric GitHub App ID |
| `GITHUB_PRIVATE_KEY_PATH` | — | Path to the App `.pem` (loaded from file only) |
| `GITHUB_WEBHOOK_SECRET` | — | HMAC secret for webhook verification |
| `REDIS_URL` | unset | Set → queue via arq. Unset → dev-inline mode |
| `REVIEW_BACKEND` | `vllm` | `vllm` \| `hf_endpoint` \| `api` \| `stub` |
| `VLLM_BASE_URL` / `VLLM_MODEL` | `http://localhost:8000` | vLLM OpenAI-compatible endpoint + model |
| `VLLM_TIMEOUT_S` | `120` | vLLM request timeout (s) |
| `HF_ENDPOINT_URL` / `HF_TOKEN` | — | Hugging Face Inference Endpoint |
| `HF_TIMEOUT_S` | `120` | HF endpoint timeout (s) |
| `FALLBACK_API_BASE` / `FALLBACK_API_KEY` / `FALLBACK_MODEL` | `gpt-4o` | Stronger model for the verify second pass |
| `CONFIDENCE_AUTO_POST` | `0.80` | ≥ this → post directly |
| `CONFIDENCE_VERIFY` | `0.55` | ≥ this → verify; below → drop. Must be ≤ `CONFIDENCE_AUTO_POST` (validated) |
| `MAX_FILES_PER_PR` | `30` | Files reviewed per PR (rest listed as skipped) |
| `MAX_CONCURRENT_FILES` | `4` | Backend concurrency |
| `MAX_FILE_BYTES` | `200000` | Skip files larger than this |
| `GITHUB_TIMEOUT_S` | `30` | GitHub API timeout (s) |
| `REPO_CONFIG_PATH` | `config.yaml` | Behavior config (allowlists, globs, categories) |
| `FEEDBACK_DIR` | `feedback_data` | JSONL feedback store |
| `LOG_LEVEL` | `INFO` | Log level |
| ★ `LOG_FORMAT` | `json` | `json` (prod) \| `console` (local dev) |
| ★ `RATE_LIMIT_WEBHOOK_PER_MIN` | `30` | Webhook token-bucket rate per installation |
| ★ `PRISM_ADAPTER_DIR` | — | LoRA adapter dir (mounted into vLLM) |
| ★ `MLFLOW_TRACKING_URI` | — | MLflow tracking server URI |
| ★ `STRUCTURED_OUTPUT_MAX_RETRIES` | `2` | Retries on schema-invalid model output |
| ★ `EVAL_MIN_F1` | `0.70` | Eval gate threshold (CI fails below) |

`config.yaml` controls: `repos_allowlist` (empty = all installed repos),
`severity_threshold` (`low|medium|high|critical`), `ignore_globs`,
per-`categories` enable/disable (style is off by default), `max_findings_per_file`.

### Observability

- `GET /healthz` — liveness, always 200 when the process is up.
- `GET /readyz` — readiness: checks settings validity, probes the review backend
  (`GET <vllm>/health`, skipped for `stub`), and pings Redis when configured.
  Returns **503** with per-check details when not ready — wire your
  load-balancer / orchestrator to this, not `/healthz`.
- Every request gets an `X-Request-ID` header (yours is echoed, otherwise
  generated) and it's attached to all structured logs for that request.
- `Settings.redacted()` / `prism.config.safe_dump()` dump the full config with
  secrets masked — safe for startup logs and support bundles.

## The confidence gate

The model self-reports a confidence per finding. The gate keeps precision high
without paying flagship-model prices on every file:

- **≥ `CONFIDENCE_AUTO_POST` (0.80):** posted directly.
- **`CONFIDENCE_VERIFY`–`AUTO_POST`:** re-asked to the stronger fallback model
  ("is this a real issue? yes/no + revised confidence"). Kept only on a "real"
  verdict; confidence replaced by the revised value. **Fail closed:** verifier
  errors or unparseable answers → finding dropped.
- **< `CONFIDENCE_VERIFY` (0.55):** dropped silently.

Every kept finding is then validated against the diff — GitHub rejects comments
on lines outside the diff, so off-diff lines are clamped to the nearest
commentable line (±3) or dropped.

## Feedback → retrain loop

1. Every posted finding is logged to `feedback_data/findings.jsonl` with a stable ID.
2. Human reactions are recorded via `FeedbackStore.record_outcome(id, "accepted" | "dismissed" | "edited")`
   (wire this to comment 👍/👎 or replies with a small follow-up job).
3. Export training pairs:
   ```bash
   python -m prism.feedback.export --dir feedback_data --out train.jsonl --min-outcomes 50
   ```
   Accepted → reproduce the finding. Dismissed → `{"findings": []}` (teaches what
   *not* to flag — this is where precision comes from). Edited → human-corrected text.
4. Fine-tune the LoRA adapter on `train.jsonl` (track the run in MLflow),
   re-run the eval gate, deploy the new adapter via the `serving` profile.

## Eval gate

`eval/golden.example.jsonl` holds golden cases with ground-truth issues. The runner replays the
real pipeline — backend → confidence gate → line validation — and scores
precision / recall / F1 plus a severity-weighted F1:

```bash
python -m prism.eval.runner --golden eval/golden.example.jsonl --backend stub --min-f1 0.70
```

CI runs this on every PR and **fails the build if F1 < 0.70** (threshold =
`EVAL_MIN_F1`). For real-model evals, use `--backend vllm` (or `api`/`hf_endpoint`) with your own
`eval/golden.jsonl`. Matching rule: predicted ≡ expected when categories match
and line numbers agree within ±2.

The full benchmark harness (`python -m prism.eval.harness`) additionally scores
per-category / per-severity F1, false-positive rate, confidence calibration (ECE),
and compares the candidate against a heuristic baseline — see the CI
`eval-gate` workflow. All CI/CD, training-pipeline, and deploy workflows are
documented in [docs/WORKFLOWS.md](docs/WORKFLOWS.md).

## Runbooks

### Deploy

```bash
# 1. build & start
docker compose up --build -d
# 2. check readiness (not just liveness)
curl -f http://localhost:8000/readyz | python -m json.tool
# 3. point the GitHub App webhook at https://<host>/webhooks/github
# 4. open a test PR on an installed repo and confirm the review lands
```

Roll back: `docker compose down && docker compose up -d` on the previous image
tag (tag your builds; `:latest` is not a rollback strategy).

### Rotate the webhook secret

1. Generate a new secret: `openssl rand -hex 32`.
2. In the GitHub App settings, set the new webhook secret — GitHub starts
   signing with it immediately.
3. Update `GITHUB_WEBHOOK_SECRET` in `.env` and restart the api service:
   `docker compose up -d api`. (Deliveries signed with the old secret in the
   ~seconds between steps 2–3 get a 401 and are safely ignored; GitHub does not
   retry them as PR events re-fire on the next synchronize.)
4. Verify: `curl -f http://localhost:8000/readyz`.

To rotate the App private key instead: generate a new key in App settings,
replace `private-key.pem`, restart api + worker.

### Retrain loop (scheduled)

```bash
# export fresh pairs, train, eval-gate, deploy
python -m prism.feedback.export --dir feedback_data --out train.jsonl --min-outcomes 50
# ... fine-tune LoRA on train.jsonl, log to MLflow ...
python -m prism.eval.runner --golden eval/golden.jsonl --backend vllm \
  --min-f1 "${EVAL_MIN_F1:-0.70}"
# gate passed → copy adapter to $PRISM_ADAPTER_DIR and restart the serving profile
docker compose --profile serving up -d vllm
```

Automate the export+train on a cron/CI schedule; the eval gate is the deploy
blocker, never skip it.

## Security notes

- Webhook bodies are HMAC-SHA256 verified with constant-time comparison;
  verification failures are logged **without** the secret or digest
  (`webhook_bad_signature` carries only event type + client).
- The App private key is loaded from a file path only — never from env contents,
  never baked into the image (mounted read-only).
- Secrets are redacted from all structured logs (any key containing
  `secret`/`token`/`key`/`password`… is masked, recursively). `LOG_FORMAT=json`
  keeps them machine-greppable without values.
- `Settings.redacted()` / `safe_dump()` for any config you print or ship.
- Webhook ingestion is rate-limited (token bucket, `RATE_LIMIT_WEBHOOK_PER_MIN`
  per installation, 429 + `Retry-After`). **Note:** the limiter is in-memory and
  single-process — for multi-replica deployments, move it to Redis (future work).
- Installation tokens are cached in memory and refreshed 60s before expiry.
- The bot only *comments* (`event: "COMMENT"`) — it never approves, requests changes, or mutates code.
- No secrets in the repo: `.env`, `private-key.pem`, `*.pem`, `config.yaml`,
  `feedback_data/` are all gitignored and excluded from the Docker build context
  (`.dockerignore`).

## Project layout

```
src/prism/
  main.py            FastAPI: /webhooks/github, /healthz, /readyz, rate limiting
  worker.py          arq worker: review_pr job
  config.py          pydantic-settings + YAML repo config + redacted() dump
  logging.py         structlog JSON/console logging (secrets redacted)
  github/
    auth.py          App JWT (RS256) → installation token (cached)
    client.py        httpx client: retries, backoff, rate-limit respect
    webhooks.py      HMAC verify + event dispatch
    reviews.py       review payload builder + posting + dedup fetch
  diff/
    fetcher.py       list/filter PR files, fetch base/head context
    hunks.py         unified-diff parsing → commentable lines
  review/
    schemas.py       Finding / ReviewResult (pydantic v2)
    prompts.py       reviewer system + per-file prompts
    backends.py      vLLM / HF endpoint / OpenAI-compatible / stub
    engine.py        orchestration + confidence gate
    dedup.py         finding fingerprints
  feedback/
    store.py         JSONL findings + outcomes
    export.py        → fine-tune train.jsonl
  eval/
    runner.py        golden eval + F1 gate
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q        # full suite
.venv/bin/python -m ruff check src tests    # lint
.venv/bin/python -m ruff format --check src tests
.venv/bin/python -m mypy src                # strict type check
```
