# PRism CI/CD Runbook

Every workflow in `.github/workflows/`: what it does, when it runs, what
secrets it needs, and what to do when it fails.

## Workflow overview

| Workflow | Purpose | Trigger |
|---|---|---|
| `ci.yml` | ruff check + format, mypy strict on `src/`, full pytest suite, golden eval gate (stub backend, min F1 0.70) | push to `main`, `pull_request` |
| `eval-gate.yml` | Deeper PR quality gate: full golden eval harness + eval report artifact; optional real-model backend via manual dispatch | `pull_request`, `workflow_dispatch` |
| `docker.yml` | Build image on PRs (build-only + smoke test); build+push multi-arch to GHCR on `main` and `v*` tags, then smoke-test the pushed image | `pull_request`, push to `main`, tags `v*` |
| `training.yml` | ML loop: data collection → QLoRA fine-tune → eval vs baseline → adapter artifact + model card → GitHub Release on pass | `workflow_dispatch` only (schedule commented out until GPU runner is online) |
| `deploy.yml` | SSH into VPS, `docker compose pull && up` the chosen image tag, post-deploy `/healthz` check | `workflow_dispatch` (staging/prod choice) |

## Secrets

Set in **Settings → Secrets and variables → Actions**. Never hardcode values
in workflow files — only `${{ secrets.NAME }}` references are used.

| Secret | Purpose | Used by |
|---|---|---|
| `GITHUB_TOKEN` | Built-in token. GHCR login/push in `docker.yml`; GHCR read in `deploy.yml`. Works automatically — no setup needed, but the repo must allow it (see "GHCR auth"). | docker, deploy |
| `GH_DATA_TOKEN` | Personal access token (classic or fine-grained) with `repo` read scope, used by `scripts/collect_pr_data.py` to fetch PR diffs for training data. Keep separate from deploy tokens. | training |
| `HF_TOKEN` | Hugging Face token: download base model in training, optional adapter push to the Hub. | training |
| `HF_ENDPOINT_URL` | Inference endpoint URL serving the fine-tuned adapter, used by the eval harness (`--backend hf_endpoint`). | training (evaluate job), eval-gate (manual) |
| `PRISM_LLM_API_KEY` | API key for the `--backend api` path (real-model PR eval), e.g. OpenAI/Anthropic key the review backend is configured for. Only needed for manual real-backend evals. | eval-gate (manual) |
| `WANDB_API_KEY` | Weights & Biases key for training run logging. Optional — training proceeds without it. | training |
| `GHCR_PAT` | PAT with `read:packages` (classic) or fine-grained equivalent, so the VPS can `docker login ghcr.io` and pull private images. Only needed if the GHCR package is private. | deploy |
| `STAGING_HOST` | Hostname/IP of the staging VPS. | deploy |
| `PROD_HOST` | Hostname/IP of the production VPS. | deploy |
| `APP_USER` | SSH username on the VPS (e.g. `prism` or `ubuntu`). | deploy |
| `SSH_PRIVATE_KEY` | Private key for `APP_USER`. Pair it with the public key in the VPS's `authorized_keys`. | deploy |
| `SSH_PORT` | SSH port. Optional — defaults to 22 if unset. | deploy |

## GPU runner setup (training)

The `training.yml` `train` and `evaluate` jobs run on a **self-hosted GPU
runner** (`runs-on: [self-hosted, gpu]`). Until one is online, training runs
queue forever — keep the cron schedule commented out.

1. Provision a machine with an NVIDIA GPU (e.g. a cloud GPU instance) and
   install the NVIDIA container toolkit + the GitHub Actions runner agent.
2. Register the runner with labels `self-hosted` and `gpu`:
   `./config.sh --url https://github.com/<owner>/PRism --token <TOKEN> --labels self-hosted,gpu`
3. Verify it appears under Settings → Actions → Runners and is "Idle".
4. Ensure it has Python 3.12, `pip`, and enough disk for the base model +
   dataset. PyTorch with CUDA is installed by the workflow itself.
5. Only then: uncomment the `schedule` cron in `training.yml` for unattended runs.

Security note: self-hosted runners execute arbitrary code from PRs if a
workflow triggers on `pull_request` — `training.yml` is dispatch-only, which
is why. Never add a `pull_request` trigger to it.

## GHCR auth

- Pushing: `docker.yml` logs in with `GITHUB_TOKEN` (granted `packages: write`
  in the workflow). Requires **Settings → Actions → General → Workflow
  permissions: "Read and write permissions"**, or the explicit `permissions:`
  block already in the file.
- Package visibility: GHCR packages default to the repo's visibility. For a
  public repo the image is public and deploy needs no login; for a private
  repo, set `GHCR_PAT` so the VPS can pull.
- First push: the package is created under the repo owner's GHCR namespace
  as `ghcr.io/<owner>/prism`.

## Environments & approvals (deploy)

Create two GitHub Environments (**Settings → Environments**): `staging` and
`prod`.

- `prod`: add **Required reviewers** (e.g. Tuba) so every production deploy
  needs a manual approval after dispatch. Add a wait timer if desired.
- `staging`: no reviewers — deploys freely for testing.
- `deploy.yml` sets `environment: ${{ inputs.environment }}`, so the right
  protection rules apply automatically.

Host-side prerequisites (do once per VPS):

```bash
# on the VPS
sudo mkdir -p /opt/prism && cd /opt/prism
# place: docker-compose.yml (api/worker services using $PRISM_IMAGE),
# .env (GitHub App creds, REDIS_URL), config.yaml, private-key.pem (0600)
docker compose up -d   # first manual bring-up; workflow handles the rest
```

The compose file on the host must reference the image as
`image: ${PRISM_IMAGE:-ghcr.io/<owner>/prism:latest}` for `api` and `worker`.

## What to do when X fails

### ci.yml — lint
- `ruff check`/`ruff format --check` failures: run `.venv/bin/ruff check --fix src tests && .venv/bin/ruff format src tests` locally, commit, push.

### ci.yml — typecheck
- `mypy src` is **strict**. Read the error, add/adjust type annotations —
  do not sprinkle `# type: ignore` (the config has `warn_unused_ignores`).
  Verify locally: `.venv/bin/mypy src`.

### ci.yml — test
- Run `.venv/bin/python -m pytest -q` locally. If only CI fails, suspect
  environment differences (Python version — CI pins 3.12; timezone —
  tests must not depend on local tz).

### ci.yml / eval-gate.yml — eval gate fails (F1 < 0.70)
- The eval report artifact (`eval-report-*`) shows per-case P/R/F1. Find the
  cases with F1 drops: usually a pipeline change broke line validation,
  the confidence gate thresholds in config, or the matching logic.
- Repro locally: `python -m prism.eval.runner --golden eval/golden.jsonl --backend stub --min-f1 0.70`
  (uses `eval/golden.example.jsonl` if the full set is absent).
- If the drop is legitimate (golden set got harder), update the threshold
  deliberately in a PR — never lower it to make red green without review.

### eval-gate.yml — real backend run fails
- Manual dispatch only. Check the three backend secrets (`PRISM_LLM_API_KEY`,
  `HF_ENDPOINT_URL`, `HF_TOKEN`) and that the backend name matches what
  `prism.review.backends.build_backend` supports (`stub | vllm | api | hf_endpoint`).

### docker.yml — build fails
- Usually a dependency or Dockerfile issue. Repro: `docker build .` locally.
  The GHA cache (`type=gha`) sometimes goes stale — re-running usually fixes;
  if not, bump a layer by touching requirements.

### docker.yml — smoke test fails (/healthz)
- The container started but the app didn't come up. Pull the PR image
  (`prism:smoke` tag in the PR run) and check `docker logs`: typical causes
  are missing env at import time or a bad CMD. Note `/healthz` needs no
  Redis — if even that fails, the app crashed on startup.

### training.yml — prepare fails
- `scripts/collect_pr_data.py` or `configs/train_qlora.yaml` missing: the
  corresponding worker hasn't merged yet. Merge those PRs first.
- GitHub API rate limits / auth errors: check `GH_DATA_TOKEN` scope and expiry.

### training.yml — train fails (GPU runner)
- "No runner matched labels": the GPU runner is offline or mislabeled — see
  "GPU runner setup". Runners appear under Settings → Actions → Runners.
- CUDA OOM: lower batch size / gradient accumulation in
  `configs/train_qlora.yaml`, or move to a larger GPU.

### training.yml — evaluate fails (below min F1)
- Expected sometimes: the new adapter didn't beat the gate. The release job
  is skipped automatically. Inspect `adapter-eval-<run_id>` artifact,
  compare against the previous release's F1, and iterate on data/config.

### deploy.yml — SSH fails
- Check `STAGING_HOST`/`PROD_HOST`, `APP_USER`, `SSH_PRIVATE_KEY`
  (must be the *private* key, no passphrase or use ssh-agent — the action
  takes the raw key), and that port 22/SSH_PORT is reachable from GitHub
  runners.

### deploy.yml — post-deploy /healthz fails
- The workflow exits non-zero; the old containers are still running only if
  `docker compose up -d` itself failed before replacing them. Check
  `docker compose logs api` on the host. **Rollback:** re-run `deploy.yml`
  with the previous working `image_tag` (tags are immutable per push, so the
  old image is still in GHCR).

## Branch protection / repo settings checklist

Configure in **Settings → Branches → Add branch protection rule** for `main`:

- [ ] Require a pull request before merging (dismiss stale approvals on new pushes)
- [ ] Require status checks to pass: `ruff`, `mypy`, `pytest`, `eval gate (golden F1)`, `golden eval` (eval-gate.yml), `build` (docker.yml)
- [ ] Require branches to be up to date before merging
- [ ] Do not allow bypassing the above settings (or allow only admins)
- [ ] Require conversation resolution before merging (optional but recommended)
- [ ] After merging: tags matching `v*` trigger `docker.yml` push — create releases from tags
- [ ] Settings → Actions → General → Workflow permissions: **Read and write** (needed for GHCR push and release creation)
- [ ] Settings → Environments: create `staging` (no reviewers) and `prod` (required reviewers)
- [ ] Settings → Secrets and variables → Actions: fill the secrets table above

## Future work

- **Per-category eval minimums** (e.g. security ≥ 0.8): the harness CLI
  (`src/prism/eval/runner.py`) currently only gates global micro-F1 via
  `--min-f1`. Adding `--min-f1-category security:0.8,...` to the runner would
  let `eval-gate.yml` enforce it.
- **`/readyz` endpoint**: only `/healthz` exists in `src/prism/main.py`.
  When a readiness check (Redis + config validity) lands, extend the deploy
  health check to curl it too.
- **MLflow model registry**: the registry of record for adapters. `training.yml`
  creates a GitHub Release with the adapter + model card; promoting a release
  artifact into MLflow (`prism-review` registered model) is currently manual.
- **`train` extra in pyproject.toml**: `training.yml` installs torch/transformers/
  peft/trl/datasets/accelerate/bitsandbytes inline. Moving these to a `[train]`
  extra would make local and CI installs identical.
- **Full `eval/golden.jsonl`**: CI and gates fall back to
  `eval/golden.example.jsonl` until the real golden set is committed.
