# PRism Fine-Tuning Runbook (QLoRA, Llama 3 8B)

This guide covers the **real GPU training run** that produces PRism's review
model: QLoRA fine-tuning of `meta-llama/Meta-Llama-3-8B-Instruct` on ~5K
PR-review SFT samples, emitting strict-JSON findings.

Pipeline layout (`src/prism/training/`):

| Module      | Role                                                                 |
|-------------|----------------------------------------------------------------------|
| `config.py` | Pydantic config; YAML-loadable; CPU-safe import                      |
| `data.py`   | JSONL → chat template → tokenize → response-only labels (`-100`)      |
| `trainer.py`| 4-bit NF4 + LoRA + TRL SFTTrainer, MLflow, ckpts, early stop, resume  |
| `smoke.py`  | CPU smoke test: tiny random GPT-2, 5 steps, adapter save/reload/gen  |

---

## 1. GPU machine requirements

- **1× GPU with ≥ 24 GB VRAM** (RTX 4090, L4, A10G, A100-40GB…). The default
  config (`configs/train_qlora.yaml`) peaks at ~16–18 GB:
  4-bit base weights ~4.6 GB + 8-bit paged optimizer ~2.5 GB + activations
  with gradient checkpointing ~8–10 GB at `max_seq_len=4096`, eff. batch 16.
- **CUDA 12.1+** with a matching PyTorch build; **Ampere+** for `bf16: true`
  (older GPUs: set `bf16: false`, expect slower training).
- **Disk:** ~30 GB free (model weights ~16 GB fp16 download cache, dataset,
  checkpoints, MLflow artifacts).
- **RAM:** ≥ 16 GB recommended for data preprocessing.

## 2. Environment setup

```bash
# Python 3.11+ (3.12 used in dev)
python -m venv .venv && source .venv/bin/activate

# PyTorch with CUDA 12.1 (CPU-only index will NOT work for training)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Pinned, mutually-verified training stack
pip install "transformers==4.44.2" "peft==0.12.0" "trl==0.9.6" \
            "bitsandbytes==0.43.3" "accelerate==0.33.0" \
            "datasets==2.21.0" "mlflow==2.16.2"

# PRism itself (editable) + dev tools
pip install -e ".[dev]"
```

bitsandbytes must match your CUDA; if `import bitsandbytes` warns about a
CUDA mismatch, install the build for your toolkit (see
https://github.com/bitsandbytes-foundation/bitsandbytes#installation).

### Dependency group for `pyproject.toml` (for the integrator)

```toml
[project.optional-dependencies]
training = [
    "torch>=2.4",            # CUDA build via the PyTorch cu121 index
    "transformers==4.44.2",
    "peft==0.12.0",
    "trl==0.9.6",
    "bitsandbytes==0.43.3",
    "accelerate==0.33.0",
    "datasets==2.21.0",
    "mlflow==2.16.2",
]
```

## 3. HuggingFace token (gated Llama 3 weights)

`meta-llama/Meta-Llama-3-8B-Instruct` is **gated**: you must accept the
license at <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>
(logged in), then provide a token with **read** scope:

```bash
export HF_TOKEN="hf_..."   # read scope is enough; never commit it
```

The token is read from the environment only — it never appears in code,
configs, or logs. (The team's token is also stored in the Secure Vault
under `custom.huggingface` for other flows; the training script itself uses
`HF_TOKEN` from the environment.)

## 4. Data

Training data is JSONL, one record per line:

```json
{"messages": [
  {"role": "system", "content": "You are a senior staff engineer..."},
  {"role": "user",   "content": "Review the following changes to `auth.py`..."},
  {"role": "assistant", "content": "{\"findings\": [...]}"}
]}
```

- `system` is optional — the PRism review system prompt
  (`prism.review.prompts.SYSTEM_PROMPT`) is prepended when missing.
- The last message **must** be `assistant` (the strict-JSON target).
- Malformed records are skipped with a warning (see `format_records`).

Point the config at your files:

```yaml
train_path: "data/sft/train.jsonl"
val_path: "data/sft/val.jsonl"   # or omit -> val_size split of train_path
```

## 5. How to run

```bash
# 0. Sanity: CPU smoke test (no GPU/token needed, < 5 min)
python scripts/train.py --smoke

# 1. Full QLoRA run
python scripts/train.py --config configs/train_qlora.yaml
```

What happens: tokenizer load → 4-bit NF4 base model on `device_map="auto"` →
LoRA adapters (r=16 on all attention+MLP projections, ~40 M trainable params)
→ SFT with response-only loss → eval every 100 steps, checkpoints every 100
steps (keep last 3), early stopping on `eval_loss` (patience 3) → best
adapter saved to `outputs/prism-qlora-llama3-8b/final-adapter/` along with
the tokenizer and the exact `train_config.yaml`.

MLflow: metrics stream via transformers' MLflowCallback; hyperparams,
library versions and GPU info are logged by `trainer.py` at train start.
Set `MLFLOW_TRACKING_URI` to point at your tracking server (defaults to
local `./mlruns`).

## 6. How to resume

Checkpoints live under `<output_dir>/checkpoint-<step>/`. Resume with the
latest checkpoint auto-detected:

```bash
python scripts/train.py --config configs/train_qlora.yaml --resume
```

or pin an explicit checkpoint:

```bash
python scripts/train.py --config configs/train_qlora.yaml \
    --resume-from outputs/prism-qlora-llama3-8b/checkpoint-500
```

You can also set `resume_from_checkpoint:` directly in the YAML. Optimizer,
scheduler and RNG state are restored, so training continues seamlessly.

## 7. Promoting the adapter to serving

The serving stack loads **base model + LoRA adapter** (see the serving
workstream). To promote a trained adapter:

```bash
# The run already wrote everything needed:
ls outputs/prism-qlora-llama3-8b/final-adapter/
#   adapter_model.safetensors  adapter_config.json  tokenizer.*  train_config.yaml

# Copy to the served model path:
cp -r outputs/prism-qlora-llama3-8b/final-adapter /models/prism-review/v1/
```

`train_config.yaml` inside the adapter dir records the exact hyperparams
and `model_id` the adapter was trained against — the serving loader must
use that same base model id.

## 8. Expected metrics

For ~5K samples, 3 epochs, eff. batch 16 (~940 steps):

| Metric | Healthy range |
|---|---|
| Train loss (final) | 0.3 – 0.8, smoothly decreasing |
| Eval loss | tracks train loss; gap < ~0.3 at end |
| Eval/train gap | widening gap + eval rising = overfitting → stop earlier / more data |
| JSON validity (eval benchmark) | > 95 % parseable by `prism.review.schemas.parse_findings` |

Loss is response-only (prompt tokens masked), so absolute values are lower
than full-sequence LM loss — compare runs against each other, not against
pretraining numbers.

## 9. Troubleshooting

**OOM (CUDA out of memory)**
- First lever: `per_device_train_batch_size: 1` + raise
  `gradient_accumulation_steps` to keep the effective batch.
- Second: `max_seq_len: 2048` (halves activation memory).
- Keep `gradient_checkpointing: true` and `optim: paged_adamw_8bit`.
- `group_by_length: true` already minimizes padding waste.

**NaN loss**
- Usually a bad record (e.g. empty assistant message) or LR too high.
  The data loader drops invalid records and logs counts — check for
  `dropping record` warnings. If NaN appears mid-run, halve
  `learning_rate` and resume from the last good checkpoint.

**`evaluation_strategy` / TRL API errors**
- You installed versions other than the pinned set. The code targets
  `transformers==4.44.2` + `trl==0.9.6`; newer transformers renamed
  `evaluation_strategy` → `eval_strategy` and newer TRL changed
  `SFTTrainer` kwargs. Either install the pinned set or adapt
  `build_training_args`/`build_trainer` accordingly.

**bitsandbytes CUDA errors**
- `bitsandbytes` was built for a different CUDA toolkit than your driver.
  Reinstall the matching wheel, or set `load_in_4bit: false` (falls back
  to bf16 full weights — needs ~2× VRAM, likely OOM on 24 GB for 8B).

**MLflow connection refused**
- Training continues; metrics are just not uploaded. Set
  `MLFLOW_TRACKING_URI` correctly or run with an empty `report_to`
  (edit `build_trainer(..., log_to_mlflow=False)` for air-gapped runs).

**Slow data loading**
- `dataloader_num_workers: 2` is a middle ground; raise to 4 on beefy
  CPU machines, drop to 0 if workers crash (shared-memory limits).

**Smoke test passes but GPU run fails at model load**
- The smoke test deliberately does NOT cover 4-bit quantization or the
  gated Llama 3 download (both need a GPU + `HF_TOKEN`). Verify those two
  with: `python -c "import bitsandbytes"` and a gated-repo `huggingface-cli
  download meta-llama/Meta-Llama-3-8B-Instruct --include config.json`.
