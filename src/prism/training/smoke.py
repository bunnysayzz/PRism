"""CPU smoke test for the PRism fine-tuning pipeline.

Proves the full loop on a machine with no GPU, in well under 5 minutes:

1. Builds 8 synthetic PR-review SFT records and formats them with the real
   :mod:`prism.training.data` code (chat template + response-only masking).
2. Builds a tiny randomly-initialized GPT-2-style model (2 layers, 128-dim)
   and attaches a real LoRA adapter via :mod:`peft`.
3. Trains for 5 steps with the real TRL ``SFTTrainer`` (via
   :mod:`prism.training.trainer`, MLflow disabled) on CPU.
4. Saves the adapter, reloads it onto a fresh base model, and generates
   tokens — proving the artifact is a working model.

Run: ``python scripts/train.py --smoke`` (or ``PRISM_RUN_SMOKE=1 pytest
tests/test_training.py -m slow``).

Heavy imports (torch, transformers, peft, trl, datasets) live inside
functions so ``import prism.training.smoke`` stays light.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from prism.logging import get_logger, setup_logging
from prism.review.prompts import SYSTEM_PROMPT
from prism.training.config import LoRAParams, TrainConfig

log = get_logger(__name__)

BEGIN_OF_TEXT = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"

_MARKERS = (BEGIN_OF_TEXT, START_HEADER, END_HEADER, EOT)
_MARKER_RE = re.compile("(" + "|".join(re.escape(m) for m in _MARKERS) + ")")


def _split_words(text: str) -> list[str]:
    """Split text into words, keeping chat-template markers as single tokens."""
    tokens: list[str] = []
    for chunk in _MARKER_RE.split(text):
        if not chunk or chunk.isspace():
            continue
        if chunk in _MARKERS:
            tokens.append(chunk)
        else:
            tokens.extend(chunk.split())
    return tokens


def _synthetic_records() -> list[dict[str, Any]]:
    """8 tiny PR-review records in the SFT JSONL schema."""
    diffs = [
        ("auth.py", "@@ -1,3 +1,3 @@\n-    if user:\n+    if user is not None:"),
        ("db.py", "@@ -10,4 +10,5 @@\n+        self.cache[k] = load(k)"),
        ("api.py", "@@ -3,3 +3,4 @@\n+    return render(tpl, user=request.user)"),
        ("util.py", "@@ -7,2 +7,3 @@\n+    assert x > 0"),
    ]
    findings = [
        '{"findings": []}',
        '{"findings": [{"path": "db.py", "line": 12, "severity": "high", '
        '"category": "correctness", "title": "Unbounded cache growth", '
        '"explanation": "cache grows without eviction; long-lived process '
        'will OOM under sustained load.", '
        '"suggestion": "use functools.lru_cache", "confidence": 0.85, '
        '"language_hint": "python"}]}',
        '{"findings": []}',
        '{"findings": [{"path": "util.py", "line": 9, "severity": "medium", '
        '"category": "correctness", "title": "assert for validation", '
        '"explanation": "assert is stripped with -O; invalid input then '
        'flows downstream silently.", "suggestion": "raise ValueError", '
        '"confidence": 0.8, "language_hint": "python"}]}',
    ]
    records: list[dict[str, Any]] = []
    for i in range(8):
        path, hunk = diffs[i % len(diffs)]
        records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Review changes to `{path}`.\n```diff\n{hunk}\n```",
                    },
                    {"role": "assistant", "content": findings[i % len(findings)]},
                ]
            }
        )
    return records


def _make_tokenizer(texts: list[str]) -> Any:
    """Build a word-level HF tokenizer over the given corpus texts."""

    from transformers.tokenization_utils import PreTrainedTokenizer

    # transformers ships no py.typed, so PreTrainedTokenizer is Any to mypy.
    class SimpleTokenizer(PreTrainedTokenizer):  # type: ignore[misc]
        """Minimal word-level tokenizer implementing ChatTokenizer."""

        def __init__(self, corpus: list[str]) -> None:
            specials = ["<pad>", "<eos>", "<unk>"]
            words: set[str] = set()
            for text in corpus:
                words.update(_split_words(text))
            vocab_list = specials + sorted(words - set(specials))
            self._word_vocab: dict[str, int] = {tok: i for i, tok in enumerate(vocab_list)}
            self._word_ids: dict[int, str] = {i: tok for tok, i in self._word_vocab.items()}
            super().__init__(
                pad_token="<pad>",
                eos_token="<eos>",
                unk_token="<unk>",
                model_max_length=4096,
            )

        def get_vocab(self) -> dict[str, int]:
            return dict(self._word_vocab)

        @property
        def vocab_size(self) -> int:
            return len(self._word_vocab)

        def _tokenize(self, text: str) -> list[str]:
            return _split_words(text)

        def _convert_token_to_id(self, token: str) -> int:
            return self._word_vocab.get(token, self._word_vocab["<unk>"])

        def _convert_id_to_token(self, index: int) -> str:
            return self._word_ids.get(index, "<unk>")

        def save_vocabulary(
            self,
            save_directory: str,
            filename_prefix: str | None = None,
        ) -> tuple[str, ...]:
            vocab_file = str(Path(save_directory) / "vocab.txt")
            with open(vocab_file, "w", encoding="utf-8") as fh:
                for token in sorted(self._word_vocab, key=self._word_vocab.get):  # type: ignore[arg-type]
                    fh.write(token + "\n")
            return (vocab_file,)

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            tokenize: bool = False,
            add_generation_prompt: bool = False,
        ) -> str:
            parts = [BEGIN_OF_TEXT]
            for message in messages:
                parts.append(
                    f"{START_HEADER}{message['role']}{END_HEADER}\n\n{message['content']}{EOT}"
                )
            if add_generation_prompt:
                parts.append(f"{START_HEADER}assistant{END_HEADER}\n\n")
            text = "".join(parts)
            if tokenize:
                raise NotImplementedError("tokenize=True not supported in smoke")
            return text

    return SimpleTokenizer(texts)


def _tiny_model(vocab_size: int, pad_id: int, eos_id: int) -> Any:
    """Randomly-initialized 2-layer GPT-2-style causal LM."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=128,
        n_layer=2,
        n_head=4,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        bos_token_id=eos_id,
        use_cache=False,
    )
    return GPT2LMHeadModel(config)


def _smoke_config(train_path: str, output_dir: str) -> TrainConfig:
    """Tiny-CPU overrides of the production config for the smoke run."""
    cfg = TrainConfig(
        train_path=train_path,
        val_size=0.25,
        max_seq_len=1024,
        output_dir=output_dir,
        seed=42,
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=5,
        warmup_ratio=0.0,
        optim="adamw_torch",
        bf16=False,
        gradient_checkpointing=False,
        group_by_length=False,
        dataloader_num_workers=0,
        logging_steps=1,
        eval_steps=5,
        save_steps=5,
        save_total_limit=2,
        early_stopping_patience=100,
        mlflow_experiment="prism-smoke",
    )
    # Tiny rank/targets for the smoke model (real run uses cfg defaults).
    cfg.lora = LoRAParams(r=4, alpha=8, dropout=0.0, target_modules=["c_attn"], bias="none")
    return cfg


def main() -> int:
    """Run the smoke test. Returns 0 on success, 1 on failure."""
    setup_logging("INFO")
    log.info("starting CPU smoke test")
    workdir = Path(tempfile.mkdtemp(prefix="prism-smoke-"))
    try:
        import torch

        from prism.training.data import build_train_eval_datasets
        from prism.training.trainer import (
            apply_lora,
            build_trainer,
            find_latest_checkpoint,
        )

        records = _synthetic_records()
        jsonl_path = workdir / "train.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

        corpus = [SYSTEM_PROMPT, BEGIN_OF_TEXT, START_HEADER, END_HEADER, EOT] + [
            m["content"] for r in records for m in r["messages"]
        ]
        tokenizer = _make_tokenizer(corpus)
        cfg = _smoke_config(str(jsonl_path), str(workdir / "out"))

        # 1-2. Real data pipeline: load JSONL -> chat template -> mask.
        train_ds, eval_ds = build_train_eval_datasets(cfg, tokenizer)
        log.info("smoke datasets", train=len(train_ds), eval=len(eval_ds))

        # 3. Tiny random model + real LoRA + real SFTTrainer, 5 steps on CPU.
        base = _tiny_model(len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id)
        model = apply_lora(base, cfg)
        trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds, log_to_mlflow=False)
        trainer.train()

        losses = [float(entry["loss"]) for entry in trainer.state.log_history if "loss" in entry]
        if not losses or not all(math.isfinite(x) for x in losses):
            log.error("no finite training loss observed", losses=losses)
            return 1
        log.info("training losses", losses=[round(x, 4) for x in losses])

        checkpoint = find_latest_checkpoint(cfg.output_dir)
        if checkpoint is None:
            log.error("expected a checkpoint under output_dir")
            return 1
        log.info("checkpoint saved", checkpoint=checkpoint)

        # 4. Save adapter, reload onto a fresh base model, generate.
        adapter_dir = workdir / "adapter"
        model.save_pretrained(str(adapter_dir))
        weight_files = list(adapter_dir.glob("adapter_model.*"))
        if not weight_files or not (adapter_dir / "adapter_config.json").exists():
            log.error("adapter files missing", dir=str(adapter_dir))
            return 1

        from peft.peft_model import PeftModel

        fresh_base = _tiny_model(len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id)
        # PeftModel.generate is untyped in transformers; treat as Any.
        loaded: Any = PeftModel.from_pretrained(fresh_base, str(adapter_dir))
        loaded.eval()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Review changes to `x.py`."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)])
        with torch.no_grad():
            generated = loaded.generate(
                input_ids,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
        if generated.shape[1] <= input_ids.shape[1]:
            log.error("generation produced no new tokens")
            return 1
        log.info("generation ok", preview=text[:120])

        log.info("smoke test PASSED")
        return 0
    except Exception as exc:  # noqa: BLE001 - smoke must report, not crash
        log.error("smoke test FAILED", error=str(exc))
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
