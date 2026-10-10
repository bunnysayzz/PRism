"""SFT dataset preparation for PRism QLoRA fine-tuning.

Record schema (JSONL, one record per line)::

    {"messages": [
        {"role": "system", "content": "..."},      # optional; PRism review
        {"role": "user", "content": "..."},        # prompt is inserted when missing
        {"role": "assistant", "content": "{...}"}  # strict-JSON findings, REQUIRED last
    ]}

* The ``user`` message carries the per-file review prompt (diff hunks +
  context, see ``prism.review.prompts.build_user_prompt``).
* The ``assistant`` message is the target: strict JSON matching
  ``prism.review.schemas.ReviewResult`` (``{"findings": [...]}``).
* Records are formatted with the model's chat template; loss is computed on
  the assistant response only (prompt tokens get label ``-100``).

Heavy imports (``datasets``) live inside functions so this module imports
cleanly without the training extras installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from prism.logging import get_logger
from prism.review.prompts import SYSTEM_PROMPT
from prism.training.config import TrainConfig

if TYPE_CHECKING:
    from datasets import Dataset

log = get_logger(__name__)

Message = dict[str, str]
VALID_ROLES = ("system", "user", "assistant")


class ChatTokenizer(Protocol):
    """Minimal tokenizer surface needed for SFT formatting."""

    def apply_chat_template(
        self,
        messages: list[Message],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str: ...

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of raw record dicts."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: record must be a JSON object")
            records.append(obj)
    return records


def validate_record(record: dict[str, Any], index: int = 0) -> list[Message]:
    """
    Validate one SFT record and return its message list.

    Raises ValueError describing the first problem found.
    """
    messages = record.get("messages")
    where = f"record {index}"
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{where}: 'messages' must be a non-empty list")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"{where}: message {i} must be an object")
        role = msg.get("role")
        content = msg.get("content")
        if role not in VALID_ROLES:
            raise ValueError(f"{where}: message {i} has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{where}: message {i} has empty content")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{where}: last message must be the assistant response")
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]


def ensure_system_prompt(messages: list[Message]) -> list[Message]:
    """Prepend the PRism review system prompt when the record has none."""
    if messages and messages[0]["role"] == "system":
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


def format_example(
    messages: list[Message],
    tokenizer: ChatTokenizer,
    max_seq_len: int,
    *,
    min_response_tokens: int = 16,
) -> dict[str, list[int]]:
    """
    Format one record into input_ids/attention_mask/labels.

    The chat template is applied twice — once for the prompt (everything
    before the assistant message, with the generation prompt appended) and
    once for the full conversation. Prompt token positions get label -100 so
    the loss is computed on the assistant response only.

    Over-long sequences are truncated from the LEFT of the prompt portion so
    the response (the training signal) is preserved whenever possible.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("SFT record must end with exactly one assistant message")

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenized prompt is not a prefix of the full conversation")
    response_ids = full_ids[len(prompt_ids) :]
    if not response_ids:
        raise ValueError("assistant response tokenized to zero tokens")

    if len(full_ids) > max_seq_len:
        overflow = len(full_ids) - max_seq_len
        if overflow < len(prompt_ids):
            # Drop tokens from the left of the prompt; keep the whole response.
            prompt_ids = prompt_ids[overflow:]
            full_ids = full_ids[overflow:]
        else:
            # Pathological: prompt alone exceeds budget. Keep the prompt tail
            # plus a head slice of the response so some signal survives.
            keep_resp = min(len(response_ids), max(min_response_tokens, max_seq_len // 8))
            prompt_keep = max(max_seq_len - keep_resp, 0)
            prompt_ids = prompt_ids[-prompt_keep:] if prompt_keep else []
            full_ids = prompt_ids + response_ids[:keep_resp]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    assert len(labels) == len(full_ids)
    assert any(label != -100 for label in labels), "no trainable tokens left"
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def format_records(
    records: list[dict[str, Any]],
    tokenizer: ChatTokenizer,
    max_seq_len: int,
) -> list[dict[str, list[int]]]:
    """
    Validate + format raw JSONL records, dropping invalid ones.

    Invalid records are skipped with a warning (a few malformed lines must
    not sink a 5 K-sample run); the drop count is logged.
    """
    formatted: list[dict[str, list[int]]] = []
    dropped = 0
    for i, record in enumerate(records):
        try:
            messages = ensure_system_prompt(validate_record(record, i))
            formatted.append(format_example(messages, tokenizer, max_seq_len))
        except ValueError as exc:
            dropped += 1
            log.warning("dropping record", index=i, reason=str(exc))
    log.info("formatted records", kept=len(formatted), dropped=dropped)
    return formatted


def load_hf_dataset(path: str | Path) -> Dataset:
    """Load a JSONL file as a HuggingFace Dataset (streaming off)."""
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(path), split="train")
    return ds


def tokenize_hf_dataset(dataset: Dataset, tokenizer: ChatTokenizer, max_seq_len: int) -> Dataset:
    """Validate + format every record of a HF Dataset, dropping invalid rows."""
    from datasets import Dataset as HFDataset

    if "messages" not in dataset.column_names:
        raise ValueError("dataset must have a 'messages' column")
    raw_records: list[dict[str, Any]] = [{"messages": row["messages"]} for row in dataset]
    formatted = format_records(raw_records, tokenizer, max_seq_len)
    if not formatted:
        raise ValueError("no valid records left after formatting")
    return HFDataset.from_list(formatted)


def build_train_eval_datasets(
    cfg: TrainConfig, tokenizer: ChatTokenizer
) -> tuple[Dataset, Dataset]:
    """
    Build tokenized train/eval datasets per the config.

    Uses ``val_path`` when set, otherwise splits ``train_path`` by
    ``val_size`` with ``seed``.
    """
    train_raw = load_hf_dataset(cfg.train_path)
    if cfg.val_path:
        val_raw = load_hf_dataset(cfg.val_path)
        log.info("using separate validation file", val_path=cfg.val_path)
    else:
        split = train_raw.train_test_split(test_size=cfg.val_size, seed=cfg.seed)
        train_raw, val_raw = split["train"], split["test"]
        log.info(
            "split train/val",
            train=len(train_raw),
            val=len(val_raw),
            val_size=cfg.val_size,
            seed=cfg.seed,
        )
    train_ds = tokenize_hf_dataset(train_raw, tokenizer, cfg.max_seq_len)
    val_ds = tokenize_hf_dataset(val_raw, tokenizer, cfg.max_seq_len)
    log.info("datasets ready", train_rows=len(train_ds), val_rows=len(val_ds))
    return train_ds, val_ds
