"""Tests for the PRism fine-tuning pipeline (src/prism/training/).

Fast tests need only pydantic/pyyaml/pytest: the training modules defer all
torch/transformers/peft/trl imports into functions. The end-to-end CPU smoke
test is marked slow and runs only with PRISM_RUN_SMOKE=1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from prism.training.config import LoRAParams, QuantParams, TrainConfig
from prism.training.data import (
    ensure_system_prompt,
    format_example,
    format_records,
    read_jsonl,
    validate_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fake tokenizer (no torch / transformers needed)
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Word-level tokenizer implementing the ChatTokenizer protocol."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self.pad_token_id: int | None = 0

    def _ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 1  # 0 reserved for pad
            ids.append(self._vocab[word])
        return ids

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = [f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>\n")
        return "".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._ids(text)


def _sample_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": "Review the diff in auth.py"},
        {"role": "assistant", "content": '{"findings": []}'},
    ]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = TrainConfig(train_path="data/sft/train.jsonl")
    assert cfg.model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert cfg.effective_batch_size == 16  # 2 per-device x 8 accum
    assert cfg.lora.r == 16
    assert cfg.lora.alpha == 32
    assert cfg.quant.bnb_4bit_quant_type == "nf4"
    assert cfg.quant.bnb_4bit_use_double_quant is True
    assert cfg.packing is False
    assert cfg.resume_from_checkpoint is None


def test_config_lora_targets_cover_llama3() -> None:
    cfg = TrainConfig(train_path="x.jsonl")
    for mod in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert mod in cfg.lora.target_modules


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(train_path="x.jsonl", learning_rate=-0.1)
    with pytest.raises(ValidationError):
        TrainConfig(train_path="x.jsonl", per_device_train_batch_size=0)
    with pytest.raises(ValidationError):
        LoRAParams(bias="sometimes")
    with pytest.raises(ValidationError):
        QuantParams(bnb_4bit_quant_type="int8")


def test_config_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = TrainConfig(train_path="x.jsonl", learning_rate=1e-4, run_name="test-run")
    yaml_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(yaml_path)
    reloaded = TrainConfig.from_yaml(yaml_path)
    assert reloaded == cfg


def test_production_yaml_loads() -> None:
    cfg = TrainConfig.from_yaml(REPO_ROOT / "configs" / "train_qlora.yaml")
    assert cfg.model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert cfg.max_seq_len == 4096
    assert cfg.effective_batch_size == 16


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        TrainConfig.from_yaml(bad)


# ---------------------------------------------------------------------------
# Data tests (fake tokenizer, no torch)
# ---------------------------------------------------------------------------


def test_validate_record_ok() -> None:
    messages = validate_record({"messages": _sample_messages()}, 0)
    assert messages[-1]["role"] == "assistant"


def test_validate_record_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        validate_record({"messages": []}, 0)
    with pytest.raises(ValueError, match="invalid role"):
        validate_record({"messages": [{"role": "bot", "content": "hi"}]}, 0)
    with pytest.raises(ValueError, match="empty content"):
        validate_record({"messages": [{"role": "user", "content": "  "}]}, 0)
    with pytest.raises(ValueError, match="last message must be the assistant"):
        validate_record({"messages": [{"role": "user", "content": "hi"}]}, 0)


def test_ensure_system_prompt_prepends_when_missing() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = ensure_system_prompt(messages)
    assert out[0]["role"] == "system"
    assert len(out) == 3


def test_ensure_system_prompt_keeps_existing() -> None:
    messages = _sample_messages()
    assert ensure_system_prompt(messages) == messages


def test_format_example_masks_prompt_tokens() -> None:
    tok = FakeTokenizer()
    messages = _sample_messages()
    out = format_example(messages, tok, max_seq_len=512)

    prompt_ids = tok.encode(tok.apply_chat_template(messages[:-1], add_generation_prompt=True))
    n_prompt = len(prompt_ids)
    assert out["input_ids"][:n_prompt] == prompt_ids
    assert out["labels"][:n_prompt] == [-100] * n_prompt
    # Response tokens are trained: labels equal input_ids there, at least one.
    assert out["labels"][n_prompt:] == out["input_ids"][n_prompt:]
    assert any(label != -100 for label in out["labels"])
    assert out["attention_mask"] == [1] * len(out["input_ids"])
    assert len(out["labels"]) == len(out["input_ids"])


def test_format_example_rejects_missing_assistant() -> None:
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="assistant message"):
        format_example([{"role": "user", "content": "hi"}], tok, 512)


def test_format_example_truncates_long_prompt() -> None:
    tok = FakeTokenizer()
    messages = [
        {"role": "user", "content": "word " * 200},
        {"role": "assistant", "content": "done"},
    ]
    out = format_example(messages, tok, max_seq_len=32)
    assert len(out["input_ids"]) == 32
    assert len(out["labels"]) == 32
    # Response survived truncation: some labels are trainable.
    assert any(label != -100 for label in out["labels"])


def test_format_records_drops_invalid(tmp_path: Path) -> None:
    tok = FakeTokenizer()
    records: list[dict[str, Any]] = [
        {"messages": _sample_messages()},
        {"messages": [{"role": "user", "content": "no assistant here"}]},
        {"not_messages": []},
    ]
    out = format_records(records, tok, max_seq_len=512)
    assert len(out) == 1


def test_read_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "a"}]}\n'
        "\n"
        '{"messages": [{"role": "user", "content": "b"}]}\n',
        encoding="utf-8",
    )
    records = read_jsonl(path)
    assert len(records) == 2  # blank line skipped


def test_read_jsonl_rejects_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        read_jsonl(path)


# ---------------------------------------------------------------------------
# Trainer module: importable without heavy deps; pure helpers testable
# ---------------------------------------------------------------------------


def test_training_modules_import_without_heavy_deps() -> None:
    import prism.training  # noqa: F401
    import prism.training.config  # noqa: F401
    import prism.training.data  # noqa: F401
    import prism.training.smoke  # noqa: F401
    import prism.training.trainer  # noqa: F401


def test_find_latest_checkpoint(tmp_path: Path) -> None:
    from prism.training.trainer import find_latest_checkpoint

    assert find_latest_checkpoint(tmp_path) is None
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-100" / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-200").mkdir()
    (tmp_path / "checkpoint-200" / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-zzz").mkdir()  # no trainer_state.json: ignored
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None and latest.endswith("checkpoint-200")


def test_flatten_dict() -> None:
    from prism.training.trainer import _flatten_dict

    flat = _flatten_dict({"a": {"b": 1}, "c": [1, 2]})
    assert flat == {"a.b": 1, "c": [1, 2]}


def test_build_training_args_assembly() -> None:
    transformers = pytest.importorskip("transformers")
    from prism.training.trainer import build_training_args

    cfg = TrainConfig(
        train_path="x.jsonl",
        output_dir="outputs/test",
        bf16=False,
        gradient_checkpointing=False,
        max_steps=5,
    )
    args = build_training_args(cfg, log_to_mlflow=False)
    assert args.eval_strategy == transformers.trainer_utils.EvaluationStrategy.STEPS
    assert args.remove_unused_columns is False
    assert args.report_to == []
    assert args.max_steps == 5
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"


# ---------------------------------------------------------------------------
# Slow: full CPU smoke test (tiny model, real SFTTrainer, 5 steps)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_end_to_end() -> None:
    if os.environ.get("PRISM_RUN_SMOKE") != "1":
        pytest.skip("set PRISM_RUN_SMOKE=1 to run the CPU smoke test")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    from prism.training.smoke import main

    assert main() == 0
