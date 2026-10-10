"""Training configuration for PRism QLoRA fine-tuning.

All heavy ML libraries are intentionally NOT imported here: ``import
prism.training.config`` must work on a bare CPU box with only pydantic+pyyaml.

Defaults target QLoRA on meta-llama/Meta-Llama-3-8B-Instruct on a single 24 GB
GPU (e.g. RTX 4090 / L4 / A10G):

* 4-bit NF4 base weights (~4.6 GB) + LoRA adapters on every attention and MLP
  projection (r=16, alpha=32). Trainable params ≈ 40 M (~0.5 % of 8 B).
* Per-device batch 2 × grad-accum 8 → effective batch 16. With gradient
  checkpointing and paged 8-bit AdamW, peak VRAM stays comfortably under 24 GB
  at max_seq_len=4096.
* 3 epochs over ~5 K samples ≈ 940 optimizer steps; cosine schedule, 3 % warmup,
  lr 2e-4 (the QLoRA paper's sweet spot for 7-8 B models).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class LoRAParams(BaseModel):
    """PEFT LoRA adapter hyperparameters."""

    r: int = Field(default=16, ge=1, le=256, description="LoRA rank.")
    alpha: int = Field(default=32, ge=1, description="LoRA scaling factor.")
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Linear layers to adapt. All attention + MLP projections "
        "for Llama 3; this is what the QLoRA paper ablates as best.",
    )
    bias: str = Field(default="none", description="One of none/all/lora_only.")
    task_type: str = Field(default="CAUSAL_LM")

    @field_validator("bias")
    @classmethod
    def _check_bias(cls, v: str) -> str:
        if v not in {"none", "all", "lora_only"}:
            raise ValueError(f"bias must be none/all/lora_only, got {v!r}")
        return v


class QuantParams(BaseModel):
    """bitsandbytes 4-bit quantization settings."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = Field(default="nf4", description="'nf4' or 'fp4'.")
    bnb_4bit_use_double_quant: bool = Field(
        default=True, description="Second quantization of the absmax constants."
    )
    bnb_4bit_compute_dtype: str = Field(
        default="bfloat16", description="Compute dtype for 4-bit matmuls."
    )

    @field_validator("bnb_4bit_quant_type")
    @classmethod
    def _check_quant_type(cls, v: str) -> str:
        if v not in {"nf4", "fp4"}:
            raise ValueError(f"bnb_4bit_quant_type must be nf4/fp4, got {v!r}")
        return v


class TrainConfig(BaseModel):
    """Full configuration for one QLoRA fine-tuning run."""

    # -- model & data ------------------------------------------------------
    model_id: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct")
    train_path: str = Field(description="JSONL with {'messages': [...]} records.")
    val_path: str | None = Field(default=None, description="Optional separate validation JSONL.")
    val_size: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Val fraction if val_path unset."
    )
    max_seq_len: int = Field(default=4096, ge=512, le=8192)

    # -- adapters & quantization -------------------------------------------
    lora: LoRAParams = Field(default_factory=LoRAParams)
    quant: QuantParams = Field(default_factory=QuantParams)

    # -- optimization -------------------------------------------------------
    learning_rate: float = Field(default=2e-4, gt=0.0, lt=1.0)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    per_device_eval_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    num_train_epochs: int = Field(default=3, ge=1)
    max_steps: int = Field(
        default=-1,
        description="If > 0, overrides num_train_epochs (HF semantics).",
    )
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    lr_scheduler_type: str = Field(default="cosine")
    weight_decay: float = Field(default=0.0, ge=0.0)
    optim: str = Field(default="paged_adamw_8bit")
    max_grad_norm: float = Field(default=0.3, gt=0.0)
    gradient_checkpointing: bool = Field(
        default=True, description="Trades compute for VRAM; required at 24 GB."
    )
    bf16: bool = Field(default=True, description="Needs Ampere+ GPU.")
    packing: bool = Field(
        default=False,
        description="Keep False: packing breaks response-only loss masking.",
    )
    group_by_length: bool = Field(default=True)
    dataloader_num_workers: int = Field(
        default=2, ge=0, description="DataLoader workers; 0 on constrained machines."
    )

    # -- checkpointing / eval / early stopping ------------------------------
    output_dir: str = Field(default="outputs/prism-qlora-llama3-8b")
    logging_steps: int = Field(default=10, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    early_stopping_patience: int = Field(
        default=3, ge=1, description="Consecutive evals without val-loss improvement."
    )
    resume_from_checkpoint: str | None = Field(
        default=None,
        description="Checkpoint dir to resume from; auto-detects latest if unset.",
    )
    seed: int = Field(default=42)

    # -- experiment tracking -------------------------------------------------
    mlflow_experiment: str = Field(default="prism-qlora")
    run_name: str | None = Field(default=None)

    @property
    def effective_batch_size(self) -> int:
        """Global batch size across gradient accumulation (single GPU)."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load config from a YAML file."""
        with open(path, encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Config YAML must be a mapping, got {type(raw).__name__}")
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """Write config to YAML (records exact hyperparams of a run)."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(), fh, sort_keys=False)
