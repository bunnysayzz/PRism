"""QLoRA training orchestration for PRism.

Full SFT pipeline for meta-llama/Meta-Llama-3-8B-Instruct on a single 24 GB
GPU: 4-bit NF4 quantization (bitsandbytes) + LoRA adapters (peft) + TRL
SFTTrainer with MLflow logging, checkpointing, early stopping and resume.

All heavy imports (torch, transformers, peft, trl, bitsandbytes, datasets,
mlflow) happen INSIDE functions. Importing this module only needs the
``prism.training.config`` / ``prism.training.data`` dependencies
(pydantic + pyyaml + structlog).

Pinned, mutually-compatible versions (see docs/TRAINING.md)::

    transformers==4.44.2  peft==0.12.0  trl==0.9.6  bitsandbytes==0.43.3
    accelerate==0.33.0    datasets==2.21.0  mlflow==2.16.2
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from prism.logging import get_logger
from prism.training.config import TrainConfig
from prism.training.data import build_train_eval_datasets

if TYPE_CHECKING:
    from datasets import Dataset
    from peft.peft_model import PeftModel
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizerBase,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
    from trl import SFTTrainer

log = get_logger(__name__)

MISSING_EXTRAS_MSG = (
    "prism.training.trainer needs the 'training' extra "
    "(torch, transformers, peft, trl, bitsandbytes, datasets, accelerate, mlflow). "
    "See docs/TRAINING.md for the GPU-machine install command."
)


def build_tokenizer(cfg: TrainConfig) -> PreTrainedTokenizerBase:
    """Load the model tokenizer; fall back to EOS as pad token."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(MISSING_EXTRAS_MSG) from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        log.info("no pad token; using eos as pad", eos=tokenizer.eos_token)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = cfg.max_seq_len
    return tokenizer


def build_quant_config(cfg: TrainConfig) -> Any:
    """bitsandbytes 4-bit quantization config (NF4 double-quant)."""
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(MISSING_EXTRAS_MSG) from exc

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtypes.get(cfg.quant.bnb_4bit_compute_dtype)
    if compute_dtype is None:
        raise ValueError(f"unsupported compute dtype: {cfg.quant.bnb_4bit_compute_dtype!r}")
    return BitsAndBytesConfig(
        load_in_4bit=cfg.quant.load_in_4bit,
        bnb_4bit_quant_type=cfg.quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def apply_lora(model: PreTrainedModel, cfg: TrainConfig) -> PeftModel:
    """Wrap a base causal-LM in LoRA adapters per the config."""
    try:
        from peft.mapping import get_peft_model
        from peft.tuners.lora.config import LoraConfig
        from peft.utils.peft_types import TaskType
    except ImportError as exc:
        raise ImportError(MISSING_EXTRAS_MSG) from exc

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        # Validated by LoRAParams to be one of the three literals.
        bias=cast(Literal["none", "all", "lora_only"], cfg.lora.bias),
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def build_model(cfg: TrainConfig, tokenizer: PreTrainedTokenizerBase) -> PeftModel:
    """
    Load the base model in 4-bit and attach LoRA adapters.

    Requires CUDA (bitsandbytes 4-bit kernels); see smoke.py for the CPU path.
    """
    try:
        import torch
        from peft.utils.other import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(MISSING_EXTRAS_MSG) from exc

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=build_quant_config(cfg),
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=False,
    )
    # Gradient checkpointing is incompatible with use_cache.
    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # prepare_model_for_kbit_training is untyped in peft: call via Any.
    model = cast(Any, prepare_model_for_kbit_training)(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )
    peft_model = apply_lora(model, cfg)
    if cfg.gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()
    return peft_model


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, name))
        else:
            flat[name] = value
    return flat


def _system_info() -> dict[str, str]:
    """GPU + library versions, logged to MLflow as run params."""
    import importlib.metadata as metadata

    try:
        import torch
    except ImportError as exc:
        raise ImportError(MISSING_EXTRAS_MSG) from exc

    info: dict[str, str] = {
        "sys.torch": torch.__version__,
        "sys.cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["sys.gpu_name"] = torch.cuda.get_device_name(0)
        info["sys.gpu_mem_gb"] = f"{props.total_memory / 1e9:.1f}"
    for pkg in ("transformers", "peft", "trl", "bitsandbytes", "datasets", "accelerate"):
        try:
            info[f"sys.{pkg}"] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info[f"sys.{pkg}"] = "missing"
    return info


def _mlflow_params_callback(cfg: TrainConfig) -> TrainerCallback:
    """
    Log hyperparams + system info to the active MLflow run at train start.

    (Metrics go through transformers' built-in MLflowCallback via
    ``report_to=["mlflow"]``; the experiment name is set from
    ``MLFLOW_EXPERIMENT_NAME``, which :func:`train` seeds from the config.)
    """

    def _make() -> TrainerCallback:
        # Only the base class needs a runtime import; the method annotations
        # below resolve to the TYPE_CHECKING imports (PEP 563 strings).
        from transformers.trainer_callback import TrainerCallback

        # transformers ships no py.typed, so TrainerCallback is Any to mypy.
        class _MlflowParamsCallback(TrainerCallback):  # type: ignore[misc]
            def on_train_begin(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs: Any,
            ) -> TrainerControl:
                try:
                    import mlflow
                except ImportError as exc:
                    raise ImportError(MISSING_EXTRAS_MSG) from exc
                mlflow.log_params(_flatten_dict(cfg.model_dump()))
                mlflow.log_params(_system_info())
                log.info("logged params + system info to mlflow")
                return control

        return _MlflowParamsCallback()

    return _make()


def build_training_args(cfg: TrainConfig, *, log_to_mlflow: bool = True) -> TrainingArguments:
    """Assemble HF TrainingArguments from the config (OOM-safe defaults)."""
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise ImportError(MISSING_EXTRAS_MSG) from exc

    return TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else None,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        max_grad_norm=cfg.max_grad_norm,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=cfg.bf16,
        tf32=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=["mlflow"] if log_to_mlflow else [],
        run_name=cfg.run_name,
        # Our datasets are pre-tokenized (input_ids/labels/attention_mask);
        # do NOT let Trainer drop "unused" columns.
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        group_by_length=cfg.group_by_length,
        ddp_find_unused_parameters=False,
    )


def build_trainer(
    cfg: TrainConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_ds: Dataset,
    eval_ds: Dataset,
    *,
    log_to_mlflow: bool = True,
) -> SFTTrainer:
    """Build the TRL SFTTrainer with MLflow + early-stopping callbacks."""
    try:
        from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
        from trl import SFTTrainer
    except ImportError as exc:
        raise ImportError(MISSING_EXTRAS_MSG) from exc

    args = build_training_args(cfg, log_to_mlflow=log_to_mlflow)
    callbacks: list[TrainerCallback] = []
    if log_to_mlflow:
        callbacks.append(_mlflow_params_callback(cfg))
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))
    # Our datasets are pre-tokenized with response-only labels (-100 on the
    # prompt). TRL's default DataCollatorForLanguageModeling does NOT pad a
    # pre-existing "labels" column, so pass DataCollatorForSeq2Seq explicitly:
    # it pads inputs with the pad token and labels with -100 (ignored by the
    # loss), preserving the response-only masking.
    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)
    # train_ds/eval_ds are pre-tokenized (input_ids/labels); SFTTrainer skips
    # its own formatting when the "input_ids" column is present.
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    return trainer


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    """Return the newest HF checkpoint dir under output_dir, if any."""
    candidates = [
        p for p in Path(output_dir).glob("checkpoint-*") if (p / "trainer_state.json").exists()
    ]
    if not candidates:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except ValueError:
            return -1

    latest = max(candidates, key=_step)
    return str(latest)


def train(cfg: TrainConfig, *, log_to_mlflow: bool = True) -> str:
    """
    Run the full QLoRA fine-tuning job. Returns the final adapter directory.

    Requires a CUDA GPU (bitsandbytes 4-bit). Saves the LoRA adapter,
    tokenizer and the exact training config under ``<output_dir>/final-adapter``.
    """
    try:
        import torch
        from transformers import set_seed
    except ImportError as exc:
        raise ImportError(MISSING_EXTRAS_MSG) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. QLoRA training needs a GPU; "
            "use scripts/train.py --smoke for the CPU smoke test."
        )
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", cfg.mlflow_experiment)
    set_seed(cfg.seed)

    log.info(
        "starting QLoRA training",
        model=cfg.model_id,
        output_dir=cfg.output_dir,
        effective_batch=cfg.effective_batch_size,
    )
    tokenizer = build_tokenizer(cfg)
    model = build_model(cfg, tokenizer)
    train_ds, eval_ds = build_train_eval_datasets(cfg, tokenizer)
    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds, log_to_mlflow=log_to_mlflow)

    resume = cfg.resume_from_checkpoint or find_latest_checkpoint(cfg.output_dir)
    if resume:
        log.info("resuming from checkpoint", checkpoint=resume)
    trainer.train(resume_from_checkpoint=resume)

    adapter_dir = str(Path(cfg.output_dir) / "final-adapter")
    trainer.save_model(adapter_dir)  # PeftModel -> saves adapter weights
    tokenizer.save_pretrained(adapter_dir)
    cfg.to_yaml(Path(adapter_dir) / "train_config.yaml")
    log.info("training complete", adapter_dir=adapter_dir)
    return adapter_dir
