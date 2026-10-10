#!/usr/bin/env python3
"""PRism fine-tuning entry point.

Usage:
    python scripts/train.py --smoke
        CPU smoke test: tiny random model, 5 steps, proves the pipeline
        end-to-end. No GPU, no HF token needed.

    python scripts/train.py --config configs/train_qlora.yaml
        Full QLoRA run on a GPU machine. Needs the 'training' extra and
        HF_TOKEN (gated Llama 3 weights).

    python scripts/train.py --config configs/train_qlora.yaml --resume \
        --resume-from outputs/prism-qlora-llama3-8b/checkpoint-500
        Resume from an explicit checkpoint (auto-detects the latest
        checkpoint in output_dir when --resume-from is omitted).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRism QLoRA fine-tuning: --smoke for CPU, else full GPU run."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "train_qlora.yaml"),
        help="Path to training YAML config (default: configs/train_qlora.yaml).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the CPU smoke test instead of full training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training (auto-detects latest checkpoint unless --resume-from is given).",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Explicit checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the config's output_dir.",
    )
    return parser.parse_args(argv)


def _run_smoke() -> int:
    from prism.training.smoke import main as smoke_main

    return smoke_main()


def _run_training(args: argparse.Namespace) -> int:
    from prism.logging import get_logger, setup_logging
    from prism.training.config import TrainConfig
    from prism.training.trainer import train

    setup_logging("INFO")
    log = get_logger("prism.train")

    cfg = TrainConfig.from_yaml(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.resume_from:
        cfg.resume_from_checkpoint = args.resume_from
    elif args.resume:
        cfg.resume_from_checkpoint = None  # auto-detect latest in output_dir

    if not os.environ.get("HF_TOKEN"):
        log.warning(
            "HF_TOKEN is not set; downloading gated Llama 3 weights will fail. "
            "Create a token at https://huggingface.co/settings/tokens (read "
            "scope) and accept the Llama 3 license at "
            "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct."
        )

    log.info(
        "loaded training config",
        config=args.config,
        model=cfg.model_id,
        output_dir=cfg.output_dir,
        effective_batch=cfg.effective_batch_size,
    )
    adapter_dir = train(cfg)
    print(f"Training complete. Adapter: {adapter_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        return _run_smoke()
    return _run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
