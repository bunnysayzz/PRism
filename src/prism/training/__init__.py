"""PRism fine-tuning pipeline.

CPU-safe package root: importing this package (or ``prism.training.config``)
must NOT require torch/transformers/peft/trl/bitsandbytes/datasets. Heavy ML
imports live inside functions in ``data.py``, ``trainer.py`` and ``smoke.py``.
"""

from __future__ import annotations

from prism.training.config import TrainConfig

__all__ = ["TrainConfig"]
