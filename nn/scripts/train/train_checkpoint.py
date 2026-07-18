"""PX0 训练当前主线共用的最小 checkpoint 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer


def learning_rate_at_step(
    step: int,
    *,
    values: tuple[float, ...],
    boundaries: tuple[int, ...],
    warmup_steps: int,
) -> float:
    """Return the pxzero-style phase-local piecewise learning rate.

    Reference: pxzero-training `tf/configs/example.yaml:16-27` and
    `tf/tfprocess.py` training-step learning-rate selection.
    """
    if step < 1:
        raise ValueError("step must be positive")
    index = sum(step > boundary for boundary in boundaries)
    lr = values[index]
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr


def set_optimizer_learning_rate(opt: Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
