"""PX0 训练当前主线共用的最小 checkpoint 工具。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer


def learning_rate_at_step(
    step: int,
    *,
    total_steps: int,
    lr: float,
    warmup_steps: int,
    min_lr_scale: float,
) -> float:
    """Linear warmup followed by cosine decay for one uninterrupted training run."""
    if step < 1 or total_steps < 1 or lr <= 0.0 or not 0.0 <= min_lr_scale <= 1.0:
        raise ValueError("invalid cosine learning-rate schedule")
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step / warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    scale = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * scale


def set_optimizer_learning_rate(opt: Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
