"""PX0 训练当前主线共用的最小 checkpoint 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from train_common import MIN_LR, WARMUP_EPOCHS


def lr_scheduler(opt: AdamW, *, epochs: int):
    warmup = WARMUP_EPOCHS
    if warmup >= epochs:
        return CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=MIN_LR)
    warm = LinearLR(opt, start_factor=1e-2, end_factor=1.0, total_iters=warmup)
    cos = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup), eta_min=MIN_LR)
    return SequentialLR(opt, [warm, cos], milestones=[warmup])


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
