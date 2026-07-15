"""PX0 主线训练共用常量。"""

from __future__ import annotations

import os

TRAIN_SEED = 42
WARMUP_EPOCHS = 1
MIN_LR = 1e-5


def default_num_workers() -> int:
    cpu_count = max(1, os.cpu_count() or 1)
    if cpu_count <= 2:
        return 0
    if os.name == "nt":
        return min(4, max(2, cpu_count // 4))
    return min(8, max(2, cpu_count - 2))
