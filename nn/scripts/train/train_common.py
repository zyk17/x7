"""训练脚本共用常量与采样器。"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Iterator

TRAIN_SEED = 42
LABEL_SMOOTHING = 0.08
WARMUP_EPOCHS = 1
MIN_LR = 1e-5


def default_num_workers() -> int:
    cpu_count = max(1, os.cpu_count() or 1)
    if cpu_count <= 2:
        return 0
    if os.name == "nt":
        return min(4, max(2, cpu_count // 4))
    return min(8, max(2, cpu_count - 2))


def default_val_num_workers() -> int:
    train_workers = default_num_workers()
    if train_workers <= 1:
        return 0
    return min(2, train_workers)


def set_requires_grad(module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(enabled)


class GameGroupedBatchSampler:
    """先随机打乱局顺序，再按局内行序串联后切块。"""

    def __init__(
        self,
        batch_size: int,
        *,
        row_group_ids: list[int],
        drop_last: bool = False,
        seed: int = TRAIN_SEED,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        gid_to_idx: dict[int, list[int]] = defaultdict(list)
        for i, gid in enumerate(row_group_ids):
            gid_to_idx[int(gid)].append(i)
        self._groups = list(gid_to_idx.items())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [(gid, list(idxs)) for gid, idxs in self._groups]
        rng.shuffle(groups)
        stream: list[int] = []
        for _, idxs in groups:
            stream.extend(idxs)
        batch_size = self.batch_size
        for i in range(0, len(stream), batch_size):
            chunk = stream[i : i + batch_size]
            if len(chunk) < batch_size and self.drop_last:
                continue
            yield chunk

    def __len__(self) -> int:
        n = sum(len(idxs) for _, idxs in self._groups)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
