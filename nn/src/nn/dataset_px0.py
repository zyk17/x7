"""PX0 主线训练数据集。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from nn.px0_record import Px0Sample, expand_chunk_globs, iter_px0_chunk_file


@dataclass(frozen=True)
class Px0DatasetConfig:
    patterns: tuple[str, ...] = ()
    file_list_path: Path | None = None
    shuffle_files: bool = False
    max_files: int = 0
    limit_samples: int = 0


def load_px0_file_list(path: Path | str) -> list[Path]:
    src = Path(path)
    data = json.loads(src.read_text(encoding="utf-8"))
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"px0 file list 缺少非空 files: {src}")
    out = [Path(str(item)).resolve() for item in files]
    missing = [str(p) for p in out if not p.is_file()]
    if missing:
        preview = ", ".join(missing[:4])
        raise FileNotFoundError(f"px0 file list 含缺失文件: {preview}")
    return out


def resolve_px0_files(config: Px0DatasetConfig) -> list[Path]:
    if config.file_list_path is not None:
        files = load_px0_file_list(config.file_list_path)
    else:
        files = expand_chunk_globs(list(config.patterns), max_files=0)
    if config.max_files > 0:
        return files[: int(config.max_files)]
    return files


class Px0ChunkDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(self, config: Px0DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.files = resolve_px0_files(config)
        if not self.files:
            raise FileNotFoundError("no px0 chunk files matched")

    def _ordered_files(self) -> list[Path]:
        files = list(self.files)
        if self.config.shuffle_files:
            random.shuffle(files)
        return files

    @staticmethod
    def _to_item(sample: Px0Sample) -> dict[str, torch.Tensor]:
        return {
            "board": torch.from_numpy(sample.planes),
            "policy": torch.from_numpy(sample.policy),
            "winner_q": torch.from_numpy(sample.winner_q),
            "winner_wdl": torch.from_numpy(sample.winner_wdl),
            "root_wdl": torch.from_numpy(sample.root_wdl),
            "search_q": torch.from_numpy(sample.search_q),
            "search_wdl": torch.from_numpy(sample.search_wdl),
            "search_visits": torch.from_numpy(sample.search_visits),
            "policy_kld": torch.from_numpy(sample.policy_kld),
            "plies_left": torch.from_numpy(sample.plies_left),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        files = self._ordered_files()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
        emitted = 0
        limit = int(self.config.limit_samples)
        for path in files:
            for sample in iter_px0_chunk_file(path):
                yield self._to_item(sample)
                emitted += 1
                if limit > 0 and emitted >= limit:
                    return
