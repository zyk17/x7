"""PX0 主线训练数据集。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TypeVar

import torch
from torch.utils.data import IterableDataset, get_worker_info

from nn.px0_record import Px0Sample, expand_chunk_globs, iter_px0_chunk_file


@dataclass(frozen=True)
class Px0DatasetConfig:
    patterns: tuple[str, ...] = ()
    file_list_path: Path | None = None
    sample_list_path: Path | None = None
    shuffle_files: bool = False
    shuffle_size: int = 0
    max_files: int = 0
    limit_samples: int = 0
    verify_files: bool = True


def load_px0_file_list(path: Path | str, *, verify_files: bool = True) -> list[Path]:
    src = Path(path)
    data = json.loads(src.read_text(encoding="utf-8"))
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"px0 file list 缺少非空 files: {src}")
    out = [Path(str(item)) for item in files]
    if verify_files:
        out = [path.resolve() for path in out]
        missing = [str(path) for path in out if not path.is_file()]
        if missing:
            preview = ", ".join(missing[:4])
            raise FileNotFoundError(f"px0 file list 含缺失文件: {preview}")
    return out


def resolve_px0_files(config: Px0DatasetConfig) -> list[Path]:
    if config.file_list_path is not None:
        files = load_px0_file_list(config.file_list_path, verify_files=config.verify_files)
    else:
        files = expand_chunk_globs(list(config.patterns), max_files=0)
    if config.max_files > 0:
        return files[: int(config.max_files)]
    return files


def load_px0_sample_list(path: Path | str) -> dict[Path, list[int]]:
    src = Path(path)
    data = json.loads(src.read_text(encoding="utf-8"))
    entries = data.get("samples")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"px0 sample list 缺少非空 samples: {src}")
    grouped: dict[Path, list[int]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"px0 sample list 包含非法 sample: {src}")
        path = Path(str(entry.get("file", ""))).resolve()
        record_index = entry.get("record_index")
        if not path.is_file() or not isinstance(record_index, int) or record_index < 0:
            raise ValueError(f"px0 sample list 包含非法引用: {src}")
        grouped.setdefault(path, []).append(record_index)
    return {path: sorted(indices) for path, indices in sorted(grouped.items())}


T = TypeVar("T")


def shuffle_stream(items: Iterator[T], *, size: int) -> Iterator[T]:
    """Bounded replacement shuffle for sequential chunk records.

    Reference: pxzero-training `tf/chunkparser.py:480-500` and
    `tf/shufflebuffer.py:56-74`. The buffer is deliberately bounded because
    decoded PyTorch samples are much larger than pxzero's packed records.
    """
    if size <= 1:
        yield from items
        return
    buffer: list[T] = []
    for item in items:
        if len(buffer) < size:
            buffer.append(item)
            continue
        index = random.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    while buffer:
        yield buffer.pop(random.randrange(len(buffer)))


class Px0ChunkDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(self, config: Px0DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.sample_indices = load_px0_sample_list(config.sample_list_path) if config.sample_list_path else None
        self.files = list(self.sample_indices) if self.sample_indices is not None else resolve_px0_files(config)
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
            "winner_wdl": torch.from_numpy(sample.winner_wdl),
            "root_wdl": torch.from_numpy(sample.root_wdl),
            "plies_left": torch.from_numpy(sample.plies_left),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        files = self._ordered_files()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]

        def samples() -> Iterator[Px0Sample]:
            for path in files:
                wanted = self.sample_indices.get(path) if self.sample_indices is not None else None
                wanted_pos = 0
                for record_index, sample in enumerate(iter_px0_chunk_file(path)):
                    if wanted is not None:
                        while wanted_pos < len(wanted) and wanted[wanted_pos] < record_index:
                            wanted_pos += 1
                        if wanted_pos >= len(wanted):
                            break
                        if wanted[wanted_pos] != record_index:
                            continue
                        wanted_pos += 1
                    yield sample

        # DataLoader workers own disjoint file streams. Divide the configured
        # total buffer across them to bound aggregate host-memory use.
        worker_count = 1 if worker is None else worker.num_workers
        buffer_size = (int(self.config.shuffle_size) + worker_count - 1) // worker_count
        emitted = 0
        limit = int(self.config.limit_samples)
        for sample in shuffle_stream(samples(), size=buffer_size):
            yield self._to_item(sample)
            emitted += 1
            if limit > 0 and emitted >= limit:
                return
