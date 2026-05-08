"""JSONL → PyTorch Dataset（棋盘平面 + 合法 mask + 标签 + 样本权重）。"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import board as xb
from augment_mirror import mirror_fen, mirror_pyffish_uci, mirror_uci_prefix
from nn.fen_tensor import fen_to_planes
from nn.jsonl_index import (
    PLIES,
    PGN_SRC_VOCAB_JSON,
    ROW_END,
    ROW_START,
    SRC_IDS,
    TARGETS,
    WEIGHTS,
    index_dir_is_complete,
)

# 训练集固定策略（不在 CLI 暴露组合）
_TRAIN_MIRROR_PROB = 0.5


def _policy_row_to_tensors(
    move_to_idx: dict[str, int],
    root: str,
    prefix: list[str],
    fen: str,
    human: str,
    *,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab_size = len(move_to_idx)
    board = fen_to_planes(fen)
    legal = xb.legal_moves_uci(root, prefix)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for u in legal:
        j = move_to_idx.get(u)
        if j is not None:
            mask[j] = True
    if not mask.any():
        raise RuntimeError("当前局面无词表内合法着（需扩充词表或检查 FEN）")

    ti = move_to_idx[human]
    if not mask[ti]:
        raise RuntimeError(f"人类着法不在当前合法集中或不在词表: {human}")

    return (
        board,
        mask,
        torch.tensor(ti, dtype=torch.long),
        torch.tensor(weight, dtype=torch.float32),
    )


class PolicyJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path | str,
        move_to_idx: dict[str, int],
        *,
        for_training: bool = False,
        skip_unknown_moves: bool = True,
        with_row_meta: bool = False,
    ) -> None:
        self.move_to_idx = move_to_idx
        self.vocab_size = len(move_to_idx)
        self.for_training = for_training
        self.with_row_meta = with_row_meta
        self.aug_mirror_p = _TRAIN_MIRROR_PROB if for_training else 0.0
        self.rows: list[dict] = []
        path = Path(jsonl_path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                mv = row.get("human_move_pyffish")
                if not mv or mv not in move_to_idx:
                    if skip_unknown_moves:
                        continue
                    raise KeyError(f"词表无此着法: {mv}")
                self.rows.append(row)

        self.position_weight_by_fen: dict[str, float] | None = None
        if for_training:
            cnt = Counter(r["fen"] for r in self.rows if r.get("fen"))
            self.position_weight_by_fen = {
                f: 1.0 / math.sqrt(n) for f, n in cnt.items() if n >= 1
            }

        self.pgn_source_vocab: list[str] = []
        self._pgn_source_to_id: dict[str, int] = {}
        if with_row_meta:
            srcs = sorted({str(r.get("pgn_source", "") or "") for r in self.rows})
            self.pgn_source_vocab = srcs
            self._pgn_source_to_id = {s: i for i, s in enumerate(srcs)}

    def __len__(self) -> int:
        return len(self.rows)

    def _row_to_tensors(
        self,
        root: str,
        prefix: list[str],
        fen: str,
        human: str,
        *,
        weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _policy_row_to_tensors(
            self.move_to_idx, root, prefix, fen, human, weight=weight
        )

    def __getitem__(
        self, i: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        row = self.rows[i]
        root0 = row["root_fen"]
        prefix0 = list(row["uci_prefix"])
        fen0 = row["fen"]
        human0 = row["human_move_pyffish"]

        w0 = 1.0
        if self.position_weight_by_fen is not None:
            w0 = self.position_weight_by_fen.get(fen0, 1.0)

        use_mirror = self.aug_mirror_p > 0.0 and random.random() < self.aug_mirror_p
        if use_mirror:
            try:
                mh = mirror_pyffish_uci(human0)
                if mh not in self.move_to_idx:
                    use_mirror = False
                else:
                    root_m = mirror_fen(root0)
                    prefix_m = mirror_uci_prefix(prefix0)
                    fen_m = mirror_fen(fen0)
                    out = self._row_to_tensors(
                        root_m, prefix_m, fen_m, mh, weight=w0
                    )
                    if self.with_row_meta:
                        ply = int(row.get("ply", 0) or 0)
                        sid = self._pgn_source_to_id.get(
                            str(row.get("pgn_source", "") or ""), 0
                        )
                        return (
                            *out,
                            torch.tensor(ply, dtype=torch.long),
                            torch.tensor(sid, dtype=torch.long),
                        )
                    return out
            except (ValueError, RuntimeError):
                use_mirror = False

        out = self._row_to_tensors(root0, prefix0, fen0, human0, weight=w0)
        if self.with_row_meta:
            ply = int(row.get("ply", 0) or 0)
            sid = self._pgn_source_to_id.get(
                str(row.get("pgn_source", "") or ""), 0
            )
            return (
                *out,
                torch.tensor(ply, dtype=torch.long),
                torch.tensor(sid, dtype=torch.long),
            )
        return out


class PolicyJsonlMmapDataset(Dataset):
    """大 JSONL：按行字节偏移 mmap 读原文；标签/权重等用 npy mmap。

    不在 ``__init__`` 里打开 memmap，避免 Windows DataLoader ``spawn`` 把整文件 pickle 进 worker。
    ``__getstate__`` 会丢弃已打开的句柄；各进程在首次 ``__getitem__`` / ``__len__`` 时再 ``mmap``。
    """

    def __init__(
        self,
        jsonl_path: Path | str,
        index_dir: Path | str,
        move_to_idx: dict[str, int],
        *,
        for_training: bool = False,
        with_row_meta: bool = False,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.index_dir = Path(index_dir)
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(self.jsonl_path)
        if not index_dir_is_complete(self.index_dir):
            raise FileNotFoundError(
                f"索引目录不完整: {self.index_dir}（需先运行 build_jsonl_index）"
            )

        self.move_to_idx = move_to_idx
        self.vocab_size = len(move_to_idx)
        self.for_training = for_training
        self.with_row_meta = with_row_meta
        self.aug_mirror_p = _TRAIN_MIRROR_PROB if for_training else 0.0

        rs0 = np.load(self.index_dir / ROW_START, mmap_mode="r")
        n = int(rs0.shape[0])
        del rs0
        self._n = n

        raw_vocab = json.loads(
            (self.index_dir / PGN_SRC_VOCAB_JSON).read_text(encoding="utf-8")
        )
        self.pgn_source_vocab = [str(x) for x in raw_vocab]

        self._mm: dict[str, np.memmap | np.ndarray] | None = None

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        d["_mm"] = None
        return d

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._mm = None

    def _ensure_mmaps(self) -> None:
        if self._mm is not None:
            return
        sz = int(self.jsonl_path.stat().st_size)
        j = np.memmap(self.jsonl_path, dtype=np.uint8, mode="r", shape=(sz,))
        rs = np.load(self.index_dir / ROW_START, mmap_mode="r")
        re = np.load(self.index_dir / ROW_END, mmap_mode="r")
        w = np.load(self.index_dir / WEIGHTS, mmap_mode="r")
        t = np.load(self.index_dir / TARGETS, mmap_mode="r")
        p = np.load(self.index_dir / PLIES, mmap_mode="r")
        s = np.load(self.index_dir / SRC_IDS, mmap_mode="r")
        n = self._n
        for name, arr in (("row_end", re), ("weights", w), ("targets", t), ("plies", p), ("src_ids", s)):
            if len(arr) != n:
                raise ValueError(f"索引长度不一致: {name}={len(arr)} 期望 {n}")
        self._mm = {"j": j, "rs": rs, "re": re, "w": w, "t": t, "p": p, "s": s}

    def __len__(self) -> int:
        return self._n

    def _parse_row(self, i: int) -> dict:
        self._ensure_mmaps()
        assert self._mm is not None
        s = int(self._mm["rs"][i])
        e = int(self._mm["re"][i])
        blob = self._mm["j"][s:e].tobytes()
        row = json.loads(blob.decode("utf-8").strip())
        if __debug__:
            ti = int(self._mm["t"][i])
            mv = row.get("human_move_pyffish")
            if mv in self.move_to_idx and self.move_to_idx[mv] != ti:
                raise AssertionError(
                    f"行 {i} 索引 targets 与 JSON human_move_pyffish 不一致"
                )
        return row

    def __getitem__(
        self, i: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        row = self._parse_row(i)
        assert self._mm is not None
        root0 = row["root_fen"]
        prefix0 = list(row["uci_prefix"])
        fen0 = row["fen"]
        human0 = row["human_move_pyffish"]
        w0 = float(self._mm["w"][i])
        ply_i = int(self._mm["p"][i])
        sid_i = int(self._mm["s"][i])

        use_mirror = self.aug_mirror_p > 0.0 and random.random() < self.aug_mirror_p
        if use_mirror:
            try:
                mh = mirror_pyffish_uci(human0)
                if mh not in self.move_to_idx:
                    use_mirror = False
                else:
                    root_m = mirror_fen(root0)
                    prefix_m = mirror_uci_prefix(prefix0)
                    fen_m = mirror_fen(fen0)
                    out = _policy_row_to_tensors(
                        self.move_to_idx,
                        root_m,
                        prefix_m,
                        fen_m,
                        mh,
                        weight=w0,
                    )
                    if self.with_row_meta:
                        return (
                            *out,
                            torch.tensor(ply_i, dtype=torch.long),
                            torch.tensor(sid_i, dtype=torch.long),
                        )
                    return out
            except (ValueError, RuntimeError):
                use_mirror = False

        out = _policy_row_to_tensors(
            self.move_to_idx, root0, prefix0, fen0, human0, weight=w0
        )
        if self.with_row_meta:
            return (
                *out,
                torch.tensor(ply_i, dtype=torch.long),
                torch.tensor(sid_i, dtype=torch.long),
            )
        return out
