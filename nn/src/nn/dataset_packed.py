"""离线 policy 包（mmap）：训练步不再调用 pyffish / json.loads。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from augment_mirror import mirror_fen, mirror_pyffish_uci
from nn.board_compact import compact_board_to_torch_planes, fen_to_compact_board
from nn.dataset import _TRAIN_MIRROR_PROB
from nn.policy_pack import (
    BOARD90,
    FEN_U8,
    LEGAL_IDX,
    LEGAL_PTR,
    PLIES,
    PGN_SRC_VOCAB_JSON,
    SRC_IDS,
    STM,
    TARGETS,
    WEIGHTS,
    assert_vocab_matches_pack,
    load_pack_meta,
    pack_dir_is_complete,
)


def _u8_line_to_str(row: np.ndarray) -> str:
    return bytes(np.asarray(row).tobytes().rstrip(b"\0")).decode("utf-8")


def _mirror_legal_indices(
    idxs: np.ndarray,
    idx_to_move: list[str],
    move_to_idx: dict[str, int],
) -> list[int]:
    out: list[int] = []
    for j in np.asarray(idxs, dtype=np.int64).tolist():
        u = idx_to_move[int(j)]
        try:
            mu = mirror_pyffish_uci(u)
        except ValueError:
            continue
        k = move_to_idx.get(mu)
        if k is not None:
            out.append(k)
    return sorted(set(out))


class PolicyPackedMmapDataset(Dataset):
    def __init__(
        self,
        pack_dir: Path | str,
        move_to_idx: dict[str, int],
        *,
        for_training: bool = False,
        with_row_meta: bool = False,
    ) -> None:
        self.pack_dir = Path(pack_dir)
        if not pack_dir_is_complete(self.pack_dir):
            raise FileNotFoundError(f"训练包不完整: {self.pack_dir}")
        meta = load_pack_meta(self.pack_dir)
        if int(meta["vocab_size"]) != len(move_to_idx):
            raise ValueError("pack_meta vocab_size 与 move_to_idx 长度不一致")
        assert_vocab_matches_pack(meta, move_to_idx)

        self.move_to_idx = move_to_idx
        self.vocab_size = len(move_to_idx)
        self.for_training = for_training
        self.with_row_meta = with_row_meta
        self.aug_mirror_p = _TRAIN_MIRROR_PROB if for_training else 0.0

        self._idx_to_move: list[str] = [""] * self.vocab_size
        for m, j in move_to_idx.items():
            self._idx_to_move[j] = m

        self._n = int(meta["n"])
        raw_vocab = json.loads(
            (self.pack_dir / PGN_SRC_VOCAB_JSON).read_text(encoding="utf-8")
        )
        self.pgn_source_vocab = [str(x) for x in raw_vocab]
        self._mm: dict[str, np.ndarray] | None = None

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
        d = self.pack_dir
        self._mm = {
            "b90": np.load(d / BOARD90, mmap_mode="r"),
            "stm": np.load(d / STM, mmap_mode="r"),
            "lp": np.load(d / LEGAL_PTR, mmap_mode="r"),
            "li": np.load(d / LEGAL_IDX, mmap_mode="r"),
            "t": np.load(d / TARGETS, mmap_mode="r"),
            "w": np.load(d / WEIGHTS, mmap_mode="r"),
            "p": np.load(d / PLIES, mmap_mode="r"),
            "s": np.load(d / SRC_IDS, mmap_mode="r"),
            "fen": np.load(d / FEN_U8, mmap_mode="r"),
        }

    def __len__(self) -> int:
        return self._n

    def __getitem__(
        self, i: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        self._ensure_mmaps()
        mm = self._mm
        assert mm is not None

        b90 = np.asarray(mm["b90"][i], dtype=np.uint8)
        stm = int(mm["stm"][i])
        board = compact_board_to_torch_planes(b90, stm)

        lo, hi = int(mm["lp"][i]), int(mm["lp"][i + 1])
        idxs = np.asarray(mm["li"][lo:hi], dtype=np.int64)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        if idxs.size > 0:
            mask[idxs] = True
        if not mask.any():
            raise RuntimeError(f"样本 {i} 无词表内合法着")

        ti = int(mm["t"][i])
        if not mask[ti]:
            raise RuntimeError(f"样本 {i} 标签不在合法掩码内: ti={ti}")
        w0 = float(mm["w"][i])
        ply_i = int(mm["p"][i])
        sid_i = int(mm["s"][i])
        human0 = self._idx_to_move[ti]

        fen0 = _u8_line_to_str(mm["fen"][i])

        use_mirror = self.aug_mirror_p > 0.0 and random.random() < self.aug_mirror_p
        if use_mirror:
            try:
                mh = mirror_pyffish_uci(human0)
                if mh not in self.move_to_idx:
                    use_mirror = False
                else:
                    fen_m = mirror_fen(fen0)
                    b90m, stmm = fen_to_compact_board(fen_m)
                    board_m = compact_board_to_torch_planes(b90m, stmm)
                    mir_ids = _mirror_legal_indices(
                        idxs, self._idx_to_move, self.move_to_idx
                    )
                    if not mir_ids:
                        use_mirror = False
                    else:
                        ti_m = self.move_to_idx[mh]
                        mask_m = torch.zeros(self.vocab_size, dtype=torch.bool)
                        mask_m[torch.tensor(mir_ids, dtype=torch.long)] = True
                        if not mask_m[ti_m]:
                            use_mirror = False
                        else:
                            out = (board_m, mask_m, torch.tensor(ti_m, dtype=torch.long), torch.tensor(w0, dtype=torch.float32))
                            if self.with_row_meta:
                                return (
                                    *out,
                                    torch.tensor(ply_i, dtype=torch.long),
                                    torch.tensor(sid_i, dtype=torch.long),
                                )
                            return out
            except (ValueError, RuntimeError):
                use_mirror = False

        out = (
            board,
            mask,
            torch.tensor(ti, dtype=torch.long),
            torch.tensor(w0, dtype=torch.float32),
        )
        if self.with_row_meta:
            return (
                *out,
                torch.tensor(ply_i, dtype=torch.long),
                torch.tensor(sid_i, dtype=torch.long),
            )
        return out
