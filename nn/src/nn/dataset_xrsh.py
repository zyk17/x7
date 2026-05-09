"""XRSH v1 分片目录 → PyTorch Dataset（紧凑棋盘 + 稀疏合法下标 + 标签）。"""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from augment_mirror import mirror_fen, mirror_pyffish_uci, mirror_uci_prefix
from nn.board_compact import compact_board_to_torch_planes, fen_to_compact_board
from nn.aux_pseudo_labels import pseudo_aux_labels_from_sample
from nn.policy_pack import assert_vocab_matches_pack

_TRAIN_MIRROR_PROB = 0.5
from nn.xrsh_io import load_pack_meta, read_shard_file, xrsh_dir_is_complete


def _mirror_legal_indices(
    idxs: list[int],
    idx_to_move: list[str],
    move_to_idx: dict[str, int],
) -> list[int]:
    out: list[int] = []
    for j in idxs:
        u = idx_to_move[int(j)]
        try:
            mu = mirror_pyffish_uci(u)
        except ValueError:
            continue
        k = move_to_idx.get(mu)
        if k is not None:
            out.append(k)
    return sorted(set(out))


class PolicyXrshDataset(Dataset):
    """从 ``shard_*.xrsh`` + ``pack_meta.json`` 载入。

    合法着已由 Rust 物化为下标；默认不在训练步调用 pyffish 枚举合法着。
    ``with_aux_labels=True`` 时结合 ``root_fen`` / ``uci_prefix``（及 Rust 物化的合法 UCI 表）
    调用 pyffish 生成伪标签；须由 XRSH v1 解析提供路径字段（见 ``xrsh_io.parse_shard_bytes``）。
    """

    def __init__(
        self,
        xrsh_dir: Path | str,
        move_to_idx: dict[str, int],
        *,
        for_training: bool = False,
        with_row_meta: bool = False,
        with_aux_labels: bool = False,
        with_value_labels: bool = False,
    ) -> None:
        self.root = Path(xrsh_dir)
        if not xrsh_dir_is_complete(self.root):
            raise FileNotFoundError(f"XRSH 目录不完整: {self.root}")
        meta = load_pack_meta(self.root)
        if meta.get("format") not in ("xrsh_v1", "xrsh_v2"):
            raise ValueError(
                f"非本仓库 XRSH 元数据 format={meta.get('format')!r}"
            )
        assert_vocab_matches_pack(meta, move_to_idx)

        self.move_to_idx = move_to_idx
        self.vocab_size = len(move_to_idx)
        self.for_training = for_training
        self.with_row_meta = with_row_meta
        self.with_aux_labels = bool(with_aux_labels)
        self.with_value_labels = bool(with_value_labels)
        self.aug_mirror_p = _TRAIN_MIRROR_PROB if for_training else 0.0

        self._idx_to_move: list[str] = [""] * self.vocab_size
        for m, j in move_to_idx.items():
            self._idx_to_move[j] = m

        rows: list[dict] = []
        ref_hash: bytes | None = None
        for sp in sorted(self.root.glob("shard_*.xrsh")):
            samples, vh = read_shard_file(sp)
            if ref_hash is None:
                ref_hash = vh
            elif vh != ref_hash:
                raise ValueError(f"分片词表哈希不一致: {sp}")
            rows.extend(samples)

        self.rows = rows

        self.position_weight_by_fen: dict[str, float] | None = None
        if for_training:
            cnt = Counter(r["fen"] for r in self.rows if r.get("fen"))
            self.position_weight_by_fen = {
                f: 1.0 / math.sqrt(n) for f, n in cnt.items() if n >= 1
            }

        self.pgn_source_vocab: list[str] = [""]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, i: int
    ) -> (
        tuple[torch.Tensor, ...]
    ):
        row = self.rows[i]
        fen0 = row["fen"]
        idxs = row["legal_idx"]
        ti = int(row["target_idx"])
        ply_i = int(row.get("ply", 0) or 0)

        b90, stm = fen_to_compact_board(fen0)
        board = compact_board_to_torch_planes(b90, stm)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for j in idxs:
            if 0 <= int(j) < self.vocab_size:
                mask[int(j)] = True
        if not mask.any():
            raise RuntimeError(f"样本 {i} 无有效合法着下标")
        if not (0 <= ti < self.vocab_size):
            raise RuntimeError(f"样本 {i} 标签下标越界: {ti}")
        if not mask[ti]:
            raise RuntimeError(f"样本 {i} 标签不在合法掩码内: ti={ti}")

        root0 = str(row.get("root_fen") or fen0)
        pfx0 = list(row.get("uci_prefix") or [])
        legal_uci0 = [
            self._idx_to_move[int(j)]
            for j in idxs
            if 0 <= int(j) < self.vocab_size
        ]

        atk0, dan0, tac0 = (0.5, 0.5, 0.5)
        need_aux_like = self.with_aux_labels or self.with_value_labels
        if need_aux_like:
            if (
                "aux_attack" in row
                and "aux_danger" in row
                and "aux_tactical" in row
            ):
                atk0 = float(row["aux_attack"])
                dan0 = float(row["aux_danger"])
                tac0 = float(row["aux_tactical"])
            else:
                atk0, dan0, tac0 = pseudo_aux_labels_from_sample(
                    fen0,
                    root_fen=root0,
                    uci_prefix=pfx0,
                    legal_uci=legal_uci0,
                )
        v0 = float(2.0 * atk0 - 1.0) if self.with_value_labels else 0.0

        w0 = 1.0
        if self.position_weight_by_fen is not None:
            w0 = self.position_weight_by_fen.get(fen0, 1.0)

        human0 = self._idx_to_move[ti]

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
                            if need_aux_like:
                                # 水平镜像不改变物质差、合法着数与「合法着吃子占比」启发 → 与分片内 Rust 标量一致。
                                # 仅在无预计算（旧 v1）时回退 pyffish。
                                if (
                                    "aux_attack" in row
                                    and "aux_danger" in row
                                    and "aux_tactical" in row
                                ):
                                    atk_m, dan_m, tac_m = atk0, dan0, tac0
                                else:
                                    root_m = mirror_fen(root0)
                                    pfx_m = mirror_uci_prefix(pfx0)
                                    mir_uci = [
                                        self._idx_to_move[int(j)] for j in mir_ids
                                    ]
                                    atk_m, dan_m, tac_m = pseudo_aux_labels_from_sample(
                                        fen_m,
                                        root_fen=root_m,
                                        uci_prefix=pfx_m,
                                        legal_uci=mir_uci,
                                    )
                            else:
                                atk_m, dan_m, tac_m = (atk0, dan0, tac0)
                            v_m = float(2.0 * atk_m - 1.0) if self.with_value_labels else 0.0
                            out = (
                                board_m,
                                mask_m,
                                torch.tensor(ti_m, dtype=torch.long),
                                torch.tensor(w0, dtype=torch.float32),
                            )
                            if self.with_aux_labels:
                                out = (
                                    *out,
                                    torch.tensor(atk_m, dtype=torch.float32),
                                    torch.tensor(dan_m, dtype=torch.float32),
                                    torch.tensor(tac_m, dtype=torch.float32),
                                )
                            if self.with_value_labels:
                                out = (*out, torch.tensor(v_m, dtype=torch.float32))
                            if self.with_row_meta:
                                return (
                                    *out,
                                    torch.tensor(ply_i, dtype=torch.long),
                                    torch.tensor(0, dtype=torch.long),
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
        if self.with_aux_labels:
            out = (
                *out,
                torch.tensor(atk0, dtype=torch.float32),
                torch.tensor(dan0, dtype=torch.float32),
                torch.tensor(tac0, dtype=torch.float32),
            )
        if self.with_value_labels:
            out = (*out, torch.tensor(v0, dtype=torch.float32))
        if self.with_row_meta:
            return (
                *out,
                torch.tensor(ply_i, dtype=torch.long),
                torch.tensor(0, dtype=torch.long),
            )
        return out
