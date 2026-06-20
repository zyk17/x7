"""XRSH v5 分片目录 -> PyTorch Dataset。"""

from __future__ import annotations

import mmap
import math
import os
import random
import tempfile
from collections import Counter, OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from augment_mirror import mirror_fen, mirror_move_uci
from nn.board_compact import compact_board_to_torch_planes, fen_to_compact_board
from nn.policy_pack import assert_vocab_matches_pack
from nn.xrsh_io import (
    XrshRowRef,
    fen_key64,
    load_pack_meta,
    read_row_train_at,
    read_shard_file,
    scan_shard_file,
    xrsh_dir_is_complete,
)

_TRAIN_MIRROR_PROB = 0.5
_SHARD_CACHE_LIMIT = 2
_EAGER_CACHE_VERSION = 4


@lru_cache(maxsize=32768)
def _fen_to_compact_cached(fen: str) -> tuple[object, int]:
    b90, stm = fen_to_compact_board(fen)
    return b90, int(stm)


def _mirror_legal_indices(
    idxs: list[int],
    idx_to_move: list[str],
    move_to_idx: dict[str, int],
) -> list[int]:
    out: list[int] = []
    for idx in idxs:
        move = idx_to_move[int(idx)]
        try:
            mirrored = mirror_move_uci(move)
        except ValueError:
            continue
        mapped = move_to_idx.get(mirrored)
        if mapped is not None:
            out.append(mapped)
    return sorted(set(out))


class PolicyXrshDataset(Dataset):
    """只保留 policy / search_q / search_counts 主线字段。"""

    def __init__(
        self,
        xrsh_dir: Path | str,
        move_to_idx: dict[str, int],
        *,
        for_training: bool = False,
        with_row_meta: bool = False,
        with_value_labels: bool = False,
        with_search_labels: bool = False,
        storage_mode: str = "eager",
    ) -> None:
        self.root = Path(xrsh_dir)
        if not xrsh_dir_is_complete(self.root):
            raise FileNotFoundError(f"XRSH 目录不完整: {self.root}")
        meta = load_pack_meta(self.root)
        if meta.get("format") != "xrsh_v5" or int(meta.get("format_version", 0)) != 5:
            raise ValueError("当前训练主线仅支持 XRSH v5")
        assert_vocab_matches_pack(meta, move_to_idx)

        self.move_to_idx = move_to_idx
        self.vocab_size = len(move_to_idx)
        self.for_training = bool(for_training)
        self.with_row_meta = bool(with_row_meta)
        self.with_value_labels = bool(with_value_labels)
        self.with_search_labels = bool(with_search_labels)
        self.storage_mode = str(storage_mode).lower()
        if self.storage_mode not in {"eager", "lazy"}:
            raise ValueError(f"未知 storage_mode: {storage_mode!r}")
        self.aug_mirror_p = _TRAIN_MIRROR_PROB if self.for_training else 0.0
        self._shard_paths = sorted(self.root.glob("shard_*.xrsh"))
        self._shard_cache: OrderedDict[int, tuple[object, mmap.mmap]] = OrderedDict()
        self._pack_meta_path = self.root / "pack_meta.json"
        self.cache_used = False
        self.cache_built = False

        self._idx_to_move: list[str] = [""] * self.vocab_size
        for move, idx in move_to_idx.items():
            self._idx_to_move[idx] = move

        self.row_refs: list[XrshRowRef] = []
        self.eager_compact_boards: np.ndarray | None = None
        self.eager_stms: np.ndarray | None = None
        self.eager_targets: np.ndarray | None = None
        self.eager_plies: np.ndarray | None = None
        self.eager_search_q: np.ndarray | None = None
        self.eager_search_visits: np.ndarray | None = None
        self.eager_legal_flat: np.ndarray | None = None
        self.eager_legal_offsets: np.ndarray | None = None
        self.eager_search_flat: np.ndarray | None = None

        if self.storage_mode == "eager":
            self._load_or_build_eager_cache()
        else:
            self._load_lazy_refs()

        self.position_weights: list[float] | None = None
        if self.for_training:
            if self.storage_mode == "eager":
                cnt = Counter(self._eager_fen_keys.tolist())
                self.position_weights = [
                    1.0 / math.sqrt(cnt[key]) for key in self._eager_fen_keys.tolist()
                ]
            else:
                cnt = Counter(ref.fen_key for ref in self.row_refs)
                self.position_weights = [
                    1.0 / math.sqrt(cnt[ref.fen_key]) for ref in self.row_refs
                ]

        self.pgn_source_vocab: list[str] = [""]

    def __len__(self) -> int:
        if self.storage_mode == "eager":
            assert self.eager_targets is not None
            return int(self.eager_targets.shape[0])
        return len(self.row_refs)

    def _dataset_signature(self) -> str:
        parts = [f"cache_v={_EAGER_CACHE_VERSION}"]
        st = self._pack_meta_path.stat()
        parts.append(
            f"meta:{self._pack_meta_path.name}:{st.st_size}:{getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))}"
        )
        for sp in self._shard_paths:
            st = sp.stat()
            parts.append(
                f"{sp.name}:{st.st_size}:{getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))}"
            )
        return "|".join(parts)

    def _eager_cache_path(self) -> Path:
        return self.root / ".cache" / "policy_xrsh_eager_v4.npz"

    def _load_eager_rows(self) -> None:
        compact_boards: list[np.ndarray] = []
        stms: list[int] = []
        targets: list[int] = []
        plies: list[int] = []
        search_qs: list[float] = []
        search_visits: list[int] = []
        fen_keys: list[int] = []
        legal_offsets: list[int] = [0]
        legal_flat: list[int] = []
        search_flat: list[int] = []
        row_group_ids: list[int] = []
        ref_hash: bytes | None = None
        game_group = 0

        for sp in self._shard_paths:
            rows, vocab_hash = read_shard_file(sp)
            if ref_hash is None:
                ref_hash = vocab_hash
            elif vocab_hash != ref_hash:
                raise ValueError(f"分片词表哈希不一致: {sp}")
            gid_to_group: dict[str, int] = {}
            for row in rows:
                gid = str(row.get("game_id", ""))
                grp = gid_to_group.get(gid)
                if grp is None:
                    grp = game_group
                    gid_to_group[gid] = grp
                    game_group += 1
                fen = str(row["fen"])
                b90, stm = fen_to_compact_board(fen)
                compact_boards.append(np.asarray(b90, dtype=np.uint8))
                stms.append(int(stm))
                targets.append(int(row["target_idx"]))
                plies.append(int(row.get("ply", 0) or 0))
                search_qs.append(float(row.get("search_q", 0.0) or 0.0))
                search_visits.append(int(row.get("search_visits", 0) or 0))
                fen_keys.append(fen_key64(fen))
                idxs = [int(x) for x in row["legal_idx"]]
                counts = [int(x) for x in row.get("search_counts", [])]
                if counts and len(counts) != len(idxs):
                    raise ValueError("search_counts 与 legal_idx 长度不一致")
                legal_flat.extend(idxs)
                search_flat.extend(counts or [0] * len(idxs))
                legal_offsets.append(len(legal_flat))
                row_group_ids.append(grp)

        self.eager_compact_boards = (
            np.stack(compact_boards, axis=0)
            if compact_boards
            else np.zeros((0, 90), dtype=np.uint8)
        )
        self.eager_stms = np.asarray(stms, dtype=np.uint8)
        self.eager_targets = np.asarray(targets, dtype=np.int32)
        self.eager_plies = np.asarray(plies, dtype=np.uint16)
        self.eager_search_q = np.asarray(search_qs, dtype=np.float32)
        self.eager_search_visits = np.asarray(search_visits, dtype=np.uint32)
        self._eager_fen_keys = np.asarray(fen_keys, dtype=np.uint64)
        self.eager_legal_flat = np.asarray(legal_flat, dtype=np.int32)
        self.eager_legal_offsets = np.asarray(legal_offsets, dtype=np.int64)
        self.eager_search_flat = np.asarray(search_flat, dtype=np.uint16)
        self.row_group_ids = row_group_ids

    def _apply_eager_arrays(
        self,
        *,
        compact_boards: np.ndarray,
        stms: np.ndarray,
        targets: np.ndarray,
        plies: np.ndarray,
        search_q: np.ndarray,
        search_visits: np.ndarray,
        fen_keys: np.ndarray,
        legal_flat: np.ndarray,
        legal_offsets: np.ndarray,
        search_flat: np.ndarray,
        row_group_ids: np.ndarray,
    ) -> None:
        self.eager_compact_boards = np.asarray(compact_boards, dtype=np.uint8)
        self.eager_stms = np.asarray(stms, dtype=np.uint8)
        self.eager_targets = np.asarray(targets, dtype=np.int32)
        self.eager_plies = np.asarray(plies, dtype=np.uint16)
        self.eager_search_q = np.asarray(search_q, dtype=np.float32)
        self.eager_search_visits = np.asarray(search_visits, dtype=np.uint32)
        self._eager_fen_keys = np.asarray(fen_keys, dtype=np.uint64)
        self.eager_legal_flat = np.asarray(legal_flat, dtype=np.int32)
        self.eager_legal_offsets = np.asarray(legal_offsets, dtype=np.int64)
        self.eager_search_flat = np.asarray(search_flat, dtype=np.uint16)
        self.row_group_ids = np.asarray(row_group_ids, dtype=np.int32).tolist()

    def _save_eager_cache(self) -> None:
        cp = self._eager_cache_path()
        cp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix="policy_xrsh_eager_",
            suffix=".npz",
            dir=str(cp.parent),
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(
                    fh,
                    signature=np.asarray(self._dataset_signature()),
                    compact_boards=self.eager_compact_boards,
                    stms=self.eager_stms,
                    targets=self.eager_targets,
                    plies=self.eager_plies,
                    search_q=self.eager_search_q,
                    search_visits=self.eager_search_visits,
                    fen_keys=self._eager_fen_keys,
                    legal_flat=self.eager_legal_flat,
                    legal_offsets=self.eager_legal_offsets,
                    search_flat=self.eager_search_flat,
                    row_group_ids=np.asarray(self.row_group_ids, dtype=np.int32),
                )
            Path(tmp_name).replace(cp)
        finally:
            tmp_path = Path(tmp_name)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _load_or_build_eager_cache(self) -> None:
        cp = self._eager_cache_path()
        sig = self._dataset_signature()
        if cp.is_file():
            try:
                with np.load(cp, allow_pickle=False) as z:
                    cached_sig = str(z["signature"].item())
                    if cached_sig == sig:
                        self._apply_eager_arrays(
                            compact_boards=z["compact_boards"],
                            stms=z["stms"],
                            targets=z["targets"],
                            plies=z["plies"],
                            search_q=z["search_q"],
                            search_visits=z["search_visits"],
                            fen_keys=z["fen_keys"],
                            legal_flat=z["legal_flat"],
                            legal_offsets=z["legal_offsets"],
                            search_flat=z["search_flat"],
                            row_group_ids=z["row_group_ids"],
                        )
                        self.cache_used = True
                        return
            except Exception:
                pass
        self._load_eager_rows()
        self._save_eager_cache()
        self.cache_built = True

    def _load_lazy_refs(self) -> None:
        row_refs: list[XrshRowRef] = []
        ref_hash: bytes | None = None
        next_game_group = 0
        for shard_index, sp in enumerate(self._shard_paths):
            refs, vocab_hash, next_game_group = scan_shard_file(
                sp,
                shard_index=shard_index,
                start_game_group=next_game_group,
            )
            if ref_hash is None:
                ref_hash = vocab_hash
            elif vocab_hash != ref_hash:
                raise ValueError(f"分片词表哈希不一致: {sp}")
            row_refs.extend(refs)
        self.row_refs = row_refs
        self.row_group_ids = [ref.game_group for ref in self.row_refs]

    def _get_shard_buf(self, shard_index: int) -> mmap.mmap:
        cached = self._shard_cache.get(shard_index)
        if cached is not None:
            fh, mm = cached
            self._shard_cache.move_to_end(shard_index)
            return mm
        fh = open(self._shard_paths[shard_index], "rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._shard_cache[shard_index] = (fh, mm)
        while len(self._shard_cache) > _SHARD_CACHE_LIMIT:
            _, (old_fh, old_mm) = self._shard_cache.popitem(last=False)
            old_mm.close()
            old_fh.close()
        return mm

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        if self.storage_mode == "eager":
            assert self.eager_compact_boards is not None
            assert self.eager_stms is not None
            assert self.eager_targets is not None
            assert self.eager_plies is not None
            assert self.eager_search_q is not None
            assert self.eager_search_visits is not None
            assert self.eager_legal_flat is not None
            assert self.eager_legal_offsets is not None
            assert self.eager_search_flat is not None
            b90 = self.eager_compact_boards[i]
            stm = int(self.eager_stms[i])
            ti = int(self.eager_targets[i])
            ply_i = int(self.eager_plies[i])
            sq_i = float(self.eager_search_q[i])
            sv_i = int(self.eager_search_visits[i])
            lo = int(self.eager_legal_offsets[i])
            hi = int(self.eager_legal_offsets[i + 1])
            idxs = self.eager_legal_flat[lo:hi]
            search_counts = self.eager_search_flat[lo:hi]
            fen0 = None
            board = compact_board_to_torch_planes(b90, stm)
        else:
            ref = self.row_refs[i]
            fen0, idxs, ti, sq_i, sv_i, search_counts = read_row_train_at(
                self._get_shard_buf(ref.shard_index),
                ref.row_offset,
            )
            b90, stm = _fen_to_compact_cached(fen0)
            board = compact_board_to_torch_planes(b90, stm)
            ply_i = int(ref.ply)

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for idx in idxs:
            if 0 <= int(idx) < self.vocab_size:
                mask[int(idx)] = True
        if not mask.any():
            raise RuntimeError(f"样本 {i} 无有效合法着下标")
        if not (0 <= ti < self.vocab_size):
            raise RuntimeError(f"样本 {i} 标签下标越界: {ti}")
        if not mask[ti]:
            raise RuntimeError(f"样本 {i} 标签不在合法掩码内: ti={ti}")

        visit_target: torch.Tensor | None = None
        if self.with_search_labels:
            visit_target = torch.zeros(self.vocab_size, dtype=torch.float32)
            total = max(int(sum(int(x) for x in search_counts)), 1)
            for idx, count in zip(idxs, search_counts):
                if 0 <= int(idx) < self.vocab_size:
                    visit_target[int(idx)] = float(count) / float(total)

        sample_weight = 1.0 if self.position_weights is None else self.position_weights[i]
        human_move = self._idx_to_move[ti]

        use_mirror = (
            fen0 is not None
            and not self.with_search_labels
            and self.aug_mirror_p > 0.0
            and random.random() < self.aug_mirror_p
        )
        if use_mirror:
            try:
                mirrored_move = mirror_move_uci(human_move)
                mirrored_idx = self.move_to_idx.get(mirrored_move)
                if mirrored_idx is not None:
                    fen_m = mirror_fen(fen0)
                    b90m, stmm = fen_to_compact_board(fen_m)
                    mirrored_legals = _mirror_legal_indices(
                        idxs,
                        self._idx_to_move,
                        self.move_to_idx,
                    )
                    if mirrored_legals:
                        mask_m = torch.zeros(self.vocab_size, dtype=torch.bool)
                        mask_m[torch.tensor(mirrored_legals, dtype=torch.long)] = True
                        if mask_m[mirrored_idx]:
                            out = (
                                compact_board_to_torch_planes(b90m, stmm),
                                mask_m,
                                torch.tensor(mirrored_idx, dtype=torch.long),
                                torch.tensor(sample_weight, dtype=torch.float32),
                            )
                            if self.with_value_labels:
                                out = (*out, torch.tensor(sq_i, dtype=torch.float32))
                            if self.with_row_meta:
                                out = (
                                    *out,
                                    torch.tensor(ply_i, dtype=torch.long),
                                    torch.tensor(0, dtype=torch.long),
                                )
                            return out
            except (RuntimeError, ValueError):
                pass

        out = (
            board,
            mask,
            torch.tensor(ti, dtype=torch.long),
            torch.tensor(sample_weight, dtype=torch.float32),
        )
        if self.with_value_labels:
            out = (*out, torch.tensor(sq_i, dtype=torch.float32))
        if self.with_search_labels and visit_target is not None:
            out = (
                *out,
                visit_target,
                torch.tensor(sq_i, dtype=torch.float32),
                torch.tensor(sv_i, dtype=torch.long),
            )
        if self.with_row_meta:
            out = (
                *out,
                torch.tensor(ply_i, dtype=torch.long),
                torch.tensor(0, dtype=torch.long),
            )
        return out
