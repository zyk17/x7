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
from nn.board_compact import fen_to_compact_board, mirror_compact_board
from nn.policy_pack import assert_vocab_matches_pack
from nn.dataset_batch import (
    SAMPLE_BOARD90,
    SAMPLE_LEGAL_IDX,
    SAMPLE_PLY,
    SAMPLE_SEARCH_COUNTS,
    SAMPLE_SEARCH_VISITS,
    SAMPLE_SRC_ID,
    SAMPLE_STM,
    SAMPLE_T_VAL,
    SAMPLE_TARGET,
    SAMPLE_VOCAB_SIZE,
    SAMPLE_WEIGHT,
)
from nn.xrsh_io import (
    XrshRowRef,
    assert_shard_binary_v5,
    fen_key64,
    load_pack_meta,
    read_row_train_at,
    scan_shard_file,
    scan_shard_train_rows,
    xrsh_dir_is_complete,
)

_TRAIN_MIRROR_PROB = 0.5
_SHARD_CACHE_LIMIT = 2
_EAGER_CACHE_VERSION = 5


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
        for sp in self._shard_paths:
            assert_shard_binary_v5(sp)
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
        return self.root / ".cache" / f"policy_xrsh_eager_v{_EAGER_CACHE_VERSION}.npz"

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
        next_game_group = 0

        for sp in self._shard_paths:
            rows, vocab_hash, next_game_group = scan_shard_train_rows(
                sp,
                start_game_group=next_game_group,
            )
            if ref_hash is None:
                ref_hash = vocab_hash
            elif vocab_hash != ref_hash:
                raise ValueError(f"分片词表哈希不一致: {sp}")
            for row in rows:
                fen = row.fen
                b90, stm = fen_to_compact_board(fen)
                compact_boards.append(np.asarray(b90, dtype=np.uint8))
                stms.append(int(stm))
                targets.append(int(row.target_idx))
                plies.append(int(row.ply))
                search_qs.append(float(row.search_q))
                search_visits.append(int(row.search_visits))
                fen_keys.append(fen_key64(fen))
                idxs = row.legal_idx
                counts = row.search_counts
                if counts and len(counts) != len(idxs):
                    raise ValueError("search_counts 与 legal_idx 长度不一致")
                legal_flat.extend(idxs)
                search_flat.extend(counts or [0] * len(idxs))
                legal_offsets.append(len(legal_flat))
                row_group_ids.append(int(row.game_group))

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

    def _sample_dict(
        self,
        *,
        board90: np.ndarray,
        stm: int,
        target: torch.Tensor,
        weight: torch.Tensor,
        sq_i: float,
        sv_i: int,
        idxs: np.ndarray | list[int],
        search_counts: np.ndarray | list[int],
        ply_i: int,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            SAMPLE_BOARD90: torch.as_tensor(board90, dtype=torch.uint8).reshape(90),
            SAMPLE_STM: torch.tensor(int(stm), dtype=torch.uint8),
            SAMPLE_LEGAL_IDX: torch.as_tensor(idxs, dtype=torch.long),
            SAMPLE_VOCAB_SIZE: torch.tensor(self.vocab_size, dtype=torch.long),
            SAMPLE_TARGET: target,
            SAMPLE_WEIGHT: weight,
        }
        if self.with_search_labels:
            out[SAMPLE_SEARCH_COUNTS] = torch.as_tensor(search_counts, dtype=torch.long)
        if self.with_value_labels:
            out[SAMPLE_T_VAL] = torch.tensor(sq_i, dtype=torch.float32)
        if self.with_value_labels:
            out[SAMPLE_SEARCH_VISITS] = torch.tensor(sv_i, dtype=torch.long)
        if self.with_row_meta:
            out[SAMPLE_PLY] = torch.tensor(ply_i, dtype=torch.long)
            out[SAMPLE_SRC_ID] = torch.tensor(0, dtype=torch.long)
        return out

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
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
        else:
            ref = self.row_refs[i]
            fen0, idxs, ti, sq_i, sv_i, search_counts = read_row_train_at(
                self._get_shard_buf(ref.shard_index),
                ref.row_offset,
            )
            b90, stm = _fen_to_compact_cached(fen0)
            ply_i = int(ref.ply)

        idx_arr = np.asarray(idxs, dtype=np.int64).reshape(-1)
        counts_arr = np.asarray(search_counts, dtype=np.int64).reshape(-1)
        valid = (idx_arr >= 0) & (idx_arr < self.vocab_size)
        idx_arr = idx_arr[valid]
        counts_arr = counts_arr[valid]
        if idx_arr.size == 0:
            raise RuntimeError(f"样本 {i} 无有效合法着下标")
        if not (0 <= ti < self.vocab_size):
            raise RuntimeError(f"样本 {i} 标签下标越界: {ti}")
        if not np.any(idx_arr == ti):
            raise RuntimeError(f"样本 {i} 标签不在合法掩码内: ti={ti}")

        sample_weight = 1.0 if self.position_weights is None else self.position_weights[i]
        human_move = self._idx_to_move[ti]

        use_mirror = (
            not self.with_search_labels
            and self.aug_mirror_p > 0.0
            and random.random() < self.aug_mirror_p
        )
        if use_mirror:
            try:
                mirrored_move = mirror_move_uci(human_move)
                mirrored_idx = self.move_to_idx.get(mirrored_move)
                if mirrored_idx is not None:
                    mirrored_legals = _mirror_legal_indices(
                        idxs,
                        self._idx_to_move,
                        self.move_to_idx,
                    )
                    if mirrored_legals:
                        mirrored_legals_arr = np.asarray(mirrored_legals, dtype=np.int64)
                        if np.any(mirrored_legals_arr == mirrored_idx):
                            if fen0 is not None:
                                fen_m = mirror_fen(fen0)
                                b90m, stmm = fen_to_compact_board(fen_m)
                            else:
                                b90m = mirror_compact_board(b90)
                                stmm = stm
                            return self._sample_dict(
                                board90=b90m,
                                stm=stmm,
                                target=torch.tensor(mirrored_idx, dtype=torch.long),
                                weight=torch.tensor(sample_weight, dtype=torch.float32),
                                sq_i=sq_i,
                                sv_i=sv_i,
                                idxs=mirrored_legals_arr,
                                search_counts=np.zeros_like(mirrored_legals_arr),
                                ply_i=ply_i,
                            )
            except (RuntimeError, ValueError):
                pass

        return self._sample_dict(
            board90=b90,
            stm=stm,
            target=torch.tensor(ti, dtype=torch.long),
            weight=torch.tensor(sample_weight, dtype=torch.float32),
            sq_i=sq_i,
            sv_i=sv_i,
            idxs=idx_arr,
            search_counts=counts_arr,
            ply_i=ply_i,
        )


class MixedPolicyXrshDataset(Dataset):
    """按权重拼接多个 XRSH 目录，用于受控混合训练。"""

    def __init__(
        self,
        sources: list[tuple[Path | str, float]],
        move_to_idx: dict[str, int],
        *,
        for_training: bool = False,
        with_row_meta: bool = False,
        with_value_labels: bool = False,
        with_search_labels: bool = False,
        storage_mode: str = "eager",
    ) -> None:
        if not sources:
            raise ValueError("sources 不能为空")
        self.parts: list[PolicyXrshDataset] = []
        self._offsets: list[int] = [0]
        self.mix_weights: list[float] = []
        self.row_group_ids: list[int] = []
        self.pgn_source_vocab: list[str] = [""]
        group_base = 0
        for xrsh_dir, mix_w in sources:
            ds = PolicyXrshDataset(
                xrsh_dir,
                move_to_idx,
                for_training=for_training,
                with_row_meta=with_row_meta,
                with_value_labels=with_value_labels,
                with_search_labels=with_search_labels,
                storage_mode=storage_mode,
            )
            self.parts.append(ds)
            n = len(ds)
            pos_w = ds.position_weights if ds.position_weights is not None else [1.0] * n
            self.mix_weights.extend(float(mix_w) * float(w) for w in pos_w)
            self.row_group_ids.extend(int(g) + group_base for g in ds.row_group_ids)
            group_base = max(self.row_group_ids) + 1 if self.row_group_ids else 0
            self._offsets.append(self._offsets[-1] + n)

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        lo, hi = 0, len(self.parts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= index:
                lo = mid
            else:
                hi = mid - 1
        return lo, index - self._offsets[lo]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        part_i, local_i = self._locate(index)
        sample = dict(self.parts[part_i][local_i])
        if self.parts[0].for_training:
            sample[SAMPLE_WEIGHT] = torch.tensor(self.mix_weights[index], dtype=torch.float32)
        return sample
