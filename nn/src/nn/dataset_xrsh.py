"""XRSH v3 分片目录 → PyTorch Dataset（紧凑棋盘 + 稀疏合法下标 + 四头标签）。"""

from __future__ import annotations

import mmap
import math
import os
import random
from collections import Counter, OrderedDict
from functools import lru_cache
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

from augment_mirror import mirror_fen, mirror_move_uci
from nn.board_compact import compact_board_to_torch_planes, fen_to_compact_board
from nn.policy_pack import assert_vocab_matches_pack

_TRAIN_MIRROR_PROB = 0.5
from nn.xrsh_io import (
    XrshRowRef,
    fen_key64,
    load_pack_meta,
    read_shard_file,
    read_row_train_at,
    scan_shard_file,
    xrsh_dir_is_complete,
)

_SHARD_CACHE_LIMIT = 2
_EAGER_CACHE_VERSION = 1


@lru_cache(maxsize=32768)
def _fen_to_compact_cached(fen: str) -> tuple[object, int]:
    b90, stm = fen_to_compact_board(fen)
    return b90, int(stm)


def value_target_side_to_move(
    fen: str,
    game_result_red: int,
    ply: int,
    ply_total: int,
    *,
    progress_gamma: float = 1.5,
) -> float:
    """当前行棋方视角结局 × ``progress ** progress_gamma``（见仓库 ``temp.md``）。

    ``progress = ply / max(1, ply_total - 1)``（0-based ply）；``gamma`` 越大越早局越贴近 0。
    ``game_result_red``：``1/0/-1``；``2`` 未知须由 Dataset 预先过滤。
    """
    parts = fen.split()
    stm = parts[1] if len(parts) > 1 else "w"
    is_red = stm == "w"
    if game_result_red == 1:
        outcome_red = 1.0
    elif game_result_red == -1:
        outcome_red = -1.0
    else:
        outcome_red = 0.0
    base = outcome_red if is_red else -outcome_red
    pt = max(int(ply_total), 1)
    if pt <= 1:
        progress = 1.0
    else:
        progress = min(max(float(ply) / float(pt - 1), 0.0), 1.0)
    w = progress ** float(progress_gamma)
    return base * w


def _mirror_legal_indices(
    idxs: list[int],
    idx_to_move: list[str],
    move_to_idx: dict[str, int],
) -> list[int]:
    out: list[int] = []
    for j in idxs:
        u = idx_to_move[int(j)]
        try:
            mu = mirror_move_uci(u)
        except ValueError:
            continue
        k = move_to_idx.get(mu)
        if k is not None:
            out.append(k)
    return sorted(set(out))


class PolicyXrshDataset(Dataset):
    """从 ``shard_*.xrsh`` + ``pack_meta.json`` 载入 XRSH v3。"""

    def __init__(
        self,
        xrsh_dir: Path | str,
        move_to_idx: dict[str, int],
        *,
        for_training: bool = False,
        with_row_meta: bool = False,
        with_aux_labels: bool = False,
        with_value_labels: bool = False,
        value_progress_gamma: float = 1.5,
        storage_mode: str = "eager",
    ) -> None:
        self.root = Path(xrsh_dir)
        if not xrsh_dir_is_complete(self.root):
            raise FileNotFoundError(f"XRSH 目录不完整: {self.root}")
        meta = load_pack_meta(self.root)
        if meta.get("format") != "xrsh_v3" or int(meta.get("format_version", 0)) != 3:
            raise ValueError(
                "当前训练主线仅支持 XRSH v3（pack_meta.format=xrsh_v3, format_version=3）"
            )
        assert_vocab_matches_pack(meta, move_to_idx)

        self.move_to_idx = move_to_idx
        self.vocab_size = len(move_to_idx)
        self.for_training = for_training
        self.with_row_meta = with_row_meta
        self.with_aux_labels = bool(with_aux_labels)
        self.with_value_labels = bool(with_value_labels)
        self._value_progress_gamma = float(value_progress_gamma)
        self.aug_mirror_p = _TRAIN_MIRROR_PROB if for_training else 0.0
        self.filtered_unknown_value_rows = 0
        self.storage_mode = str(storage_mode).lower()
        if self.storage_mode not in {"eager", "lazy"}:
            raise ValueError(f"未知 storage_mode: {storage_mode!r}")
        self._shard_paths = sorted(self.root.glob("shard_*.xrsh"))
        self._shard_cache: OrderedDict[int, tuple[object, mmap.mmap]] = OrderedDict()
        self._pack_meta_path = self.root / "pack_meta.json"
        self.cache_used = False
        self.cache_built = False

        self._idx_to_move: list[str] = [""] * self.vocab_size
        for m, j in move_to_idx.items():
            self._idx_to_move[j] = m

        self.row_refs: list[XrshRowRef] = []
        self.eager_compact_boards: np.ndarray | None = None
        self.eager_stms: np.ndarray | None = None
        self.eager_targets: np.ndarray | None = None
        self.eager_plies: np.ndarray | None = None
        self.eager_aux: np.ndarray | None = None
        self.eager_result_red: np.ndarray | None = None
        self.eager_ply_total: np.ndarray | None = None
        self.eager_legal_flat: np.ndarray | None = None
        self.eager_legal_offsets: np.ndarray | None = None
        if self.storage_mode == "eager":
            self._load_or_build_eager_cache()
        else:
            self._load_lazy_refs()

        if self.with_aux_labels:
            # XRSH v3 扫描阶段已验证版本；aux 字段是否存在由单行解析时保证。
            self._aux_field_guard = True

        self.position_weights: list[float] | None = None
        if for_training:
            if self.storage_mode == "eager":
                assert self.eager_compact_boards is not None
                assert self.eager_stms is not None
                cnt = Counter(self._eager_fen_keys.tolist())
                self.position_weights = [
                    1.0 / math.sqrt(cnt[k]) for k in self._eager_fen_keys.tolist()
                ]
            else:
                cnt = Counter(r.fen_key for r in self.row_refs)
                self.position_weights = [
                    1.0 / math.sqrt(cnt[r.fen_key]) for r in self.row_refs
                ]

        self.pgn_source_vocab: list[str] = [""]

    def __len__(self) -> int:
        if self.storage_mode == "eager":
            assert self.eager_targets is not None
            return int(self.eager_targets.shape[0])
        return len(self.row_refs)

    def _load_eager_rows(self) -> None:
        compact_boards: list[np.ndarray] = []
        stms: list[int] = []
        targets: list[int] = []
        plies: list[int] = []
        auxs: list[tuple[float, float, float]] = []
        result_reds: list[int] = []
        ply_totals: list[int] = []
        fen_keys: list[int] = []
        legal_offsets: list[int] = [0]
        legal_flat: list[int] = []
        row_group_ids: list[int] = []
        ref_hash: bytes | None = None
        game_group = 0
        for sp in self._shard_paths:
            samples, vh = read_shard_file(sp)
            if ref_hash is None:
                ref_hash = vh
            elif vh != ref_hash:
                raise ValueError(f"分片词表哈希不一致: {sp}")
            gid_to_group: dict[str, int] = {}
            for r in samples:
                gid = str(r.get("game_id", ""))
                grp = gid_to_group.get(gid)
                if grp is None:
                    grp = game_group
                    gid_to_group[gid] = grp
                    game_group += 1
                gr = int(r.get("game_result_red", 2))
                if self.with_value_labels and gr == 2:
                    self.filtered_unknown_value_rows += 1
                    continue
                fen = str(r["fen"])
                b90, stm = fen_to_compact_board(fen)
                compact_boards.append(np.asarray(b90, dtype=np.uint8))
                stms.append(int(stm))
                targets.append(int(r["target_idx"]))
                plies.append(int(r.get("ply", 0) or 0))
                auxs.append(
                    (
                        float(r["aux_attack"]),
                        float(r["aux_danger"]),
                        float(r["aux_tactical"]),
                    )
                )
                result_reds.append(gr)
                ply_totals.append(int(r.get("ply_total", 0) or 0))
                fen_keys.append(fen_key64(fen))
                idxs = [int(x) for x in r["legal_idx"]]
                legal_flat.extend(idxs)
                legal_offsets.append(len(legal_flat))
                row_group_ids.append(grp)
        if self.with_value_labels and not compact_boards:
            raise ValueError(
                "启用 value 头后，XRSH 中不存在已解析 [Result] 的样本；"
                "请补齐 PGN 结局，或训练时加 --no-value-head"
            )
        self.eager_compact_boards = np.stack(compact_boards, axis=0) if compact_boards else np.zeros((0, 90), dtype=np.uint8)
        self.eager_stms = np.asarray(stms, dtype=np.uint8)
        self.eager_targets = np.asarray(targets, dtype=np.int32)
        self.eager_plies = np.asarray(plies, dtype=np.uint16)
        self.eager_aux = np.asarray(auxs, dtype=np.float32).reshape(-1, 3)
        self.eager_result_red = np.asarray(result_reds, dtype=np.int8)
        self.eager_ply_total = np.asarray(ply_totals, dtype=np.uint16)
        self._eager_fen_keys = np.asarray(fen_keys, dtype=np.uint64)
        self.eager_legal_flat = np.asarray(legal_flat, dtype=np.int32)
        self.eager_legal_offsets = np.asarray(legal_offsets, dtype=np.int64)
        self.row_group_ids = row_group_ids

    def _dataset_signature(self) -> str:
        parts: list[str] = [f"cache_v={_EAGER_CACHE_VERSION}"]
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
        return self.root / ".cache" / "policy_xrsh_eager_v1.npz"

    def _apply_eager_arrays(
        self,
        *,
        compact_boards: np.ndarray,
        stms: np.ndarray,
        targets: np.ndarray,
        plies: np.ndarray,
        aux: np.ndarray,
        result_red: np.ndarray,
        ply_total: np.ndarray,
        fen_keys: np.ndarray,
        legal_flat: np.ndarray,
        legal_offsets: np.ndarray,
        row_group_ids: np.ndarray,
    ) -> None:
        if self.with_value_labels:
            keep = result_red != 2
            self.filtered_unknown_value_rows = int((~keep).sum())
            if not bool(keep.any()):
                raise ValueError(
                    "启用 value 头后，XRSH 中不存在已解析 [Result] 的样本；"
                    "请补齐 PGN 结局，或训练时加 --no-value-head"
                )
            if not bool(keep.all()):
                idx_keep = np.nonzero(keep)[0].astype(np.int64)
                new_offsets = [0]
                new_flat: list[np.ndarray] = []
                for idx in idx_keep.tolist():
                    lo = int(legal_offsets[idx])
                    hi = int(legal_offsets[idx + 1])
                    new_flat.append(legal_flat[lo:hi])
                    new_offsets.append(new_offsets[-1] + (hi - lo))
                legal_flat = (
                    np.concatenate(new_flat, axis=0)
                    if new_flat
                    else np.zeros((0,), dtype=np.int32)
                )
                legal_offsets = np.asarray(new_offsets, dtype=np.int64)
                compact_boards = compact_boards[idx_keep]
                stms = stms[idx_keep]
                targets = targets[idx_keep]
                plies = plies[idx_keep]
                aux = aux[idx_keep]
                result_red = result_red[idx_keep]
                ply_total = ply_total[idx_keep]
                fen_keys = fen_keys[idx_keep]
                row_group_ids = row_group_ids[idx_keep]
        self.eager_compact_boards = np.asarray(compact_boards, dtype=np.uint8)
        self.eager_stms = np.asarray(stms, dtype=np.uint8)
        self.eager_targets = np.asarray(targets, dtype=np.int32)
        self.eager_plies = np.asarray(plies, dtype=np.uint16)
        self.eager_aux = np.asarray(aux, dtype=np.float32).reshape(-1, 3)
        self.eager_result_red = np.asarray(result_red, dtype=np.int8)
        self.eager_ply_total = np.asarray(ply_total, dtype=np.uint16)
        self._eager_fen_keys = np.asarray(fen_keys, dtype=np.uint64)
        self.eager_legal_flat = np.asarray(legal_flat, dtype=np.int32)
        self.eager_legal_offsets = np.asarray(legal_offsets, dtype=np.int64)
        self.row_group_ids = np.asarray(row_group_ids, dtype=np.int32).tolist()

    def _save_eager_cache(self) -> None:
        cp = self._eager_cache_path()
        cp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="policy_xrsh_eager_", suffix=".npz", dir=str(cp.parent))
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(
                    fh,
                    signature=np.asarray(self._dataset_signature()),
                    compact_boards=self.eager_compact_boards,
                    stms=self.eager_stms,
                    targets=self.eager_targets,
                    plies=self.eager_plies,
                    aux=self.eager_aux,
                    result_red=self.eager_result_red,
                    ply_total=self.eager_ply_total,
                    fen_keys=self._eager_fen_keys,
                    legal_flat=self.eager_legal_flat,
                    legal_offsets=self.eager_legal_offsets,
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
                            aux=z["aux"],
                            result_red=z["result_red"],
                            ply_total=z["ply_total"],
                            fen_keys=z["fen_keys"],
                            legal_flat=z["legal_flat"],
                            legal_offsets=z["legal_offsets"],
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
            refs, vh, next_game_group = scan_shard_file(
                sp, shard_index=shard_index, start_game_group=next_game_group
            )
            if ref_hash is None:
                ref_hash = vh
            elif vh != ref_hash:
                raise ValueError(f"分片词表哈希不一致: {sp}")
            row_refs.extend(refs)
        if self.with_value_labels:
            filtered_refs = [r for r in row_refs if int(r.game_result_red) != 2]
            self.filtered_unknown_value_rows = len(row_refs) - len(filtered_refs)
            row_refs = filtered_refs
            if not row_refs:
                raise ValueError(
                    "启用 value 头后，XRSH 中不存在已解析 [Result] 的样本；"
                    "请补齐 PGN 结局，或训练时加 --no-value-head"
                )
        self.row_refs = row_refs
        self.row_group_ids = [r.game_group for r in self.row_refs]

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

    def __getitem__(
        self, i: int
    ) -> (
        tuple[torch.Tensor, ...]
    ):
        if self.storage_mode == "eager":
            assert self.eager_compact_boards is not None
            assert self.eager_stms is not None
            assert self.eager_targets is not None
            assert self.eager_plies is not None
            assert self.eager_aux is not None
            assert self.eager_result_red is not None
            assert self.eager_ply_total is not None
            assert self.eager_legal_flat is not None
            assert self.eager_legal_offsets is not None
            b90 = self.eager_compact_boards[i]
            stm = int(self.eager_stms[i])
            ti = int(self.eager_targets[i])
            ply_i = int(self.eager_plies[i])
            atk0 = float(self.eager_aux[i, 0])
            dan0 = float(self.eager_aux[i, 1])
            tac0 = float(self.eager_aux[i, 2])
            gr_i = int(self.eager_result_red[i])
            pt_i = int(self.eager_ply_total[i])
            lo = int(self.eager_legal_offsets[i])
            hi = int(self.eager_legal_offsets[i + 1])
            idxs = self.eager_legal_flat[lo:hi]
            fen0 = None
        else:
            ref = self.row_refs[i]
            fen0, idxs, ti, atk0, dan0, tac0 = read_row_train_at(
                self._get_shard_buf(ref.shard_index), ref.row_offset
            )
            ply_i = int(ref.ply)
            gr_i = ref.game_result_red
            pt_i = ref.ply_total

        if fen0 is not None:
            b90, stm_i = _fen_to_compact_cached(fen0)
            stm = stm_i
            board = compact_board_to_torch_planes(b90, stm)
        else:
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

        if self.with_value_labels:
            if fen0 is not None:
                v0 = value_target_side_to_move(
                    fen0,
                    gr_i,
                    ply_i,
                    pt_i,
                    progress_gamma=self._value_progress_gamma,
                )
            else:
                # eager 模式下不再保留 FEN 字符串；当前行棋方直接由 stm 决定。
                base = float(gr_i if stm == 1 else -gr_i) if gr_i in (-1, 1) else 0.0
                if pt_i <= 1:
                    progress = 1.0
                else:
                    progress = min(max(float(ply_i) / float(pt_i - 1), 0.0), 1.0)
                v0 = base * (progress ** float(self._value_progress_gamma))
        else:
            v0 = 0.0

        w0 = 1.0
        if self.position_weights is not None:
            w0 = self.position_weights[i]

        human0 = self._idx_to_move[ti]

        use_mirror = (
            fen0 is not None
            and self.aug_mirror_p > 0.0
            and random.random() < self.aug_mirror_p
        )
        if use_mirror:
            try:
                mh = mirror_move_uci(human0)
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
                            atk_m, dan_m, tac_m = atk0, dan0, tac0
                            v_m = (
                                value_target_side_to_move(
                                    fen_m,
                                    gr_i,
                                    ply_i,
                                    pt_i,
                                    progress_gamma=self._value_progress_gamma,
                                )
                                if self.with_value_labels
                                else 0.0
                            )
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
