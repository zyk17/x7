"""JSONL 行级索引（mmap）：大语料不把所有行 dict 驻留内存。"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

ROW_START = "row_start.npy"
ROW_END = "row_end.npy"
GAME_IDX = "game_idx.npy"
WEIGHTS = "weights.npy"
TARGETS = "targets.npy"
PLIES = "plies.npy"
SRC_IDS = "src_ids.npy"
GAMES_JSON = "games.json"
PGN_SRC_VOCAB_JSON = "pgn_source_vocab.json"
# 按局组 batch：行下标的稳定排序 + 每局在排序轴上的 [ptr[k], ptr[k+1])
SAMPLER_ORDER = "sampler_order.npy"
SAMPLER_SEG_PTR = "sampler_seg_ptr.npy"


def _write_sampler_files(gix: np.ndarray, out_dir: Path) -> None:
    """在 memmap 落盘关闭前用内存快照生成按局采样文件，避免 Windows 上立即再 open 同一路径失败。"""
    gix = np.asarray(gix, dtype=np.int32, order="C")
    order = np.argsort(gix, kind="stable").astype(np.int32)
    sg = gix[order]
    idx = np.flatnonzero(np.r_[True, sg[1:] != sg[:-1]])
    ptr = np.empty(len(idx) + 1, dtype=np.int64)
    ptr[:-1] = idx.astype(np.int64, copy=False)
    ptr[-1] = np.int64(len(sg))
    np.save(out_dir / SAMPLER_ORDER, order)
    np.save(out_dir / SAMPLER_SEG_PTR, ptr)


def _count_valid_and_maybe_fen(
    jsonl_path: Path,
    move_to_idx: dict[str, int],
    *,
    weight_by_fen: bool,
) -> tuple[int, Counter[str] | None]:
    """单遍扫描：有效样本数 N；若 weight_by_fen 则同时统计 fen 频数（仍占「不同 fen」量级内存）。"""
    fen_counts: Counter[str] | None = Counter() if weight_by_fen else None
    n = 0
    with jsonl_path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            mv = row.get("human_move_pyffish")
            if not mv or mv not in move_to_idx:
                continue
            n += 1
            if weight_by_fen and fen_counts is not None:
                fen = row.get("fen", "")
                if fen:
                    fen_counts[fen] += 1
    return n, fen_counts


def build_jsonl_index(
    jsonl_path: Path,
    move_to_idx: dict[str, int],
    out_dir: Path,
    *,
    weight_by_fen: bool,
) -> int:
    """
    流式扫描 JSONL，写出 mmap 用辅助数组；原 JSONL 路径不变。
    weight_by_fen：训练集 True（weights=1/sqrt(fen 次数)，fen 计数仍驻内存）；
    验证集 False（单遍计数 + 单遍写入，全 1 权重）。
    主数组按行直接写入 np.memmap，避免 N 行 Python list 爆内存。
    返回有效样本行数 N。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n, fen_counts = _count_valid_and_maybe_fen(
        jsonl_path, move_to_idx, weight_by_fen=weight_by_fen
    )
    if weight_by_fen:
        assert fen_counts is not None
    else:
        assert fen_counts is None

    if n == 0:
        raise RuntimeError(f"无有效样本: {jsonl_path}")

    path_rs = out_dir / ROW_START
    path_re = out_dir / ROW_END
    path_gix = out_dir / GAME_IDX
    path_w = out_dir / WEIGHTS
    path_t = out_dir / TARGETS
    path_p = out_dir / PLIES
    path_s = out_dir / SRC_IDS

    mm_rs = open_memmap(path_rs, mode="w+", dtype=np.int64, shape=(n,))
    mm_re = open_memmap(path_re, mode="w+", dtype=np.int64, shape=(n,))
    mm_gix = open_memmap(path_gix, mode="w+", dtype=np.int32, shape=(n,))
    mm_w = open_memmap(path_w, mode="w+", dtype=np.float32, shape=(n,))
    mm_t = open_memmap(path_t, mode="w+", dtype=np.int32, shape=(n,))
    mm_p = open_memmap(path_p, mode="w+", dtype=np.int32, shape=(n,))
    mm_s = open_memmap(path_s, mode="w+", dtype=np.int32, shape=(n,))

    game_ids_order: list[str] = []
    gid_to_i: dict[str, int] = {}
    src_strings_set: set[str] = set()
    src_to_i: dict[str, int] = {}

    def pgn_src_id(s: str) -> int:
        if s not in src_to_i:
            i = len(src_to_i)
            src_to_i[s] = i
            src_strings_set.add(s)
            return i
        return src_to_i[s]

    i = 0
    with jsonl_path.open("rb") as f:
        while True:
            off_start = f.tell()
            line = f.readline()
            if not line:
                break
            off_end = f.tell()
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            mv = row.get("human_move_pyffish")
            if not mv or mv not in move_to_idx:
                continue
            ti = move_to_idx[mv]
            gid = str(row.get("game_id", "") or "")
            if gid not in gid_to_i:
                gid_to_i[gid] = len(game_ids_order)
                game_ids_order.append(gid)
            gix = gid_to_i[gid]
            fen = row.get("fen", "")
            if weight_by_fen and fen_counts is not None:
                c = max(1, fen_counts.get(fen, 1))
                w = 1.0 / math.sqrt(float(c))
            else:
                w = 1.0
            ply = int(row.get("ply", 0) or 0)
            ps = str(row.get("pgn_source", "") or "")
            six = pgn_src_id(ps)

            mm_rs[i] = off_start
            mm_re[i] = off_end
            mm_gix[i] = gix
            mm_w[i] = w
            mm_t[i] = ti
            mm_p[i] = ply
            mm_s[i] = six
            i += 1

    if i != n:
        raise RuntimeError(f"索引行数不一致: 计数 {n} 写入 {i}")

    (out_dir / GAMES_JSON).write_text(
        json.dumps(game_ids_order, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    src_vocab = sorted(src_strings_set, key=lambda s: src_to_i[s])
    (out_dir / PGN_SRC_VOCAB_JSON).write_text(
        json.dumps(src_vocab, ensure_ascii=False, indent=0), encoding="utf-8"
    )

    gix_for_sampler = np.asarray(mm_gix, dtype=np.int32).copy()
    for mm in (mm_rs, mm_re, mm_gix, mm_w, mm_t, mm_p, mm_s):
        mm.flush()
        del mm

    _write_sampler_files(gix_for_sampler, out_dir)

    return n


def index_dir_is_complete(index_dir: Path) -> bool:
    d = Path(index_dir)
    return all(
        (d / f).is_file()
        for f in (
            ROW_START,
            ROW_END,
            GAME_IDX,
            WEIGHTS,
            TARGETS,
            PLIES,
            SRC_IDS,
            GAMES_JSON,
            PGN_SRC_VOCAB_JSON,
        )
    )


def index_sampler_is_complete(index_dir: Path) -> bool:
    """训练 DataLoader 按局组 batch 所需的稳定排序与段指针（由 build_jsonl_index 写出）。"""
    d = Path(index_dir)
    return index_dir_is_complete(d) and (d / SAMPLER_ORDER).is_file() and (d / SAMPLER_SEG_PTR).is_file()
