"""JSONL + 行索引 → 离线 policy 训练包（库入口，供脚本与测试调用）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

import board as xb
from nn import jsonl_index as ji
from nn.board_compact import fen_to_compact_board
from nn.policy_pack import (
    BOARD90,
    FEN_BYTES,
    FEN_U8,
    LEGAL_IDX,
    LEGAL_PTR,
    PACK_META,
    STM,
    pad_utf8_bytes,
    vocab_fingerprint_ordered_moves,
)


def _parse_row(jsonl_mm: np.memmap, rs: np.ndarray, re: np.ndarray, i: int) -> dict:
    s, e = int(rs[i]), int(re[i])
    raw = jsonl_mm[s:e].tobytes()
    return json.loads(raw.decode("utf-8").strip())


def materialize_pack(
    jsonl: Path,
    index_dir: Path,
    vocab: Path,
    out_dir: Path,
    *,
    show_progress: bool = True,
) -> tuple[int, int]:
    """物化到 ``out_dir``，返回 ``(N, total_legal)``。"""
    if not ji.index_dir_is_complete(index_dir):
        raise FileNotFoundError(f"索引不完整: {index_dir}")
    if not ji.index_sampler_is_complete(index_dir):
        raise FileNotFoundError(f"索引缺少 sampler 文件，请先 build_jsonl_index: {index_dir}")

    vocab_data = json.loads(vocab.read_text(encoding="utf-8"))
    moves: list[str] = vocab_data["moves"]
    move_to_idx = {m: i for i, m in enumerate(moves)}
    V = len(moves)
    vocab_sha256 = vocab_fingerprint_ordered_moves(moves)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    rs = np.load(index_dir / ji.ROW_START, mmap_mode="r")
    re = np.load(index_dir / ji.ROW_END, mmap_mode="r")
    ti_arr = np.load(index_dir / ji.TARGETS, mmap_mode="r")
    N = int(rs.shape[0])
    if int(re.shape[0]) != N or int(ti_arr.shape[0]) != N:
        raise ValueError("索引长度不一致")

    sz = jsonl.stat().st_size
    jsonl_mm = np.memmap(jsonl, dtype=np.uint8, mode="r", shape=(sz,))

    legal_counts = np.zeros(N, dtype=np.int32)
    for i in tqdm(range(N), desc="pass1 legal sizes", disable=not show_progress):
        row = _parse_row(jsonl_mm, rs, re, i)
        legal = xb.legal_moves_uci(row["root_fen"], list(row["uci_prefix"]))
        ids = [move_to_idx[u] for u in legal if u in move_to_idx]
        if not ids:
            raise RuntimeError(f"行 {i} 无词表内合法着")
        ti = int(ti_arr[i])
        if ti not in ids:
            raise RuntimeError(f"行 {i} 标签不在合法着集合内: ti={ti}")
        legal_counts[i] = len(ids)

    legal_ptr = np.zeros(N + 1, dtype=np.int64)
    legal_ptr[1:] = np.cumsum(legal_counts.astype(np.int64))
    total_legal = int(legal_ptr[-1])
    if total_legal <= 0:
        raise ValueError("total_legal 非法")

    mm_b90 = open_memmap(out_p / BOARD90, mode="w+", dtype=np.uint8, shape=(N, 90))
    mm_stm = open_memmap(out_p / STM, mode="w+", dtype=np.uint8, shape=(N,))
    mm_lp = open_memmap(out_p / LEGAL_PTR, mode="w+", dtype=np.int64, shape=(N + 1,))
    mm_lp[:] = legal_ptr[:]
    mm_li = open_memmap(out_p / LEGAL_IDX, mode="w+", dtype=np.int32, shape=(total_legal,))
    mm_fen = open_memmap(out_p / FEN_U8, mode="w+", dtype=np.uint8, shape=(N, FEN_BYTES))

    for i in tqdm(range(N), desc="pass2 write pack", disable=not show_progress):
        row = _parse_row(jsonl_mm, rs, re, i)
        b90, stm = fen_to_compact_board(row["fen"])
        mm_b90[i] = b90
        mm_stm[i] = stm
        mm_fen[i] = np.frombuffer(pad_utf8_bytes(str(row["fen"]), FEN_BYTES), dtype=np.uint8)

        legal = xb.legal_moves_uci(row["root_fen"], list(row["uci_prefix"]))
        ids = [move_to_idx[u] for u in legal if u in move_to_idx]
        lo, hi = int(legal_ptr[i]), int(legal_ptr[i + 1])
        mm_li[lo:hi] = np.asarray(ids, dtype=np.int32)

    for mm in (mm_b90, mm_stm, mm_lp, mm_li, mm_fen):
        mm.flush()
        del mm
    del jsonl_mm

    for fn in (
        ji.TARGETS,
        ji.WEIGHTS,
        ji.PLIES,
        ji.SRC_IDS,
        ji.SAMPLER_ORDER,
        ji.SAMPLER_SEG_PTR,
        ji.PGN_SRC_VOCAB_JSON,
    ):
        shutil.copy2(index_dir / fn, out_p / fn)

    (out_p / PACK_META).write_text(
        json.dumps(
            {
                "format": "policy_pack_v2",
                "n": N,
                "vocab_size": V,
                "vocab_sha256": vocab_sha256,
                "fen_bytes": FEN_BYTES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return N, total_legal
