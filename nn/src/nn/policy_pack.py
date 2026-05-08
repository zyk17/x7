"""离线 policy 训练包（mmap）：紧凑棋盘 + 稀疏合法着下标 + 定长当前局 FEN 字节列。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# 定长 UTF-8 列（零填充）；须与物化脚本一致（仅当前局 ``fen``；``root``/``uci_prefix`` 训练步不用故不落盘）
FEN_BYTES = 128

BOARD90 = "board90.npy"
STM = "stm.npy"
LEGAL_PTR = "legal_ptr.npy"
LEGAL_IDX = "legal_idx.npy"
FEN_U8 = "fen_u8.npy"
TARGETS = "targets.npy"
WEIGHTS = "weights.npy"
PLIES = "plies.npy"
SRC_IDS = "src_ids.npy"
PGN_SRC_VOCAB_JSON = "pgn_source_vocab.json"
SAMPLER_ORDER = "sampler_order.npy"
SAMPLER_SEG_PTR = "sampler_seg_ptr.npy"
PACK_META = "pack_meta.json"

_PACK_FILES = (
    BOARD90,
    STM,
    LEGAL_PTR,
    LEGAL_IDX,
    FEN_U8,
    TARGETS,
    WEIGHTS,
    PLIES,
    SRC_IDS,
    PGN_SRC_VOCAB_JSON,
    SAMPLER_ORDER,
    SAMPLER_SEG_PTR,
    PACK_META,
)


def pack_dir_is_complete(pack_dir: Path) -> bool:
    d = Path(pack_dir)
    return all((d / f).is_file() for f in _PACK_FILES)


def load_pack_meta(pack_dir: Path) -> dict[str, Any]:
    return json.loads((Path(pack_dir) / PACK_META).read_text(encoding="utf-8"))


def vocab_fingerprint_ordered_moves(moves: list[str]) -> str:
    """与物化时一致：按下标 0..V-1 顺序，将各着法 UTF-8 与 NUL 拼接后 SHA-256。"""
    h = hashlib.sha256()
    for m in moves:
        h.update(m.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def moves_list_from_move_to_idx(move_to_idx: dict[str, int]) -> list[str]:
    n = len(move_to_idx)
    out = [""] * n
    for mv, j in move_to_idx.items():
        out[j] = mv
    return out


def assert_vocab_matches_pack(meta: dict[str, Any], move_to_idx: dict[str, int]) -> None:
    """防止「词表长度相同但着法顺序/内容不同」导致下标语义错位。"""
    exp = meta.get("vocab_sha256")
    if not exp:
        raise ValueError(
            "训练包 pack_meta 缺少 vocab_sha256（为过旧格式），请用当前仓库重新 materialize_policy_pack"
        )
    got = vocab_fingerprint_ordered_moves(moves_list_from_move_to_idx(move_to_idx))
    if got != exp:
        raise ValueError(
            "词表与训练包不一致：请使用建包时同一 move_vocab.json（着法顺序与内容须完全一致）"
        )


def pad_utf8_bytes(s: str, max_len: int) -> bytes:
    b = s.encode("utf-8")
    if len(b) > max_len:
        raise ValueError(f"UTF-8 长度 {len(b)} 超过上限 {max_len}: {s[:60]!r}…")
    return b + b"\0" * (max_len - len(b))
