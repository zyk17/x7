"""词表指纹（与 XRSH ``pack_meta.json`` / Rust 侧一致）。"""

from __future__ import annotations

import hashlib
from typing import Any


def vocab_fingerprint_ordered_moves(moves: list[str]) -> str:
    """按下标 0..V-1 顺序，将各着法 UTF-8 与 NUL 拼接后 SHA-256。"""
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
            "pack_meta 缺少 vocab_sha256；请使用与词表一致的 XRSH 目录"
        )
    got = vocab_fingerprint_ordered_moves(moves_list_from_move_to_idx(move_to_idx))
    if got != exp:
        raise ValueError(
            "词表与数据包不一致：请使用建包时同一 move_vocab.json（着法顺序与内容须完全一致）"
        )
