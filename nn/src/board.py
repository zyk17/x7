from __future__ import annotations

import pyffish as sf

from constants import VARIANT


def legal_moves_uci(fen: str, uci_prefix: list[str] | None = None) -> list[str]:
    """当前局面合法 UCI 着法列表。`uci_prefix` 为从初始局面走到当前局面的 UCI 序列。"""
    movelist = list(uci_prefix or [])
    return list(sf.legal_moves(VARIANT, fen, movelist))


def apply_uci(fen: str, uci: str, uci_prefix: list[str] | None = None) -> str:
    """在 `fen + uci_prefix` 上走一步 `uci`，返回新 FEN。"""
    movelist = list(uci_prefix or [])
    movelist.append(uci)
    return sf.get_fen(VARIANT, fen, movelist)


def validate_fen(fen: str) -> int:
    return int(sf.validate_fen(fen, VARIANT))
