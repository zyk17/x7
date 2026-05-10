from __future__ import annotations

import re

import pyffish as sf

from constants import VARIANT

_CORE_MOVE = re.compile(r"^([a-i])([0-9])([a-i])([0-9])([a-z])?$")
_PYFFISH_MOVE = re.compile(r"^([a-i])(10|[1-9])([a-i])(10|[1-9])([a-z])?$")


def core_uci_to_pyffish(uci: str) -> str:
    m = _CORE_MOVE.match(uci.strip().lower())
    if not m:
        raise ValueError(f"非标准 UCI（期望 a0~i9）: {uci!r}")

    def enc_rank(r: int) -> str:
        return "10" if r == 9 else str(r + 1)

    r1 = int(m.group(2))
    r2 = int(m.group(4))
    out = f"{m.group(1)}{enc_rank(r1)}{m.group(3)}{enc_rank(r2)}"
    if m.group(5):
        out += m.group(5)
    return out


def pyffish_uci_to_core(uci: str) -> str:
    m = _PYFFISH_MOVE.match(uci.strip().lower())
    if not m:
        raise ValueError(f"无法解析 pyffish UCI: {uci!r}")

    def dec_rank(rs: str) -> int:
        pr = 10 if rs == "10" else int(rs)
        return pr - 1

    r1 = dec_rank(m.group(2))
    r2 = dec_rank(m.group(4))
    out = f"{m.group(1)}{r1}{m.group(3)}{r2}"
    if m.group(5):
        out += m.group(5)
    return out


def legal_moves_uci(fen: str, uci_prefix: list[str] | None = None) -> list[str]:
    """当前局面合法 UCI 着法列表（标准 UCI：`a0`~`i9`）。"""
    movelist = [core_uci_to_pyffish(m) for m in list(uci_prefix or [])]
    return [pyffish_uci_to_core(m) for m in sf.legal_moves(VARIANT, fen, movelist)]


def apply_uci(fen: str, uci: str, uci_prefix: list[str] | None = None) -> str:
    """在 `fen + uci_prefix` 上走一步标准 UCI，返回新 FEN。"""
    movelist = [core_uci_to_pyffish(m) for m in list(uci_prefix or [])]
    movelist.append(core_uci_to_pyffish(uci))
    return sf.get_fen(VARIANT, fen, movelist)


def validate_fen(fen: str) -> int:
    return int(sf.validate_fen(fen, VARIANT))
