"""
ICCS 纵线记谱（与 77 象棋 / 皮卡鱼 UCI 一致：小写 a–i，横线 0–9，a0 = 红方左下角）
→ pyffish / Fairy-Stockfish 内部 UCI：纵线仍 a–i，纵坐标为「ICCS 位上的数字 + 1」（故可有 10）。

参见：`77xiangqi` 中 `notation/importer.rs` 的 `idx_to_uci_half` 与 `engine/uci_ucci_engine.rs` 坐标说明。
"""

from __future__ import annotations

import re

_ICCS_HALF = re.compile(r"^([A-Ia-i])(\d+)$")
_ICCS_MOVE_COMPACT = re.compile(r"^([A-Ia-i]\d+)([A-Ia-i]\d+)$")


def iccs_half_to_pyffish(half: str) -> str:
    """单格 ICCS（如 `C3`、`a0`）→ pyffish 半串（如 `c4`、`a1`）。"""
    s = half.strip()
    m = _ICCS_HALF.match(s)
    if not m:
        raise ValueError(f"非法 ICCS 半格: {half!r}")
    f = m.group(1).lower()
    r = int(m.group(2), 10)
    return f"{f}{r + 1}"


def iccs_move_to_pyffish(move: str) -> str:
    """`C3-C4` / `c3-c4` / `c3c4` / `c10e7`（紧凑）→ pyffish UCI，如 `c4c5`、`c10e8`。"""
    t = move.strip().replace(" ", "")
    if "-" in t:
        a, b = t.split("-", 1)
    else:
        m = _ICCS_MOVE_COMPACT.match(t)
        if not m:
            raise ValueError(f"无法解析 ICCS 着法: {move!r}")
        a, b = m.group(1), m.group(2)
    return iccs_half_to_pyffish(a) + iccs_half_to_pyffish(b)


def pyffish_half_to_iccs(half: str) -> str:
    """pyffish 半串（如 `c4`、`a1`、`c10`）→ ICCS 半串（如 `c3`、`a0`、`c9`）。"""
    m = re.match(r"^([a-i])(\d+)$", half.strip().lower())
    if not m:
        raise ValueError(f"非法 pyffish 半格: {half!r}")
    f = m.group(1)
    r = int(m.group(2), 10)
    if r < 1:
        raise ValueError(f"非法 pyffish 纵坐标: {half!r}")
    return f"{f}{r - 1}"


def split_pyffish_move(uci: str) -> tuple[str, str]:
    """将 pyffish UCI 切成两半（支持 c10e8）。"""
    u = uci.strip().lower()
    m = re.match(r"^([a-i])(\d+)([a-i])(\d+)$", u)
    if not m:
        raise ValueError(f"非法 UCI: {uci!r}")
    return f"{m.group(1)}{m.group(2)}", f"{m.group(3)}{m.group(4)}"


def pyffish_move_to_iccs(uci: str) -> str:
    """pyffish UCI → `c3-c4` 形式 ICCS（小写）。"""
    a, b = split_pyffish_move(uci)
    return f"{pyffish_half_to_iccs(a)}-{pyffish_half_to_iccs(b)}"
