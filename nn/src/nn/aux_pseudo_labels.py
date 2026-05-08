"""训练侧伪标签：由 FEN + pyffish 生成的 attack / danger / tactical 标量（无人工标注）。"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import pyffish as sf

from constants import VARIANT

# 与子力常见分值同量级，仅用于局面强度启发（非引擎估值）
_PIECE_MATERIAL: dict[str, float] = {
    "R": 9.0,
    "N": 2.0,
    "B": 2.0,
    "A": 2.0,
    "C": 4.5,
    "P": 1.0,
    "K": 0.0,
    "r": 9.0,
    "n": 2.0,
    "b": 2.0,
    "a": 2.0,
    "c": 4.5,
    "p": 1.0,
    "k": 0.0,
}

# 全局计数：pyffish is_capture 失败 → FEN 回退 → 仍失败 的次数；供 DataLoader 侧诊断
_pyffish_capture_fallback_count = 0
_pyffish_capture_total_calls = 0


def pyffish_capture_diag() -> tuple[int, int]:
    return _pyffish_capture_fallback_count, _pyffish_capture_total_calls


def _material_red_black(board_field: str) -> tuple[float, float]:
    """FEN 棋盘段：红方（大写）与黑方（小写）子力分值和（不含将/帅）。"""
    red = 0.0
    black = 0.0
    for ch in board_field:
        v = _PIECE_MATERIAL.get(ch)
        if v is None:
            continue
        if ch.isupper():
            red += v
        else:
            black += v
    return red, black


def _fen_board_to_flat(fen_board: str) -> list[str]:
    """将 FEN 棋盘字段展开为逐格列表（行主序，rank 9 → rank 0，col a → i）。"""
    ranks = fen_board.split("/")
    if len(ranks) != 10:
        raise ValueError(f"期望 10 行棋盘，得到 {len(ranks)}")
    cells: list[str] = []
    for rank_str in ranks:
        for ch in rank_str:
            if ch.isdigit():
                cells.extend(["."] * int(ch))
            else:
                cells.append(ch)
        if len(cells) % 9 != 0:
            raise ValueError(f"行 {rank_str!r} 展开后非 9 的倍数")
    return cells


def _uci_dest_fen_index(uci: str) -> int | None:
    """将 pyffish UCI（如 ``b2b6``）的目的格转为 0..89 的 FEN 棋盘索引，失败返回 None。"""
    m = re.fullmatch(r"([a-i])(\d+)([a-i])(\d+)", uci.lower())
    if not m:
        return None
    file = ord(m.group(3)) - ord("a")
    pyr = int(m.group(4))  # pyffish rank 1..10
    if pyr < 1 or pyr > 10:
        return None
    fen_rank = pyr - 1  # FEN rank 0..9（0=底）
    return (9 - fen_rank) * 9 + file  # row-major: rank 9 在最前


def _fen_dest_piece(fen: str, uci: str) -> str | None:
    """从当前 fen 读取 uci 目标格的棋子符号；失败返回 None。"""
    idx = _uci_dest_fen_index(uci)
    if idx is None:
        return None
    parts = fen.split(None, 2)
    if len(parts) < 1:
        return None
    try:
        cells = _fen_board_to_flat(parts[0])
    except ValueError:
        return None
    if not (0 <= idx < len(cells)):
        return None
    return cells[idx]


def _is_capture_safe(fen: str, base: str, prefix: list[str], uci: str) -> bool:
    global _pyffish_capture_fallback_count, _pyffish_capture_total_calls
    _pyffish_capture_total_calls += 1
    try:
        return bool(sf.is_capture(VARIANT, base, prefix, uci))
    except (ValueError, SystemError, RuntimeError):
        pass

    # pyffish 失败：回退到 FEN 检查（目标格是否有对方棋子）
    ch = _fen_dest_piece(fen, uci)
    if ch is not None and ch not in (".", ""):
        stm = fen.split()[1] if len(fen.split()) > 1 else "w"
        is_white_side = stm == "w"
        enemy_lower = not is_white_side and ch.islower()
        enemy_upper = is_white_side and ch.isupper()
        if enemy_lower or enemy_upper:
            _pyffish_capture_fallback_count += 1
            return True
    _pyffish_capture_fallback_count += 1
    return False


def pseudo_aux_labels_from_sample(
    fen: str,
    *,
    root_fen: str | None = None,
    uci_prefix: Sequence[str] | None = None,
    legal_uci: Sequence[str] | None = None,
) -> tuple[float, float, float]:
    """
    返回三维伪标签，取值约在 [0, 1]，可与 sigmoid 回归头配合。

    - **base / prefix**：须与分片编码时一致（``pyffish.legal_moves(VARIANT, root_fen, uci_prefix)``），
      以计入长将 / 重复等路径依赖约束；若缺省则退化为 ``root_fen=fen``、空 prefix（与旧行为一致）。
    - **legal_uci**：若提供，**机动性 / 吃子比例** 仅在该集合上统计（与 XRSH 物化的 ``legal_idx`` 一致，
      减少与 policy mask 的合法集分裂；仍用 pyffish 判定各着是否吃子）。

    物质项 **attack** 仍从当前局面 ``fen`` 的棋盘段读取（与行内 ``fen`` 一致）。

    注意：policy 目标来自 **Rust** 合法集，本函数在吃子判定时仍调 **pyffish**；若两引擎规则有差异，
    可能仍有静默不一致，长期应用 ``xiangqi_core`` 在数据侧预计算伪标签以统一规则源（见 ARCHITECTURE）。
    """
    base = root_fen if root_fen else fen
    prefix = list(uci_prefix or [])

    if legal_uci is not None:
        moves = list(legal_uci)
    else:
        moves = list(sf.legal_moves(VARIANT, base, prefix))

    n = len(moves)
    if n < 1:
        return 0.5, 1.0, 0.0

    caps = sum(1 for u in moves if _is_capture_safe(fen, base, prefix, u))
    tactical = caps / float(n)

    parts = fen.split()
    stm = parts[1] if len(parts) > 1 else "w"
    red, black = _material_red_black(parts[0])
    if stm == "w":
        adv = red - black
    else:
        adv = black - red

    attack = 0.5 * (1.0 + math.tanh(adv / 12.0))

    mob_norm = min(1.0, n / 48.0)
    danger_from_moves = 1.0 - mob_norm
    mat_stress = 0.5 * (1.0 + math.tanh(-adv / 12.0))
    danger = max(0.0, min(1.0, 0.55 * danger_from_moves + 0.45 * mat_stress))

    return attack, danger, tactical


def pseudo_aux_labels_from_fen(fen: str) -> tuple[float, float, float]:
    """兼容旧调用：无根局面、无走子历史、不绑定 Rust 合法表。"""
    return pseudo_aux_labels_from_sample(fen)
