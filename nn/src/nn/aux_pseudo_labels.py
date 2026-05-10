"""训练侧伪标签：与 ``crates/xiangqi_dataset/src/aux_labels.rs`` 语义对齐（FEN + pyffish 回退路径）。

主数据应由 Rust 预计算写入 XRSH；本模块仅在缺 aux 字段时调用。
"""

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

_CORE_MOVE = re.compile(r"^([a-i])([0-9])([a-i])([0-9])([a-z])?$")
_PYFFISH_MOVE = re.compile(r"^([a-i])(10|[1-9])([a-i])(10|[1-9])([a-z])?$")

# 全局计数：pyffish is_capture 失败 → FEN 回退 → 仍失败 的次数；供 DataLoader 侧诊断
_pyffish_capture_fallback_count = 0
_pyffish_capture_total_calls = 0


def pyffish_capture_diag() -> tuple[int, int]:
    return _pyffish_capture_fallback_count, _pyffish_capture_total_calls


def core_uci_to_pyffish(uci: str) -> str:
    m = _CORE_MOVE.match(uci.strip().lower())
    if not m:
        raise ValueError(f"非标准 UCI: {uci!r}")

    def enc_rank(r: int) -> str:
        return "10" if r == 9 else str(r + 1)

    r1 = int(m.group(2))
    r2 = int(m.group(4))
    s = f"{m.group(1)}{enc_rank(r1)}{m.group(3)}{enc_rank(r2)}"
    if m.group(5):
        s += m.group(5)
    return s


def _material_red_black(board_field: str) -> tuple[float, float]:
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


def _rank_from_cell_index(i: int) -> int:
    """纵坐标 0..9，与 xiangqi_core 一致（0=红方底线）。"""
    return 9 - (i // 9)


def _cheb_cell(i: int, j: int) -> int:
    return max(abs(i // 9 - j // 9), abs(i % 9 - j % 9))


def _uci_dest_fen_index(uci: str) -> int | None:
    m = re.fullmatch(r"([a-i])([0-9])([a-i])([0-9])([a-z])?", uci.strip().lower())
    if not m:
        return None
    file = ord(m.group(3)) - ord("a")
    fen_rank = int(m.group(4))
    if fen_rank > 9 or not (0 <= file <= 8):
        return None
    return (9 - fen_rank) * 9 + file


def _fen_dest_piece(fen: str, uci: str) -> str | None:
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


def _is_capture_safe(fen: str, base: str, prefix_pf: list[str], uci: str) -> bool:
    global _pyffish_capture_fallback_count, _pyffish_capture_total_calls
    _pyffish_capture_total_calls += 1
    u_pf = core_uci_to_pyffish(uci)
    try:
        return bool(sf.is_capture(VARIANT, base, prefix_pf, u_pf))
    except (ValueError, SystemError, RuntimeError):
        pass

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


def _in_check_sf(base: str, prefix_pf: list[str]) -> bool:
    try:
        return bool(sf.gives_check(VARIANT, base, prefix_pf))
    except (ValueError, SystemError, RuntimeError):
        return False


def _gives_check_move(base: str, prefix_pf: list[str], uci: str) -> bool:
    u_pf = core_uci_to_pyffish(uci)
    try:
        return bool(sf.gives_check(VARIANT, base, prefix_pf, u_pf))
    except (ValueError, SystemError, RuntimeError):
        return False


def pyffish_uci_to_core(uci: str) -> str:
    m = _PYFFISH_MOVE.match(uci.strip().lower())
    if not m:
        return uci

    def dec_rank(rs: str) -> int:
        return 10 if rs == "10" else int(rs)

    r1 = dec_rank(m.group(2)) - 1
    r2 = dec_rank(m.group(4)) - 1
    core = f"{m.group(1)}{r1}{m.group(3)}{r2}"
    if m.group(5):
        core += m.group(5)
    return core


def _king_exposure_proxy(cells: list[str], *, stm_white: bool) -> float:
    our_k, enemy_k = ("K", "k") if stm_white else ("k", "K")
    try:
        i_our = cells.index(our_k)
    except ValueError:
        return 0.0
    n_enemy_near = 0
    for i, ch in enumerate(cells):
        if ch in (".", ""):
            continue
        enemy = ch.islower() if stm_white else ch.isupper()
        if not enemy or ch in ("K", "k"):
            continue
        if _cheb_cell(i, i_our) <= 2:
            n_enemy_near += 1
    return min(1.0, n_enemy_near / 6.0)


def _attack_pressure_proxy(cells: list[str], *, stm_white: bool) -> tuple[float, float, float]:
    """(威胁老将邻域, 过河兵比例, 深入对方半场子力) 均 ∈ [0,1]。"""
    our_k, enemy_k = ("K", "k") if stm_white else ("k", "K")
    try:
        i_ek = cells.index(enemy_k)
    except ValueError:
        return 0.0, 0.0, 0.0

    threat = 0
    for i, ch in enumerate(cells):
        if ch in (".", ""):
            continue
        ours = ch.isupper() if stm_white else ch.islower()
        if not ours or ch in ("K", "k"):
            continue
        if _cheb_cell(i, i_ek) <= 2:
            threat += 1
    threat_k = min(1.0, threat / 8.0)

    pawns = 0
    crossed = 0
    for i, ch in enumerate(cells):
        if stm_white and ch == "P":
            pawns += 1
            if _rank_from_cell_index(i) >= 5:
                crossed += 1
        if not stm_white and ch == "p":
            pawns += 1
            if _rank_from_cell_index(i) <= 4:
                crossed += 1
    crossed_ratio = 0.0 if pawns == 0 else crossed / float(pawns)

    deep = 0
    for i, ch in enumerate(cells):
        if ch in (".", ""):
            continue
        ours = ch.isupper() if stm_white else ch.islower()
        if not ours or ch in ("K", "k"):
            continue
        r = _rank_from_cell_index(i)
        on_enemy = r >= 5 if stm_white else r <= 4
        if on_enemy:
            deep += 1
    half_norm = min(1.0, deep / 12.0)

    return threat_k, crossed_ratio, half_norm


def pseudo_aux_labels_from_sample(
    fen: str,
    *,
    root_fen: str | None = None,
    uci_prefix: Sequence[str] | None = None,
    legal_uci: Sequence[str] | None = None,
) -> tuple[float, float, float]:
    """
    返回 attack / danger / tactical，与 Rust ``pseudo_aux_labels`` 同语义（本实现为几何+pyffish 近似）。

    - **legal_uci**：若提供，统计仅在该集合上（与 XRSH ``legal_idx`` 一致）。
    """
    base = root_fen if root_fen else fen
    prefix = list(uci_prefix or [])
    prefix_pf = [core_uci_to_pyffish(m) for m in prefix]

    if legal_uci is not None:
        moves = list(legal_uci)
    else:
        moves = [pyffish_uci_to_core(m) for m in sf.legal_moves(VARIANT, base, prefix_pf)]

    n = len(moves)
    if n < 1:
        return 0.5, 1.0, 0.0

    caps = sum(1 for u in moves if _is_capture_safe(fen, base, prefix_pf, u))
    checks = sum(1 for u in moves if _gives_check_move(base, prefix_pf, u))
    capture_ratio = caps / float(n)
    check_ratio = checks / float(n)
    in_check_bonus = 1.0 if _in_check_sf(base, prefix_pf) else 0.0
    tactical = min(
        1.0,
        max(0.0, 0.5 * capture_ratio + 0.3 * check_ratio + 0.2 * in_check_bonus),
    )

    parts = fen.split()
    stm = parts[1] if len(parts) > 1 else "w"
    stm_white = stm == "w"
    red, black = _material_red_black(parts[0])
    adv = red - black if stm_white else black - red

    danger_check = 1.0 if in_check_bonus else 0.0
    mob_norm = min(1.0, n / 48.0)
    low_mobility = 1.0 - mob_norm
    material_stress = max(0.0, min(1.0, 0.5 * (1.0 + math.tanh(-adv / 12.0))))

    try:
        cells = _fen_board_to_flat(parts[0])
        king_exposure = _king_exposure_proxy(cells, stm_white=stm_white)
        threat_k, crossed_ratio, half_norm = _attack_pressure_proxy(
            cells, stm_white=stm_white
        )
    except ValueError:
        king_exposure = 0.0
        threat_k, crossed_ratio, half_norm = 0.0, 0.0, 0.0

    danger = max(
        0.0,
        min(
            1.0,
            0.35 * danger_check
            + 0.30 * low_mobility
            + 0.20 * material_stress
            + 0.15 * king_exposure,
        ),
    )

    attack = max(
        0.0,
        min(1.0, 0.45 * threat_k + 0.35 * crossed_ratio + 0.20 * half_norm),
    )

    return attack, danger, tactical


def pseudo_aux_labels_from_fen(fen: str) -> tuple[float, float, float]:
    return pseudo_aux_labels_from_sample(fen)
