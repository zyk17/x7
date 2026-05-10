"""水平镜像：棋盘 + 四字符着法 UCI（仅交换纵线 a↔i …），不改变先后手语义。"""

from __future__ import annotations

_FILES = "abcdefghi"
_FILE_TR = str.maketrans(_FILES, "ihgfedcba")


def mirror_uci_file(ch: str) -> str:
    if ch not in _FILES:
        raise ValueError(f"非纵线字符: {ch!r}")
    return ch.translate(_FILE_TR)


def mirror_move_uci(u: str) -> str:
    """四字符 UCI 着法，例如 c4c5 → g4g5（仅镜像纵线）。"""
    if len(u) != 4:
        raise ValueError(f"期望 4 字符着法，得到 {len(u)}: {u!r}")
    return (
        mirror_uci_file(u[0])
        + u[1]
        + mirror_uci_file(u[2])
        + u[3]
    )


def _expand_rank(rank_str: str) -> list[str]:
    cells: list[str] = []
    for ch in rank_str:
        if ch.isdigit():
            cells.extend(["."] * int(ch))
        else:
            cells.append(ch)
    return cells


def _compress_rank(cells: list[str]) -> str:
    if len(cells) != 9:
        raise ValueError(f"期望 9 格，得到 {len(cells)}")
    parts: list[str] = []
    i = 0
    while i < 9:
        if cells[i] != ".":
            parts.append(cells[i])
            i += 1
        else:
            n = 0
            while i < 9 and cells[i] == ".":
                n += 1
                i += 1
            parts.append(str(n))
    return "".join(parts)


def mirror_board_fen_field(board_field: str) -> str:
    """只镜像 FEN 第一段（10 行以 / 分隔）。"""
    ranks = board_field.split("/")
    if len(ranks) != 10:
        raise ValueError(f"期望 10 行棋盘，得到 {len(ranks)}")
    out_ranks: list[str] = []
    for r in ranks:
        cells = _expand_rank(r)
        if len(cells) != 9:
            raise ValueError(f"行长度应为 9，得到 {len(cells)}: {r!r}")
        out_ranks.append(_compress_rank(list(reversed(cells))))
    return "/".join(out_ranks)


def mirror_fen(fen: str) -> str:
    """镜像完整 FEN 字符串的棋盘段，其余字段不变。"""
    parts = fen.split()
    if not parts:
        raise ValueError("空 FEN")
    parts[0] = mirror_board_fen_field(parts[0])
    return " ".join(parts)


def mirror_uci_prefix(prefix: list[str]) -> list[str]:
    return [mirror_move_uci(m) for m in prefix]
