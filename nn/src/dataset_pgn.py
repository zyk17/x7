"""从 ParsedGame 生成训练行（position → move）"""

from __future__ import annotations

import board as xb
from constants import START_FEN
from notation_iccs import iccs_move_to_pyffish, pyffish_move_to_iccs
from pgn import (
    ParsedGame,
    movetext_iccs_pairs,
    movetext_uci_tokens,
    pgn_format,
)


def starting_fen(game: ParsedGame) -> str:
    fen = game.headers.get("FEN", "").strip()
    if fen:
        if xb.validate_fen(fen) != 1:
            raise ValueError(f"无效 FEN: {fen[:80]}")
        return fen
    return START_FEN


def side_to_move_from_fen(fen: str) -> str:
    parts = fen.split()
    return parts[1] if len(parts) > 1 else "w"


def _moves_for_game(game: ParsedGame) -> tuple[list[str], str]:
    """
    返回 (pyffish_uci 列表, 记谱格式说明)。
    默认：有 ICCS 形 `c3-c4` 则按 ICCS；否则尝试已是 pyffish 的四字母 UCI。
    """
    fmt = pgn_format(game.headers)
    text = game.movetext_raw

    iccs_pairs = movetext_iccs_pairs(text)
    if iccs_pairs and fmt != "UCI":
        py_moves = [iccs_move_to_pyffish(p) for p in iccs_pairs]
        return py_moves, fmt or "ICCS"

    uci_toks = movetext_uci_tokens(text)
    if uci_toks:
        return uci_toks, fmt or "UCI"

    if iccs_pairs:
        py_moves = [iccs_move_to_pyffish(p) for p in iccs_pairs]
        return py_moves, "ICCS"

    return [], fmt


_PGN_HEADER_KEYS = (
    "White",
    "Black",
    "Event",
    "Date",
    "Result",
    "TimeControl",
)


def _header_fields(game: ParsedGame) -> dict[str, str]:
    h = game.headers
    out: dict[str, str] = {}
    for key in _PGN_HEADER_KEYS:
        v = h.get(key, "").strip()
        if v:
            out[key] = v
    return out


def iter_training_rows(game: ParsedGame, *, game_id: str) -> list[dict]:
    """
    生成一局内每条 position→move 样本。
    保留 root_fen + uci_prefix，以便 pyffish 计算合法走法（重复局面等规则）。
    """
    py_moves, fmt = _moves_for_game(game)
    if not py_moves:
        return []

    fen0 = starting_fen(game)
    rows: list[dict] = []
    prefix: list[str] = []
    cur_fen = fen0
    meta = _header_fields(game)

    for ply, py_uci in enumerate(py_moves):
        legal = xb.legal_moves_uci(fen0, prefix)
        if py_uci not in legal:
            break

        stm = side_to_move_from_fen(cur_fen)
        engine_uci = pyffish_move_to_iccs(py_uci).replace("-", "")
        row = {
            "fen": cur_fen,
            "root_fen": fen0,
            "uci_prefix": list(prefix),
            "human_move_pyffish": py_uci,
            "human_move_uci": engine_uci,
            "ply": ply,
            "side_to_move": stm,
            "pgn_format": fmt,
            "game_id": game_id,
        }
        row.update(meta)
        rows.append(row)

        next_fen = xb.apply_uci(fen0, py_uci, prefix)
        prefix.append(py_uci)
        cur_fen = next_fen

    return rows
