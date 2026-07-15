"""孤立 FEN 的 px0 classical `124 x 10 x 9` 输入编码。"""

from __future__ import annotations

import numpy as np
import torch

from constants import START_FEN

_PIECE_PLANES = "RACPNBK"
_ROWS = 10
_COLS = 9
_HISTORY_LENGTH = 8
_PLANES_PER_POSITION = 15
_AUX_BASE = _HISTORY_LENGTH * _PLANES_PER_POSITION
_INPUT_PLANES = _AUX_BASE + 4


def _expand_rank(rank_str: str) -> list[str]:
    cells: list[str] = []
    for ch in rank_str:
        if ch.isdigit():
            cells.extend(["."] * int(ch))
        else:
            cells.append(ch)
    return cells


def _parse_board(fen: str) -> tuple[str, str, int]:
    parts = fen.split()
    if len(parts) < 2:
        raise ValueError(f"FEN must include board and side to move: {fen[:80]}")

    board, stm = parts[0], parts[1]
    if stm not in {"w", "b"}:
        raise ValueError(f"invalid side to move {stm!r}: {fen[:80]}")

    rule60_ply = 0
    if len(parts) >= 5:
        try:
            rule60_ply = int(parts[4])
        except ValueError as exc:
            raise ValueError(f"invalid rule60 counter in FEN: {fen[:80]}") from exc
    return board, stm, rule60_ply


def _write_position(planes: np.ndarray, base: int, board_field: str, stm: str) -> None:
    """Write px0's current-side-relative piece planes into one history block."""
    ranks = board_field.split("/")
    if len(ranks) != _ROWS:
        raise ValueError(f"expected 10 board rows, got {len(ranks)}: {board_field[:80]}")

    black_to_move = stm == "b"
    for fen_row, rank in enumerate(ranks):
        cells = _expand_rank(rank)
        if len(cells) != _COLS:
            raise ValueError(f"expected 9 files in row {fen_row}, got {len(cells)}: {rank!r}")
        for file_idx, piece in enumerate(cells):
            if piece == ".":
                continue
            if piece.upper() not in _PIECE_PLANES:
                raise ValueError(f"unknown piece symbol {piece!r} in {board_field[:80]}")

            # px0 ChessBoard::Mirror flips ranks and swaps ours for black.
            rank_idx = _ROWS - 1 - fen_row
            if black_to_move:
                rank_idx = _ROWS - 1 - rank_idx
            ours = piece.islower() if black_to_move else piece.isupper()
            plane = _PIECE_PLANES.index(piece.upper()) + (0 if ours else 7)
            planes[base + plane, rank_idx, file_idx] = 1.0


def fen_to_planes(fen: str, *, device: torch.device | None = None) -> torch.Tensor:
    """Encode an isolated FEN as px0 classical input `[124, 10, 9]`.

    This is the `FillEmptyHistory::FEN_ONLY` fallback in px0
    `src/neural/encoder.cc:118-218`: the initial position has no synthetic
    history; other isolated FENs repeat the current board in missing slots.
    A caller holding real moves must encode real history rather than use this
    fallback.
    """
    board_field, stm, rule60_ply = _parse_board(fen)
    planes = np.zeros((_INPUT_PLANES, _ROWS, _COLS), dtype=np.float32)
    _write_position(planes, 0, board_field, stm)

    start_board, start_stm, _ = _parse_board(START_FEN)
    if (board_field, stm) != (start_board, start_stm):
        for block in range(1, _HISTORY_LENGTH):
            _write_position(planes, block * _PLANES_PER_POSITION, board_field, stm)

    if stm == "b":
        planes[_AUX_BASE].fill(1.0)
    planes[_AUX_BASE + 1].fill(float(rule60_ply))
    planes[_AUX_BASE + 3].fill(1.0)

    tensor = torch.from_numpy(planes)
    return tensor.to(device) if device is not None else tensor
