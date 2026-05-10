"""FEN → 紧凑棋盘（供离线训练包落盘；在线再扩成与 ``fen_to_planes`` 一致的 float 平面）。"""

from __future__ import annotations

import numpy as np
import torch

from nn.fen_tensor import RED_CHARS, BLACK_CHARS, _expand_rank


def fen_to_compact_board(fen: str) -> tuple[np.ndarray, np.uint8]:
    """
    将局面编码为 ``uint8[90]``（行主序：FEN 第 0 行 col 0..8，再第 1 行…）与 ``stm``。

    格值：0 空；1..7 红子（顺序同 ``RED_CHARS``→通道 0..6）；8..14 黑子（同 ``BLACK_CHARS``→通道 7..13）。
    通道 14（轮到谁走）不放在格子里，单独用 ``stm``：1=红走 ``w``，0=黑走。
    """
    parts = fen.split()
    board = parts[0]
    stm_s = parts[1] if len(parts) > 1 else "w"
    ranks = board.split("/")
    if len(ranks) != 10:
        raise ValueError(f"期望 10 行棋盘，得到 {len(ranks)}: {fen[:80]}")

    red_id = {c: i + 1 for i, c in enumerate(RED_CHARS)}
    black_id = {c: i + 8 for i, c in enumerate(BLACK_CHARS)}
    out = np.zeros(90, dtype=np.uint8)
    for ri, rank in enumerate(ranks):
        cells = _expand_rank(rank)
        if len(cells) != 9:
            raise ValueError(f"行 {ri} 长度应为 9，得到 {len(cells)}: {rank!r}")
        for fi, ch in enumerate(cells):
            if ch == ".":
                continue
            if ch in red_id:
                out[ri * 9 + fi] = red_id[ch]
            elif ch in black_id:
                out[ri * 9 + fi] = black_id[ch]
            else:
                raise ValueError(f"未知棋子符号: {ch!r} in {fen[:80]}")
    stm = np.uint8(1 if stm_s == "w" else 0)
    return out, stm


def compact_board_to_planes(board90: np.ndarray, stm: np.uint8 | int) -> np.ndarray:
    """``uint8[90]`` + stm → ``float32[15,10,9]``（与 ``fen_to_planes`` 数值一致）。"""
    b = np.asarray(board90, dtype=np.uint8).reshape(90)
    planes = np.zeros((15, 10, 9), dtype=np.float32)
    nz = np.flatnonzero(b)
    vals = b[nz].astype(np.int64) - 1
    rows = nz // 9
    cols = nz % 9
    planes[vals, rows, cols] = 1.0
    planes[14, :, :] = float(int(stm) & 1)
    return planes


def compact_board_to_torch_planes(board90: np.ndarray, stm: np.uint8 | int) -> torch.Tensor:
    return torch.from_numpy(compact_board_to_planes(board90, stm))


def compact_boards_to_torch_planes(
    boards90: np.ndarray, stms: np.ndarray | list[int]
) -> torch.Tensor:
    """批量 ``uint8[N,90]`` + ``stm[N]`` → ``float32[N,15,10,9]``。"""
    b = np.asarray(boards90, dtype=np.uint8).reshape(-1, 90)
    stm_arr = np.asarray(stms, dtype=np.uint8).reshape(-1)
    n = b.shape[0]
    planes = np.zeros((n, 15, 10, 9), dtype=np.float32)
    nz_batch, nz_pos = np.nonzero(b)
    vals = b[nz_batch, nz_pos].astype(np.int64) - 1
    rows = nz_pos // 9
    cols = nz_pos % 9
    planes[nz_batch, vals, rows, cols] = 1.0
    planes[:, 14, :, :] = stm_arr[:, None, None].astype(np.float32)
    return torch.from_numpy(planes)
