"""FEN → CNN 输入平面：红方 7 类棋子 + 黑方 7 类 + 轮到谁走（共 15 通道），形状 [C, 10, 9]。"""

from __future__ import annotations

import numpy as np
import torch


RED_CHARS = "RNBAKCP"
BLACK_CHARS = "rnbakcp"


def _expand_rank(rank_str: str) -> list[str]:
    cells: list[str] = []
    for ch in rank_str:
        if ch.isdigit():
            cells.extend(["."] * int(ch))
        else:
            cells.append(ch)
    return cells


def fen_to_planes(fen: str, *, device: torch.device | None = None) -> torch.Tensor:
    """
    将局面 FEN（完整字符串）编码为 float32 tensor [15, 10, 9]。
    坐标：row 0 = FEN 第一行（棋谱通常最远一侧），col 0..8。
    """
    parts = fen.split()
    board = parts[0]
    stm = parts[1] if len(parts) > 1 else "w"

    ranks = board.split("/")
    if len(ranks) != 10:
        raise ValueError(f"期望 10 行棋盘，得到 {len(ranks)}: {fen[:80]}")

    planes = np.zeros((15, 10, 9), dtype=np.float32)
    red_idx = {c: i for i, c in enumerate(RED_CHARS)}
    black_idx = {c: i + 7 for i, c in enumerate(BLACK_CHARS)}

    for ri, rank in enumerate(ranks):
        cells = _expand_rank(rank)
        if len(cells) != 9:
            raise ValueError(f"行 {ri} 长度应为 9，得到 {len(cells)}: {rank!r}")
        for fi, ch in enumerate(cells):
            if ch == ".":
                continue
            if ch in red_idx:
                planes[red_idx[ch], ri, fi] = 1.0
            elif ch in black_idx:
                planes[black_idx[ch], ri, fi] = 1.0
            else:
                raise ValueError(f"未知棋子符号: {ch!r} in {fen[:80]}")

    # 通道 14：轮到红走（w）为全 1，否则全 0（表示当前行动方为黑方可反过来约定，训练一致即可）
    if stm == "w":
        planes[14, :, :] = 1.0
    else:
        planes[14, :, :] = 0.0

    t = torch.from_numpy(planes)
    if device is not None:
        t = t.to(device)
    return t
