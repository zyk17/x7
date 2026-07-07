from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PLANES_PER_BOARD = 15
HISTORY_LEN = 8
TOTAL_PLANES = 124
ROWS = 10
COLS = 9
START_BOARD = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"

PIECE_OFFSETS = {
    "r": 0,
    "a": 1,
    "c": 2,
    "p": 3,
    "n": 4,
    "b": 5,
    "k": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare engin dump_planes output against px0 classical rules.")
    parser.add_argument("--dump", required=True, type=Path, help="NDJSON emitted by dump_planes")
    return parser.parse_args()


def expand_board(board: str) -> list[list[str]]:
    rows = []
    for rank in board.split("/"):
        row: list[str] = []
        for ch in rank:
            if ch.isdigit():
                row.extend(["."] * int(ch))
            else:
                row.append(ch)
        if len(row) != COLS:
            raise ValueError(f"bad board row: {rank}")
        rows.append(row)
    if len(rows) != ROWS:
        raise ValueError(f"bad board rows: {board}")
    return rows


def board_part(fen: str) -> str:
    return fen.split()[0]


def piece_color(piece: str) -> str:
    return "w" if piece.isupper() else "b"


def rebuild_planes(payload: dict) -> np.ndarray:
    history_entries = payload["history_entries"]
    root_stm = history_entries[-1]["side_to_move"]
    root_rule60 = int(history_entries[-1]["rule60"])
    planes = np.zeros((TOTAL_PLANES, ROWS, COLS), dtype=np.float32)

    earliest_board = board_part(history_entries[0]["fen"])
    for block in range(HISTORY_LEN):
        if block < len(history_entries):
            entry = history_entries[len(history_entries) - 1 - block]
            repeated = bool(entry["repeated"])
        elif earliest_board != START_BOARD:
            entry = history_entries[0]
            repeated = False
        else:
            continue

        stm = entry["side_to_move"]
        flip = stm != root_stm
        board = expand_board(board_part(entry["fen"]))
        base = block * PLANES_PER_BOARD
        for fen_row, row in enumerate(board):
            for col, piece in enumerate(row):
                if piece == ".":
                    continue
                offset = PIECE_OFFSETS[piece.lower()]
                ours = piece_color(piece) == stm
                plane = base + offset if ours else base + offset + 7
                dst_row = ROWS - 1 - fen_row if flip else fen_row
                planes[plane, dst_row, col] = 1.0
        if repeated:
            planes[base + 14, :, :] = 1.0

    planes[120, :, :] = 1.0 if root_stm == "b" else 0.0
    planes[121, :, :] = float(max(0, min(119, root_rule60))) / 119.0
    planes[123, :, :] = 1.0
    return planes


def compare_payload(payload: dict) -> dict:
    actual = np.asarray(payload["planes"], dtype=np.float32)
    expected = rebuild_planes(payload)
    diff = np.abs(actual - expected)
    mismatch = np.argwhere(diff > 1e-6)
    per_plane = (diff > 1e-6).reshape(TOTAL_PLANES, -1).sum(axis=1).tolist()
    first = None
    if mismatch.size:
        plane, row, col = mismatch[0].tolist()
        first = {
            "plane": plane,
            "row": row,
            "col": col,
            "actual": float(actual[plane, row, col]),
            "expected": float(expected[plane, row, col]),
        }
    return {
        "input": payload["input"],
        "fen": payload["fen"],
        "history_len": payload["history_len"],
        "mismatch_count": int((diff > 1e-6).sum()),
        "planes_with_mismatch": [idx for idx, count in enumerate(per_plane) if count],
        "per_plane_mismatch_count": per_plane,
        "first_mismatch": first,
    }


def main() -> None:
    args = parse_args()
    lines = [line for line in args.dump.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        payload = json.loads(line)
        if "error" in payload:
            print(json.dumps(payload, ensure_ascii=False))
            continue
        result = compare_payload(payload)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
