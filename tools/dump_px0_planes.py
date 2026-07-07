from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

TOTAL_PLANES = 124
ROWS = 10
COLS = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump official px0 classical planes through the px0 Python bindings.")
    parser.add_argument("--px0-build-dir", required=True, type=Path, help="px0 Meson build dir containing backends.cp311-win_amd64.pyd")
    parser.add_argument("--fen", default=None, help="Optional FEN. Omit to use startpos.")
    parser.add_argument("--moves", nargs="*", default=[], help="Optional UCI/ICCS moves applied on top of FEN/startpos")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def load_backends(build_dir: Path):
    sys.path.insert(0, str(build_dir))
    dll_dir = build_dir.resolve()
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))
        zlib_dir = dll_dir / "subprojects" / "zlib-1.3.1"
        if zlib_dir.exists():
            os.add_dll_directory(str(zlib_dir))
    import backends  # type: ignore

    return backends


def planes_to_nested(inp) -> list[list[list[float]]]:
    planes = np.zeros((TOTAL_PLANES, ROWS, COLS), dtype=np.float32)
    for plane in range(TOTAL_PLANES):
        lo, hi = inp.mask128(plane)
        mask = (int(hi) << 64) | int(lo)
        value = float(inp.val(plane))
        for index in range(ROWS * COLS):
            if (mask >> index) & 1:
                rank, col = divmod(index, COLS)
                row = rank
                planes[plane, row, col] = value
    return planes.tolist()


def plane_nonzero_counts(planes: list[list[list[float]]]) -> list[int]:
    counts: list[int] = []
    for plane in planes:
        count = 0
        for row in plane:
            for value in row:
                if value != 0.0:
                    count += 1
        counts.append(count)
    return counts


def main() -> None:
    args = parse_args()
    backends = load_backends(args.px0_build_dir)
    state = backends.GameState(args.fen, args.moves)
    inp = state.as_input_classical()
    planes = planes_to_nested(inp)
    payload = {
        "fen": args.fen or "",
        "moves": args.moves,
        "shape": [TOTAL_PLANES, ROWS, COLS],
        "legal_moves": list(state.moves()),
        "policy_indices": list(state.policy_indices()),
        "plane_nonzero_counts": plane_nonzero_counts(planes),
        "planes": planes,
    }
    text = json.dumps(payload, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
