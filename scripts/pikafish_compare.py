#!/usr/bin/env python3
"""在固定局面上收集 Pikafish 的最终 MultiPV。

仅用于本地搜索诊断。协议参考 UCI 的 ``uci/isready/position/go`` 生命周期；
不参与 Rust 引擎、训练或正式基准吞吐。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_ENGINE = Path(r"C:\games\xiangqi\engines\pikafish-bmi2.exe")
MULTIPV_RE = re.compile(r"(?:^|\s)multipv\s+(\d+)(?:\s|$)")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="print Pikafish final MultiPV for one position")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--fen", help="FEN root; omit for startpos")
    parser.add_argument("--moves", default="", help="space-separated ICCS moves")
    parser.add_argument(
        "--searchmove",
        action="append",
        default=[],
        metavar="ICCS",
        help="restrict root search; repeat for multiple moves",
    )
    parser.add_argument("--movetime", type=int, default=3000, help="thinking time in milliseconds")
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--raw", action="store_true", help="also print all UCI output")
    args = parser.parse_args()
    if args.movetime <= 0 or args.multipv <= 0:
        parser.error("--movetime and --multipv must be positive")
    if not args.engine.is_file():
        parser.error(f"engine missing: {args.engine}")
    return args


def send(process: subprocess.Popen[str], command: str) -> None:
    assert process.stdin is not None
    process.stdin.write(command + "\n")
    process.stdin.flush()


def read_until(process: subprocess.Popen[str], expected: str, raw: bool) -> None:
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Pikafish exited before {expected}")
        line = line.rstrip()
        if raw:
            print(line)
        if line == expected:
            return


def main() -> int:
    args = arguments()
    process = subprocess.Popen(
        [str(args.engine)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        send(process, "uci")
        read_until(process, "uciok", args.raw)
        send(process, f"setoption name MultiPV value {args.multipv}")
        send(process, "isready")
        read_until(process, "readyok", args.raw)

        position = f"position fen {args.fen}" if args.fen else "position startpos"
        moves = args.moves.split()
        if moves:
            position += " moves " + " ".join(moves)
        send(process, position)
        # Pikafish consumes every token after `searchmoves` as a move, so it must
        # be the final go field even though UCI permits flexible field ordering.
        go = f"go movetime {args.movetime}"
        if args.searchmove:
            go += " searchmoves " + " ".join(args.searchmove)
        send(process, go)

        final_info: dict[int, str] = {}
        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Pikafish exited before bestmove")
            line = line.rstrip()
            if args.raw:
                print(line)
            match = MULTIPV_RE.search(line)
            if line.startswith("info ") and match:
                final_info[int(match.group(1))] = line
            if line.startswith("bestmove "):
                for index in sorted(final_info):
                    print(final_info[index])
                print(line)
                return 0
    finally:
        if process.poll() is None:
            send(process, "quit")
            process.wait(timeout=2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"pikafish_compare: {error}", file=sys.stderr)
        raise SystemExit(2) from error
