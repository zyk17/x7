"""按 PGN 的完整历史重放 UCI 搜索，诊断跨回合 graph reuse。

默认只搜索红方回合；每一回合仍以完整 `position startpos moves` 推进，因而能以
同一 Engine 连续观察跨回合 graph reuse 下的 NPS 与实际着一致性。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


MOVE = re.compile(r"\b[a-i][0-9]-[a-i][0-9]\b")
FIELD = re.compile(r"\b(nodes|nps|eps|time)\s+([0-9]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_moves(pgn: Path) -> list[str]:
    return [move.replace("-", "") for move in MOVE.findall(pgn.read_text(encoding="utf-8"))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--engine", type=Path, default=Path("target/release/x7.exe"))
    parser.add_argument("--weights", default="data/x7.onnx")
    parser.add_argument("--movetime", type=int, default=3000)
    parser.add_argument(
        "--from-ply",
        type=int,
        default=0,
        help="first ply to search; the engine is still positioned with the full earlier history",
    )
    parser.add_argument("--until-ply", type=int, default=0, help="0 means the whole PGN")
    parser.add_argument("--both-sides", action="store_true", help="also search black turns")
    args = parser.parse_args()
    moves = read_moves(args.pgn)
    until = args.until_ply or len(moves)
    if until < 1 or until > len(moves):
        parser.error("--until-ply must be within the PGN move list")
    if not 0 <= args.from_ply < until:
        parser.error("--from-ply must be non-negative and smaller than --until-ply")

    process = subprocess.Popen(
        [args.engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None

    def send(command: str) -> None:
        process.stdin.write(command + "\n")
        process.stdin.flush()

    def wait_for(marker: str) -> None:
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"engine exited before {marker}")
            if line.strip() == marker:
                return

    def search(ply: int) -> tuple[str, dict[str, str]]:
        prefix = " ".join(moves[:ply])
        send("position startpos" + (f" moves {prefix}" if prefix else ""))
        send(f"go movetime {args.movetime}")
        send("wait")
        info = ""
        while line := process.stdout.readline():
            line = ANSI.sub("", line).strip()
            if line.startswith("info "):
                info = line
            if line.startswith("bestmove "):
                return line.split()[1], dict(FIELD.findall(info))
        raise RuntimeError("engine exited while searching")

    try:
        send("uci")
        wait_for("uciok")
        send(f"setoption name WeightsFile value {args.weights}")
        send("isready")
        wait_for("readyok")
        for ply in range(args.from_ply, until):
            if ply % 2 and not args.both_sides:
                continue
            best, fields = search(ply)
            actual = moves[ply]
            print(
                f"ply={ply:3} actual={actual} best={best} same={best == actual} "
                f"time={fields.get('time', '?')} nodes={fields.get('nodes', '?')} "
                f"nps={fields.get('nps', '?')} eps={fields.get('eps', '?')}",
                flush=True,
            )
    finally:
        if process.poll() is None:
            send("quit")
            process.wait(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
