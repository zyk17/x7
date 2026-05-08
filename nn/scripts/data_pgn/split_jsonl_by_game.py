#!/usr/bin/env python3
"""按 game_id 整局划分 train / val（默认 95/5），避免 position 级随机泄漏。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="JSONL 按 game_id 划分")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="合并后的 JSONL")
    ap.add_argument("--train-out", type=Path, required=True)
    ap.add_argument("--val-out", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_game: dict[str, list[dict]] = defaultdict(list)
    with args.inp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gid = row.get("game_id", "")
            by_game[gid].append(row)

    gids = list(by_game.keys())
    rng = random.Random(args.seed)
    rng.shuffle(gids)
    if len(gids) <= 1:
        val_set = set()
        train_set = set(gids)
    else:
        n_val = max(1, int(len(gids) * args.val_ratio))
        n_val = min(n_val, len(gids) - 1)
        val_set = set(gids[:n_val])
        train_set = set(gids[n_val:])

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)

    nt = nv = 0
    with args.train_out.open("w", encoding="utf-8") as ft, args.val_out.open(
        "w", encoding="utf-8"
    ) as fv:
        for gid, rows in by_game.items():
            out = fv if gid in val_set else ft
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if gid in val_set:
                    nv += 1
                else:
                    nt += 1

    print(
        f"games: {len(gids)} train_games={len(train_set)} val_games={len(val_set)} "
        f"rows train={nt} val={nv}"
    )


if __name__ == "__main__":
    main()
