#!/usr/bin/env python3
"""
从 PGN / `.pgns` 导出训练行（JSONL）：局面 + 人类着法（UCI）。

记谱：`[Format "ICCS"]` 且着法形如 `C3-C4` 时自动转换；纯四字母且已为 pyffish 坐标时请写 `[Format "UCI"]`。
说明见仓库根目录 `pgn格式.txt`、`pgns格式.txt`。

用法:
  python scripts/data_pgn/extract_rows.py --pgn a.pgns --out data/train.jsonl
  python scripts/data_pgn/extract_rows.py --pgn a.pgns --pgn b.pgns --out data/all.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from constants import VARIANT
from dataset_pgn import iter_training_rows
from pgn import read_pgn_games


def main() -> None:
    ap = argparse.ArgumentParser(description="PGN → JSONL（position→move）")
    ap.add_argument(
        "--pgn",
        type=Path,
        action="append",
        dest="pgns",
        required=True,
        metavar="PATH",
        help="输入 PGN / .pgns（可重复多次，依次写入同一 JSONL）",
    )
    ap.add_argument("--out", type=Path, required=True, help="输出 JSONL 路径")
    ap.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="每个输入文件最多处理对局数，0 表示不限制",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_games = 0
    n_rows = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for pgn_path in args.pgns:
            stem = pgn_path.stem
            games = list(read_pgn_games(pgn_path))
            if args.max_games:
                games = games[: args.max_games]
            desc = f"games[{stem}]"
            for gi, game in enumerate(tqdm(games, desc=desc)):
                game_id = f"{stem}_{gi:06d}"
                try:
                    rows = iter_training_rows(game, game_id=game_id)
                except Exception as e:
                    tqdm.write(f"skip game {game_id}: {e}")
                    continue
                for row in rows:
                    row["variant"] = VARIANT
                    row["pgn_source"] = stem
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1
                n_games += 1

    print(f"wrote {n_rows} rows from {n_games} games -> {args.out}")


if __name__ == "__main__":
    main()
