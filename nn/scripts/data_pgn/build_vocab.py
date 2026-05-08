#!/usr/bin/env python3
"""扫描 JSONL，收集 `human_move_pyffish` 构建固定走法词表（与 pyffish legal_moves 字符串一致）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="JSONL → move_vocab.json")
    ap.add_argument("--jsonl", type=Path, action="append", required=True, help="可多次指定")
    ap.add_argument("--out", type=Path, required=True, help="输出 JSON：{ moves: [str,...] }")
    args = ap.parse_args()

    seen: set[str] = set()
    for path in args.jsonl:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                m = row.get("human_move_pyffish")
                if m:
                    seen.add(m)

    moves = sorted(seen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"moves": moves, "size": len(moves)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(moves)} moves -> {args.out}")


if __name__ == "__main__":
    main()
