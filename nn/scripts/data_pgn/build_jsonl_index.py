#!/usr/bin/env python3
"""为超大 JSONL 建 mmap 训练用行索引（row 字节范围 + game_idx + 权重等）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn import build_jsonl_index


def main() -> None:
    ap = argparse.ArgumentParser(description="JSONL → mmap 索引目录（npy + json）")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True, help="build_vocab 生成的 JSON")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--weight-by-fen",
        action="store_true",
        help="训练集推荐：两遍扫描，weights=1/sqrt(训练集中该 fen 出现次数)。验证集不要加此开关。",
    )
    args = ap.parse_args()

    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    moves: list[str] = vocab["moves"]
    move_to_idx = {m: i for i, m in enumerate(moves)}

    n = build_jsonl_index(
        args.jsonl,
        move_to_idx,
        args.out_dir,
        weight_by_fen=args.weight_by_fen,
    )
    print(f"indexed rows={n} -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
