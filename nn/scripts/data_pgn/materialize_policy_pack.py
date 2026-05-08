#!/usr/bin/env python3
"""JSONL + 行索引 → 离线 policy 训练包（紧凑棋盘 + 稀疏合法着 + 定长 FEN/前缀列）。

物化阶段调用 pyffish / 解析 FEN；训练时不再跑 legal_moves、不再 json.loads。
须与 ``build_jsonl_index`` 使用同一 ``--jsonl`` 与 ``--vocab``，且索引目录含 sampler 文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn.materialize_pack import materialize_pack


def main() -> None:
    ap = argparse.ArgumentParser(description="物化 policy 训练 mmap 包")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, required=True, help="对应 JSONL 的 build_jsonl_index 输出目录")
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    n, total_legal = materialize_pack(args.jsonl, args.index_dir, args.vocab, args.out_dir)
    print(f"wrote pack n={n} total_legal={total_legal} -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
