#!/usr/bin/env python3
"""对比 ``xiangqi_core`` 与 pyffish 在若干种子局面（含 ``root_fen`` + ``uci_prefix``）上的合法 UCI 集合。

用法（仓库根目录）::

    cd nn && pip install -e .
    python scripts/parity/pyffish_xiangqi_core_parity.py

依赖：已安装 ``pyffish``、系统 PATH 中有 ``cargo``。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pyffish as sf

# 与 nn/src/constants.py 一致
VARIANT = "xiangqi"

# （fen, prefix_moves）：prefix 为从根 FEN 依次执行的 pyffish UCI；空列表表示仅根局面。
# 用例选自现有 Rust 测试（起始面、双王残面）及经脚本校验一致的开局前缀。
PARITY_CASES: list[tuple[str, list[str]]] = [
    (
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        [],
    ),
    ("9/9/9/9/9/9/9/9/4k4/4K4 w - - 0 1", []),
    (
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        ["b1c3"],
    ),
    (
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        ["a1a2", "a10a8"],
    ),
]


def repo_root_from_script() -> Path:
    # nn/scripts/parity/this_file.py -> parents[3] = workspace root
    return Path(__file__).resolve().parents[3]


def rust_sorted_legal(repo: Path, fen: str, prefix: list[str]) -> list[str]:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "xiangqi_core",
        "--bin",
        "legal_moves_dump",
        "--",
        "--fen",
        fen,
        "--prefix",
        " ".join(prefix),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "legal_moves_dump 失败:\n"
            + (proc.stderr or proc.stdout or "").strip()
        )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    lines.sort()
    return lines


def pyffish_sorted_legal(fen: str, prefix: list[str]) -> list[str]:
    moves = list(sf.legal_moves(VARIANT, fen, prefix))
    moves.sort()
    return moves


def run_cases(repo: Path, verbose: bool = False) -> list[str]:
    """返回人类可读的错误信息列表；空列表表示全部通过。"""
    errors: list[str] = []
    for fen, prefix in PARITY_CASES:
        try:
            r = rust_sorted_legal(repo, fen, prefix)
            p = pyffish_sorted_legal(fen, prefix)
        except Exception as e:
            errors.append(f"局面异常 prefix={prefix!r}: {e}")
            continue
        if r != p:
            rs, ps = set(r), set(p)
            errors.append(
                f"合法集不一致 prefix={prefix!r}\n"
                f"  仅 Rust: {sorted(rs - ps)}\n"
                f"  仅 pyffish: {sorted(ps - rs)}"
            )
        elif verbose:
            print(f"OK prefix={prefix!r} count={len(r)}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Monorepo 根目录（默认从脚本路径推断）",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    repo = args.repo if args.repo is not None else repo_root_from_script()
    if not (repo / "Cargo.toml").is_file():
        print(f"错误: 未找到 Cargo.toml: {repo}", file=sys.stderr)
        return 2
    errs = run_cases(repo, verbose=args.verbose)
    if errs:
        print("对拍失败:", file=sys.stderr)
        for e in errs:
            print(e, file=sys.stderr)
        return 1
    if args.verbose:
        print(f"全部通过（{len(PARITY_CASES)} 例）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
