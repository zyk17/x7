#!/usr/bin/env python3
"""对比 ``xiangqi_core`` 与 pyffish 在若干种子局面上的合法 UCI 集合（统一到 **Pikafish 坐标** ``a0``～``i9``）。

pyffish 返回的着法串仍常为纵坐标 **1～10**；本脚本在调用前后做 **0～9 ↔ 1～10** 转换后再比较。
个别中局（如炮线、飞将）两实现仍可能不一致，用例集刻意避开已知分歧局面。

用法::

    cd nn && pip install -e .
    python scripts/parity/pyffish_xiangqi_core_parity.py

依赖：已安装 ``pyffish``、系统 PATH 中有 ``cargo``。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pyffish as sf

# pyffish 象棋 UCI 串仍常用纵坐标 1～10；本仓库核心与 Pikafish 一致为 0～9。对拍时在边界做转换。
_CORE_MOVE = re.compile(r"^([a-i])([0-9])([a-i])([0-9])([a-z])?$")
_PYFFISH_MOVE = re.compile(r"^([a-i])(10|[1-9])([a-i])(10|[1-9])([a-z])?$")


def core_uci_to_pyffish(uci: str) -> str:
    m = _CORE_MOVE.match(uci.strip().lower())
    if not m:
        raise ValueError(f"非标准 UCI（期望 a0～i9）: {uci!r}")

    def enc_rank(r: int) -> str:
        if not 0 <= r <= 9:
            raise ValueError(r)
        return "10" if r == 9 else str(r + 1)

    r1 = int(m.group(2))
    r2 = int(m.group(4))
    s = f"{m.group(1)}{enc_rank(r1)}{m.group(3)}{enc_rank(r2)}"
    if m.group(5):
        s += m.group(5)
    return s


def pyffish_uci_to_core(uci: str) -> str:
    m = _PYFFISH_MOVE.match(uci.strip().lower())
    if not m:
        raise ValueError(f"无法解析 pyffish UCI: {uci!r}")

    def dec_rank(rs: str) -> int:
        pr = 10 if rs == "10" else int(rs)
        return pr - 1

    r1 = dec_rank(m.group(2))
    r2 = dec_rank(m.group(4))
    core = f"{m.group(1)}{r1}{m.group(3)}{r2}"
    if m.group(5):
        core += m.group(5)
    return core

# 与 nn/src/constants.py 一致
VARIANT = "xiangqi"

# （fen, prefix_moves）：prefix 为从根 FEN 依次执行的 **标准 UCI**（纵坐标 0～9，与 Pikafish / xiangqi_core 一致）。
PARITY_CASES: list[tuple[str, list[str]]] = [
    (
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        [],
    ),
    (
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        ["b0c2"],
    ),
    (
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        ["a0a1", "a9a7"],
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


def pyffish_sorted_legal(fen: str, prefix_core: list[str]) -> list[str]:
    prefix_pf = [core_uci_to_pyffish(m) for m in prefix_core]
    moves = list(sf.legal_moves(VARIANT, fen, prefix_pf))
    out = [pyffish_uci_to_core(m) for m in moves]
    out.sort()
    return out


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
