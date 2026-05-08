from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class ParsedGame:
    headers: dict[str, str]
    movetext_raw: str


_UCI_TOKEN = re.compile(r"^[a-i][0-9][a-i][0-9][a-z]?$")
_ICCS_MOVE = re.compile(
    r"\b([A-Ia-i]\d+)\s*-\s*([A-Ia-i]\d+)\b",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(r'^\[(\w+)\s+"((?:\\.|[^"])*)"\s*\]$')


def strip_comments_and_variations(movetext: str) -> str:
    """去掉 { } 注释与 ( ) 变着，保留主干着法串。"""
    s = movetext
    while "{" in s:
        s = re.sub(r"\{[^{}]*\}", " ", s)
    depth = 0
    out: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def movetext_uci_tokens(movetext: str) -> list[str]:
    """从清理后的 movetext 中提取 **pyffish 风格** 四格 UCI（极少见于本仓库 PGN，保留兼容）。"""
    clean = strip_comments_and_variations(movetext)
    tokens: list[str] = []
    for raw in clean.split():
        t = raw.strip()
        if not t:
            continue
        if t in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        if re.match(r"^\d+\.\.\.", t):
            continue
        if re.match(r"^\d+\.", t):
            continue
        if _UCI_TOKEN.match(t):
            tokens.append(t)
    return tokens


def movetext_iccs_pairs(movetext: str) -> list[str]:
    """
    提取 ICCS 着法 `C3-C4`（东萍 / 联赛 PGN 常见）。
    返回小写半格、带连字符，如 `c3-c4`，便于再转为引擎 UCI 或 pyffish。
    """
    clean = strip_comments_and_variations(movetext)
    out: list[str] = []
    for m in _ICCS_MOVE.finditer(clean):
        a, b = m.group(1), m.group(2)
        out.append(f"{a.lower()}-{b.lower()}")
    return out


def pgn_format(game_headers: dict[str, str]) -> str:
    """`ICCS` | `WXF` | `UCI` | 空（空则按棋谱推断）。"""
    return (game_headers.get("Format") or "").strip().upper()


def read_pgn_games(path: str | Path) -> Iterator[ParsedGame]:
    """
    读取 PGN：以 `[Event` 作为新对局起点；每个对局内标签头读到首个空行，
    其后直至下一 `[Event`（或文件尾）为 movetext（可与头块之间隔一个空行）。
    """
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    starts = [m.start() for m in re.finditer(r"(?m)^\[Event\s", raw)]
    if not starts:
        return
    starts.append(len(raw))
    for i in range(len(starts) - 1):
        chunk = raw[starts[i] : starts[i + 1]].strip()
        if not chunk:
            continue
        lines = chunk.split("\n")
        headers: dict[str, str] = {}
        movetext_lines: list[str] = []
        phase = "headers"
        for line in lines:
            s = line.strip()
            if phase == "headers":
                if not s:
                    phase = "moves"
                    continue
                if s.startswith("["):
                    m = _HEADER_RE.match(s)
                    if m:
                        key, val = m.group(1), m.group(2).replace('\\"', '"')
                        headers[key] = val
                    continue
                phase = "moves"
            if phase == "moves" and s:
                movetext_lines.append(s)
        if headers:
            yield ParsedGame(headers=headers, movetext_raw=" ".join(movetext_lines).strip())
