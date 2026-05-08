from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    """仓库根目录（含 engines/、src/ 的上一级）。"""
    return Path(__file__).resolve().parent.parent.parent


def resolve_pikafish_executable(explicit: str | Path | None = None) -> Path:
    """
    解析皮卡鱼可执行文件路径：
    1. `explicit` 参数
    2. 环境变量 `PIKAFISH_PATH`
    3. 仓库 `engines/` 下默认文件名（Windows / Unix）
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"皮卡鱼可执行文件不存在: {p}")
        return p.resolve()

    env = os.environ.get("PIKAFISH_PATH", "").strip()
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"PIKAFISH_PATH 指向的文件不存在: {p}")
        return p.resolve()

    root = project_root()
    candidates = [
        root / "engines" / "pikafish.exe",
        root / "engines" / "Pikafish.exe",
        root / "engines" / "pikafish",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()

    raise FileNotFoundError(
        "未找到皮卡鱼：请设置环境变量 PIKAFISH_PATH，或将可执行文件放入 "
        f"{root / 'engines'}（见 engines/README.txt）"
    )


@dataclass
class EngineInfo:
    bestmove: str | None
    score_cp: int | None
    mate: int | None


class PikafishUCI:
    """皮卡鱼 UCI 子进程（阻塞式读写，适合批处理标注）。"""

    def __init__(self, executable: str | Path | None = None) -> None:
        self._exe = str(resolve_pikafish_executable(executable))
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _read_until(self, pred) -> list[str]:
        assert self._proc and self._proc.stdout
        lines: list[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\r\n")
            lines.append(line)
            if pred(line):
                break
        return lines

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [self._exe],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until(lambda s: s == "uciok")
        self._send("setoption name UCI_Variant value xiangqi")
        self._send("isready")
        self._read_until(lambda s: s == "readyok")

    def _send(self, cmd: str) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._send("quit")
        except BrokenPipeError:
            pass
        self._proc.wait(timeout=5)
        self._proc = None

    def go_fen(self, fen: str, movetime_ms: int = 200) -> EngineInfo:
        """对局面 `fen`（不含前置 moves 时从当前 FEN 开始）短思考，解析 bestmove 与 cp/mate。"""
        with self._lock:
            self.start()
            assert self._proc
            self._send(f"position fen {fen}")
            self._send(f"go movetime {movetime_ms}")
            lines = self._read_until(lambda s: s.startswith("bestmove"))
        best = None
        score_cp: int | None = None
        mate: int | None = None
        for ln in lines:
            if "score cp" in ln:
                m = re.search(r"score cp (-?\d+)", ln)
                if m:
                    score_cp = int(m.group(1))
            if "score mate" in ln:
                m = re.search(r"score mate (-?\d+)", ln)
                if m:
                    mate = int(m.group(1))
        for ln in reversed(lines):
            if ln.startswith("bestmove"):
                parts = ln.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    best = parts[1]
                break
        return EngineInfo(bestmove=best, score_cp=score_cp, mate=mate)

    def __enter__(self) -> PikafishUCI:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
