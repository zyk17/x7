"""xiangqi_core 合法 UCI 与 pyffish ``legal_moves`` 交叉对拍（需 pyffish + cargo）。"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyffish")


def _repo_root() -> Path:
    # nn/tests/ -> repo root
    return Path(__file__).resolve().parents[2]


def test_pyffish_xiangqi_core_parity_script():
    if shutil.which("cargo") is None:
        pytest.skip("PATH 中无 cargo")
    repo = _repo_root()
    script = repo / "nn" / "scripts" / "parity" / "pyffish_xiangqi_core_parity.py"
    r = subprocess.run(
        [sys.executable, str(script), "--repo", str(repo)],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
