import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import START_FEN
from nn import (
    compact_board_to_planes,
    fen_to_compact_board,
)
from nn.fen_tensor import fen_to_planes


def test_compact_roundtrip_startpos():
    ref = fen_to_planes(START_FEN).numpy()
    b90, stm = fen_to_compact_board(START_FEN)
    got = compact_board_to_planes(b90, stm)
    np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)
