import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import START_FEN
from nn.fen_tensor import fen_to_planes


def test_startpos_matches_px0_classical_encoder():
    """px0 src/neural/encoder_test.cc:25-101."""
    planes = fen_to_planes(START_FEN)

    assert tuple(planes.shape) == (124, 10, 9)
    assert planes[0, 0, 0] == 1.0  # our rook a0
    assert planes[0, 0, 8] == 1.0  # our rook i0
    assert planes[6, 0, 4] == 1.0  # our king e0
    assert planes[13, 9, 4] == 1.0  # their king e9
    assert torch.count_nonzero(planes[15:120]) == 0
    assert torch.count_nonzero(planes[120]) == 0
    assert torch.count_nonzero(planes[121]) == 0
    assert torch.count_nonzero(planes[122]) == 0
    assert torch.all(planes[123] == 1.0)


def test_non_start_fen_only_repeats_missing_history_and_flips_black():
    """px0 src/neural/encoder.cc:156-212, FillEmptyHistory::FEN_ONLY."""
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR b - - 7 1"
    planes = fen_to_planes(fen)

    for block in range(1, 8):
        torch.testing.assert_close(planes[:15], planes[block * 15 : (block + 1) * 15])
    assert torch.all(planes[120] == 1.0)
    assert torch.all(planes[121] == 7.0)
