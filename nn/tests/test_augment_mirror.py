import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from augment_mirror import (
    mirror_board_fen_field,
    mirror_fen,
    mirror_move_uci,
)
from constants import START_FEN
from nn.fen_tensor import fen_to_planes


def test_mirror_uci_involution():
    u = "c4c5"
    assert mirror_move_uci(mirror_move_uci(u)) == u
    assert mirror_move_uci(u) == "g4g5"


def test_mirror_fen_involution():
    m = mirror_fen(START_FEN)
    assert mirror_fen(m) == START_FEN


def test_planes_match_horizontal_flip():
    t0 = fen_to_planes(START_FEN)
    m = mirror_fen(START_FEN)
    t1 = fen_to_planes(m)
    torch.testing.assert_close(t0.flip(-1), t1)


def test_mirror_board_field_roundtrip_compress():
    b = START_FEN.split()[0]
    assert mirror_board_fen_field(mirror_board_fen_field(b)) == b
