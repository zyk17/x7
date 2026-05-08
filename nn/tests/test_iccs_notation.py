import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_pgn import iter_training_rows
from notation_iccs import iccs_move_to_pyffish, pyffish_move_to_iccs
from pgn import read_pgn_games


def test_iccs_roundtrip():
    assert iccs_move_to_pyffish("C3-C4") == "c4c5"
    assert pyffish_move_to_iccs("c4c5") == "c3-c4"


def test_full_sample_game_50_moves():
    pgn = ROOT / "tests" / "fixtures" / "iccs_sample.pgn"
    game = next(read_pgn_games(pgn))
    rows = iter_training_rows(game, game_id="sample")
    # 50 回合 ×2 约 100 个半着；若某一着与 pyffish 规则不一致会提前停止
    assert len(rows) >= 95
    assert rows[0]["human_move_pyffish"] == "c4c5"
    assert rows[0]["human_move_uci"] == "c3c4"
    assert rows[1]["human_move_pyffish"] == "c10e8"
