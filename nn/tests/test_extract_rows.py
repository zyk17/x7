import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_pgn import iter_training_rows
from pgn import read_pgn_games


def test_iter_one_game():
    pgn = ROOT / "tests" / "fixtures" / "one_game_uci.pgn"
    game = next(read_pgn_games(pgn))
    rows = iter_training_rows(game, game_id="t0")
    assert len(rows) >= 1
    r0 = rows[0]
    assert r0["human_move_pyffish"] == "a1a2"
    assert r0["human_move_uci"] == "a0a1"
    assert r0["game_id"] == "t0"
    assert r0["root_fen"]
    assert r0["uci_prefix"] == []
