"""XRSH v1 二进制解析与 Dataset 冒烟。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import START_FEN
from nn.dataset_xrsh import PolicyXrshDataset
from nn.policy_pack import vocab_fingerprint_ordered_moves
from nn.xrsh_io import HEADER_SIZE, parse_shard_bytes


def _write_str_u16(s: str) -> bytes:
    b = s.encode("utf-8")
    return len(b).to_bytes(2, "little") + b


def _write_prefix_list(pfx: list[str]) -> bytes:
    out = len(pfx).to_bytes(2, "little")
    for s in pfx:
        bb = s.encode("utf-8")
        out += bytes([len(bb)]) + bb
    return out


def _minimal_shard_bytes(*, vocab_hash32: bytes, games_body: bytes) -> bytes:
    assert len(vocab_hash32) == 32
    header = (
        b"XRSH"
        + (1).to_bytes(4, "little")
        + vocab_hash32
        + (1).to_bytes(4, "little")
        + bytes(20)
    )
    assert len(header) == HEADER_SIZE
    return header + games_body


def test_parse_xrsh_single_game_row():
    moves_v = ["m0"]
    vh = bytes.fromhex(vocab_fingerprint_ordered_moves(moves_v))

    body = (
        _write_str_u16("game_a")
        + (1).to_bytes(4, "little")
        + _write_str_u16(START_FEN)
        + _write_str_u16(START_FEN)
        + _write_prefix_list([])
        + (0).to_bytes(4, "little", signed=True)
        + (1).to_bytes(2, "little")
        + (0).to_bytes(4, "little", signed=True)
        + (5).to_bytes(2, "little")
    )
    blob = _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    rows, got_hash = parse_shard_bytes(blob)
    assert got_hash == vh
    assert len(rows) == 1
    r = rows[0]
    assert r["game_id"] == "game_a"
    assert r["fen"] == START_FEN
    assert r["root_fen"] == START_FEN
    assert r["uci_prefix"] == []
    assert r["legal_idx"] == [0]
    assert r["target_idx"] == 0
    assert r["ply"] == 5


def test_policy_xrsh_dataset_smoke(tmp_path: Path) -> None:
    moves = ["m0"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {moves[0]: 0}

    d = tmp_path / "xrsh"
    d.mkdir()
    (d / "pack_meta.json").write_text(
        json.dumps(
            {
                "format": "xrsh_v1",
                "format_version": 1,
                "vocab_sha256": fp_hex,
                "shard_count": 1,
                "source": "test",
            }
        ),
        encoding="utf-8",
    )

    body = (
        _write_str_u16("g1")
        + (1).to_bytes(4, "little")
        + _write_str_u16(START_FEN)
        + _write_str_u16(START_FEN)
        + _write_prefix_list([])
        + (0).to_bytes(4, "little", signed=True)
        + (1).to_bytes(2, "little")
        + (0).to_bytes(4, "little", signed=True)
        + (1).to_bytes(2, "little")
    )
    vh = bytes.fromhex(fp_hex)
    shard = _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    (d / "shard_00000.xrsh").write_bytes(shard)

    ds = PolicyXrshDataset(d, move_to_idx, for_training=True)
    assert len(ds) == 1
    boards, masks, targets, weights = ds[0]
    assert boards.ndim == 3  # (15, 10, 9) 平面
    assert tuple(masks.shape) == (1,)
    assert int(targets.item()) == 0
    assert float(weights.item()) > 0
