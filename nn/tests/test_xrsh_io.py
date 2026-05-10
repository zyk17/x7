"""XRSH v3 二进制解析与 Dataset 冒烟。"""

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
from nn.dataset_xrsh import value_target_side_to_move
from nn.xrsh_io import HEADER_SIZE, parse_shard_bytes

import struct


def _write_str_u16(s: str) -> bytes:
    b = s.encode("utf-8")
    return len(b).to_bytes(2, "little") + b


def _write_prefix_list(pfx: list[str]) -> bytes:
    out = len(pfx).to_bytes(2, "little")
    for s in pfx:
        bb = s.encode("utf-8")
        out += bytes([len(bb)]) + bb
    return out


def _minimal_shard_bytes(
    *, vocab_hash32: bytes, games_body: bytes, file_version: int = 3
) -> bytes:
    assert len(vocab_hash32) == 32
    header = (
        b"XRSH"
        + int(file_version).to_bytes(4, "little")
        + vocab_hash32
        + (1).to_bytes(4, "little")
        + bytes(20)
    )
    assert len(header) == HEADER_SIZE
    return header + games_body


def test_parse_xrsh_v3_game_fields():
    moves_v = ["m0"]
    vh = bytes.fromhex(vocab_fingerprint_ordered_moves(moves_v))
    body = (
        _write_str_u16("game_c")
        + (1).to_bytes(4, "little")
        + _write_str_u16(START_FEN)
        + _write_str_u16(START_FEN)
        + _write_prefix_list([])
        + (0).to_bytes(4, "little", signed=True)
        + (1).to_bytes(2, "little")
        + (0).to_bytes(4, "little", signed=True)
        + (3).to_bytes(2, "little")
        + struct.pack("<fff", 0.1, 0.2, 0.3)
        + struct.pack("<bH", 1, 10)
    )
    blob = _minimal_shard_bytes(vocab_hash32=vh, games_body=body, file_version=3)
    rows, _ = parse_shard_bytes(blob)
    assert len(rows) == 1
    r = rows[0]
    assert r["game_result_red"] == 1
    assert r["ply_total"] == 10
    assert r["aux_attack"] == pytest.approx(0.1)
    assert r["aux_tactical"] == pytest.approx(0.3)


def test_policy_xrsh_dataset_smoke(tmp_path: Path) -> None:
    moves = ["m0"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {moves[0]: 0}

    d = tmp_path / "xrsh"
    d.mkdir()
    (d / "pack_meta.json").write_text(
        json.dumps(
            {
                "format": "xrsh_v3",
                "format_version": 3,
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
        + struct.pack("<fff", 0.25, 0.75, 0.33)
        + struct.pack("<bH", 1, 1)
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


def test_policy_xrsh_dataset_eager_cache_reuse(tmp_path: Path) -> None:
    moves = ["m0"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {moves[0]: 0}

    d = tmp_path / "xrsh_cache"
    d.mkdir()
    (d / "pack_meta.json").write_text(
        json.dumps(
            {
                "format": "xrsh_v3",
                "format_version": 3,
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
        + struct.pack("<fff", 0.25, 0.75, 0.33)
        + struct.pack("<bH", 1, 1)
    )
    vh = bytes.fromhex(fp_hex)
    (d / "shard_00000.xrsh").write_bytes(
        _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    )

    ds1 = PolicyXrshDataset(d, move_to_idx, storage_mode="eager")
    assert ds1.cache_built is True
    assert (d / ".cache" / "policy_xrsh_eager_v1.npz").is_file()

    ds2 = PolicyXrshDataset(d, move_to_idx, storage_mode="eager")
    assert ds2.cache_used is True


def test_policy_xrsh_dataset_filters_unknown_value_rows(tmp_path: Path) -> None:
    moves = ["m0"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {moves[0]: 0}

    d = tmp_path / "xrsh_unknown_value"
    d.mkdir()
    (d / "pack_meta.json").write_text(
        json.dumps(
            {
                "format": "xrsh_v3",
                "format_version": 3,
                "vocab_sha256": fp_hex,
                "shard_count": 1,
                "source": "test",
            }
        ),
        encoding="utf-8",
    )

    body = (
        _write_str_u16("g1")
        + (2).to_bytes(4, "little")
        # row 1: unknown result, should be filtered when with_value_labels=True
        + _write_str_u16(START_FEN)
        + _write_str_u16(START_FEN)
        + _write_prefix_list([])
        + (0).to_bytes(4, "little", signed=True)
        + (1).to_bytes(2, "little")
        + (0).to_bytes(4, "little", signed=True)
        + (0).to_bytes(2, "little")
        + struct.pack("<fff", 0.1, 0.2, 0.3)
        + struct.pack("<bH", 2, 2)
        # row 2: known result, should remain
        + _write_str_u16(START_FEN)
        + _write_str_u16(START_FEN)
        + _write_prefix_list([])
        + (0).to_bytes(4, "little", signed=True)
        + (1).to_bytes(2, "little")
        + (0).to_bytes(4, "little", signed=True)
        + (1).to_bytes(2, "little")
        + struct.pack("<fff", 0.2, 0.3, 0.4)
        + struct.pack("<bH", 1, 2)
    )
    vh = bytes.fromhex(fp_hex)
    shard = _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    (d / "shard_00000.xrsh").write_bytes(shard)

    ds = PolicyXrshDataset(
        d,
        move_to_idx,
        for_training=False,
        with_value_labels=True,
    )
    assert len(ds) == 1
    assert ds.filtered_unknown_value_rows == 1


def test_value_target_progress_weight() -> None:
    assert value_target_side_to_move(START_FEN, 1, 0, 5) == 0.0
    assert value_target_side_to_move(START_FEN, 1, 4, 5) == pytest.approx(1.0)
    assert value_target_side_to_move(START_FEN, 0, 2, 5) == pytest.approx(0.0)
    assert value_target_side_to_move(
        START_FEN, 1, 2, 5, progress_gamma=2.0
    ) == pytest.approx(0.25)
