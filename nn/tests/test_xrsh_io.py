"""XRSH v5 二进制解析与 Dataset 冒烟。"""

from __future__ import annotations

import json
import struct
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


def _minimal_shard_bytes(
    *,
    vocab_hash32: bytes,
    games_body: bytes,
    file_version: int = 5,
) -> bytes:
    header = (
        b"XRSH"
        + int(file_version).to_bytes(4, "little")
        + vocab_hash32
        + (1).to_bytes(4, "little")
        + bytes(20)
    )
    assert len(header) == HEADER_SIZE
    return header + games_body


def _row_bytes(
    *,
    target_idx: int = 0,
    ply: int = 0,
    game_result_red: int = 1,
    ply_total: int = 1,
    search_q: float = 0.0,
    search_visits: int = 0,
    search_count: int = 0,
) -> bytes:
    return (
        _write_str_u16(START_FEN)
        + _write_str_u16(START_FEN)
        + _write_prefix_list([])
        + int(target_idx).to_bytes(4, "little", signed=True)
        + (1).to_bytes(2, "little")
        + (0).to_bytes(4, "little", signed=True)
        + int(ply).to_bytes(2, "little")
        + struct.pack("<bH", game_result_red, ply_total)
        + struct.pack("<fI", search_q, search_visits)
        + int(search_count).to_bytes(2, "little")
    )


def _write_pack_meta(root: Path, fp_hex: str) -> None:
    (root / "pack_meta.json").write_text(
        json.dumps(
            {
                "format": "xrsh_v5",
                "format_version": 5,
                "vocab_sha256": fp_hex,
                "shard_count": 1,
                "source": "test",
            }
        ),
        encoding="utf-8",
    )


def test_parse_xrsh_v5_game_fields() -> None:
    moves_v = ["m0"]
    vh = bytes.fromhex(vocab_fingerprint_ordered_moves(moves_v))
    body = _write_str_u16("game_c") + (1).to_bytes(4, "little") + _row_bytes(
        ply=3,
        game_result_red=1,
        ply_total=10,
        search_q=0.6,
        search_visits=32,
        search_count=12,
    )
    rows, _ = parse_shard_bytes(_minimal_shard_bytes(vocab_hash32=vh, games_body=body))
    assert len(rows) == 1
    r = rows[0]
    assert r["game_result_red"] == 1
    assert r["ply_total"] == 10
    assert r["search_q"] == pytest.approx(0.6)
    assert r["search_visits"] == 32
    assert r["search_counts"] == [12]


def test_policy_xrsh_dataset_smoke(tmp_path: Path) -> None:
    moves = ["m0"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {moves[0]: 0}

    d = tmp_path / "xrsh"
    d.mkdir()
    _write_pack_meta(d, fp_hex)
    body = _write_str_u16("g1") + (1).to_bytes(4, "little") + _row_bytes(
        ply=1,
        game_result_red=1,
        ply_total=1,
        search_q=0.5,
        search_visits=16,
        search_count=16,
    )
    vh = bytes.fromhex(fp_hex)
    (d / "shard_00000.xrsh").write_bytes(
        _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    )

    ds = PolicyXrshDataset(d, move_to_idx, for_training=True)
    assert len(ds) == 1
    boards, masks, targets, weights = ds[0]
    assert boards.ndim == 3
    assert tuple(masks.shape) == (1,)
    assert int(targets.item()) == 0
    assert float(weights.item()) > 0


def test_policy_xrsh_dataset_eager_cache_reuse(tmp_path: Path) -> None:
    moves = ["m0"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {moves[0]: 0}

    d = tmp_path / "xrsh_cache"
    d.mkdir()
    _write_pack_meta(d, fp_hex)
    body = _write_str_u16("g1") + (1).to_bytes(4, "little") + _row_bytes(
        game_result_red=1,
        search_q=0.5,
        search_visits=16,
        search_count=16,
    )
    vh = bytes.fromhex(fp_hex)
    (d / "shard_00000.xrsh").write_bytes(
        _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    )

    ds1 = PolicyXrshDataset(d, move_to_idx, storage_mode="eager")
    assert ds1.cache_built is True
    assert (d / ".cache" / "policy_xrsh_eager_v4.npz").is_file()

    ds2 = PolicyXrshDataset(d, move_to_idx, storage_mode="eager")
    assert ds2.cache_used is True


def test_policy_xrsh_dataset_search_labels(tmp_path: Path) -> None:
    moves = ["m0"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {moves[0]: 0}

    d = tmp_path / "xrsh_search"
    d.mkdir()
    _write_pack_meta(d, fp_hex)
    body = _write_str_u16("g1") + (1).to_bytes(4, "little") + _row_bytes(
        game_result_red=1,
        search_q=0.75,
        search_visits=24,
        search_count=24,
    )
    vh = bytes.fromhex(fp_hex)
    (d / "shard_00000.xrsh").write_bytes(
        _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    )

    ds = PolicyXrshDataset(d, move_to_idx, with_search_labels=True)
    board, mask, target, weight, visit_target, search_q, search_visits = ds[0]
    assert tuple(mask.shape) == (1,)
    assert visit_target[0].item() == pytest.approx(1.0)
    assert search_q.item() == pytest.approx(0.75)
    assert int(search_visits.item()) == 24


def test_policy_xrsh_dataset_search_q_value_target(tmp_path: Path) -> None:
    moves = ["m0"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {moves[0]: 0}

    d = tmp_path / "xrsh_search_q"
    d.mkdir()
    _write_pack_meta(d, fp_hex)
    body = _write_str_u16("g1") + (1).to_bytes(4, "little") + _row_bytes(
        search_q=0.33,
        search_visits=12,
        search_count=12,
    )
    vh = bytes.fromhex(fp_hex)
    (d / "shard_00000.xrsh").write_bytes(
        _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    )

    ds = PolicyXrshDataset(
        d,
        move_to_idx,
        with_value_labels=True,
    )
    _, _, _, _, value = ds[0]
    assert value.item() == pytest.approx(0.33)
