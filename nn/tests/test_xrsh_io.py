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
from nn.dataset_batch import (
    SAMPLE_BOARD,
    SAMPLE_BOARD90,
    SAMPLE_LEGAL_IDX,
    SAMPLE_MASK,
    SAMPLE_SEARCH_VISITS,
    SAMPLE_SEARCH_COUNTS,
    SAMPLE_STM,
    SAMPLE_T_VAL,
    SAMPLE_TARGET,
    SAMPLE_VISIT_TARGET,
    SAMPLE_VOCAB_SIZE,
    SAMPLE_WEIGHT,
    collate_xrsh_samples,
)
from nn.dataset_xrsh import PolicyXrshDataset
from nn.policy_pack import vocab_fingerprint_ordered_moves
from nn.xrsh_io import HEADER_SIZE, parse_shard_bytes, read_row_train_at
from nn.xrsh_migrate_io import parse_shard_bytes as parse_shard_bytes_migrate


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
    sample = ds[0]
    assert sample[SAMPLE_BOARD90].shape == (90,)
    assert int(sample[SAMPLE_STM].item()) in (0, 1)
    batch = collate_xrsh_samples([sample])
    assert batch[SAMPLE_BOARD].ndim == 4
    assert tuple(batch[SAMPLE_MASK].shape) == (1, 1)
    assert int(sample[SAMPLE_TARGET].item()) == 0
    assert float(sample[SAMPLE_WEIGHT].item()) > 0


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
    assert (d / ".cache" / "policy_xrsh_eager_v5.npz").is_file()

    ds2 = PolicyXrshDataset(d, move_to_idx, storage_mode="eager")
    assert ds2.cache_used is True


def test_read_row_train_at_reads_without_dict_roundtrip() -> None:
    body = _write_str_u16("g1") + (1).to_bytes(4, "little") + _row_bytes(
        target_idx=7,
        ply=4,
        search_q=0.25,
        search_visits=13,
        search_count=11,
    )
    buf = _minimal_shard_bytes(vocab_hash32=bytes(32), games_body=body)
    row_offset = HEADER_SIZE + len(_write_str_u16("g1")) + 4
    fen, legal_idx, target_idx, search_q, search_visits, search_counts = read_row_train_at(
        buf,
        row_offset,
    )
    assert fen == START_FEN
    assert legal_idx == [0]
    assert target_idx == 7
    assert search_q == pytest.approx(0.25)
    assert search_visits == 13
    assert search_counts == [11]


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
    sample = ds[0]
    batch = collate_xrsh_samples([sample])
    assert tuple(batch[SAMPLE_MASK].shape) == (1, 1)
    assert batch[SAMPLE_VISIT_TARGET][0, 0].item() == pytest.approx(1.0)


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
    sample = ds[0]
    assert sample[SAMPLE_T_VAL].item() == pytest.approx(0.33)
    assert int(sample[SAMPLE_SEARCH_VISITS].item()) == 12


def test_parse_xrsh_v3_maps_search_defaults() -> None:
    moves_v = ["m0"]
    vh = bytes.fromhex(vocab_fingerprint_ordered_moves(moves_v))
    body = (
        _write_str_u16("game_v3")
        + (1).to_bytes(4, "little")
        + _write_str_u16(START_FEN)
        + _write_str_u16(START_FEN)
        + _write_prefix_list([])
        + (0).to_bytes(4, "little", signed=True)
        + (1).to_bytes(2, "little")
        + (0).to_bytes(4, "little", signed=True)
        + (0).to_bytes(2, "little")
        + struct.pack("<fff", 0.0, 0.0, 0.0)
        + struct.pack("<bH", 1, 10)
    )
    rows, _ = parse_shard_bytes_migrate(
        _minimal_shard_bytes(vocab_hash32=vh, games_body=body, file_version=3)
    )
    assert rows[0]["game_result_red"] == 1
    assert rows[0]["ply_total"] == 10
    assert rows[0]["search_visits"] == 0
    assert rows[0]["search_counts"] == [0]


def test_lazy_mirror_value_batch_includes_search_visits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """镜像增强 + value_head 时 batch 须含 search_visits（lazy 训练路径）。"""
    from nn.dataset_xrsh import PolicyXrshDataset

    moves = ["h2e2", "h7e7"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {m: i for i, m in enumerate(moves)}

    d = tmp_path / "xrsh_mirror"
    d.mkdir()
    _write_pack_meta(d, fp_hex)
    body = _write_str_u16("g1") + (1).to_bytes(4, "little") + _row_bytes(
        target_idx=0,
        search_q=0.25,
        search_visits=8,
        search_count=8,
    )
    vh = bytes.fromhex(fp_hex)
    (d / "shard_00000.xrsh").write_bytes(
        _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    )

    monkeypatch.setattr("nn.dataset_xrsh.random.random", lambda: 0.0)
    ds = PolicyXrshDataset(
        d,
        move_to_idx,
        for_training=True,
        with_value_labels=True,
        storage_mode="lazy",
    )
    sample = ds[0]
    assert SAMPLE_T_VAL in sample
    assert SAMPLE_SEARCH_VISITS in sample
    assert sample[SAMPLE_T_VAL].shape == sample[SAMPLE_SEARCH_VISITS].shape == ()


def test_eager_mirror_updates_target_and_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moves = ["h2e2", "b2e2"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {m: i for i, m in enumerate(moves)}

    d = tmp_path / "xrsh_mirror_eager"
    d.mkdir()
    _write_pack_meta(d, fp_hex)
    body = _write_str_u16("g1") + (1).to_bytes(4, "little") + _row_bytes(
        target_idx=0,
        search_q=0.25,
        search_visits=8,
        search_count=8,
    )
    vh = bytes.fromhex(fp_hex)
    (d / "shard_00000.xrsh").write_bytes(
        _minimal_shard_bytes(vocab_hash32=vh, games_body=body)
    )

    monkeypatch.setattr("nn.dataset_xrsh.random.random", lambda: 0.0)
    ds = PolicyXrshDataset(
        d,
        move_to_idx,
        for_training=True,
        storage_mode="eager",
    )
    sample = ds[0]
    batch = collate_xrsh_samples([sample])
    assert int(batch[SAMPLE_TARGET].item()) == 1
    assert batch[SAMPLE_MASK][0].tolist() == [False, True]


def test_dataset_rejects_v3_binary_with_v5_meta(tmp_path: Path) -> None:
    moves = ["m0"]
    fp_hex = vocab_fingerprint_ordered_moves(moves)
    move_to_idx = {moves[0]: 0}

    d = tmp_path / "xrsh_bad"
    d.mkdir()
    _write_pack_meta(d, fp_hex)
    body = _write_str_u16("g1") + (1).to_bytes(4, "little") + _row_bytes()
    vh = bytes.fromhex(fp_hex)
    (d / "shard_00000.xrsh").write_bytes(
        _minimal_shard_bytes(vocab_hash32=vh, games_body=body, file_version=3)
    )

    with pytest.raises(ValueError, match="v3"):
        PolicyXrshDataset(d, move_to_idx)
