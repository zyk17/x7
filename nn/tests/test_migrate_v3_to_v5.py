"""XRSH v3→v5 迁移：字段保真 + stage/verify 流程测试。"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from constants import START_FEN
from migrate_xrsh_v3_to_v5 import (
    compare_policy_rows,
    finalize_dir,
    stage_dir,
    staging_dir,
    verify_dir,
    verify_shard_pair,
)
from nn.xrsh_io import load_pack_meta, shard_file_version
from nn.xrsh_migrate_io import (
    XRSH_V5,
    encode_shard_v5,
    parse_shard_bytes,
    samples_to_games,
    write_pack_meta,
)
from nn.policy_pack import vocab_fingerprint_ordered_moves
from test_xrsh_io import _minimal_shard_bytes, _write_prefix_list, _write_str_u16


def _v3_row_bytes(
    *,
    target_idx: int = 7,
    ply: int = 3,
    game_result_red: int = 1,
    ply_total: int = 40,
    legal: list[int] | None = None,
) -> bytes:
    legal = legal if legal is not None else [0, 1, 2]
    body = (
        _write_str_u16(START_FEN)
        + _write_str_u16(START_FEN)
        + _write_prefix_list(["a0a1"])
        + int(target_idx).to_bytes(4, "little", signed=True)
        + len(legal).to_bytes(2, "little")
    )
    for idx in legal:
        body += int(idx).to_bytes(4, "little", signed=True)
    body += int(ply).to_bytes(2, "little")
    body += struct.pack("<fff", 0.1, 0.2, 0.3)
    body += struct.pack("<bH", game_result_red, ply_total)
    return body


def _make_v3_shard(tmp: Path, *, n_rows: int = 3) -> Path:
    moves = ["m0", "m1", "m2"]
    vh = bytes.fromhex(vocab_fingerprint_ordered_moves(moves))
    rows_body = b""
    for i in range(n_rows):
        rows_body += _v3_row_bytes(target_idx=i, ply=i, legal=[0, 1, 2])
    body = _write_str_u16("game_a") + (n_rows).to_bytes(4, "little") + rows_body
    shard = tmp / "shard_00000.xrsh"
    shard.write_bytes(_minimal_shard_bytes(vocab_hash32=vh, games_body=body, file_version=3))
    write_pack_meta(
        tmp,
        vocab_hash=vh,
        shard_count=1,
        source="test:v3",
    )
    return shard


def test_v3_to_v5_roundtrip_preserves_policy_fields() -> None:
    moves = ["m0", "m1", "m2"]
    vh = bytes.fromhex(vocab_fingerprint_ordered_moves(moves))
    body = _write_str_u16("g1") + (2).to_bytes(4, "little") + _v3_row_bytes(ply=0) + _v3_row_bytes(ply=1)
    v3_buf = _minimal_shard_bytes(vocab_hash32=vh, games_body=body, file_version=3)
    v3_rows, _ = parse_shard_bytes(v3_buf)
    v5_buf = encode_shard_v5(samples_to_games(v3_rows), vocab_hash=vh)
    v5_rows, _ = parse_shard_bytes(v5_buf)
    assert len(v3_rows) == len(v5_rows) == 2
    for a, b in zip(v3_rows, v5_rows):
        assert compare_policy_rows(a, b) == []


def test_stage_and_verify_local_pack(tmp_path: Path) -> None:
    pack = tmp_path / "xrsh"
    pack.mkdir()
    _make_v3_shard(pack, n_rows=4)
    n = stage_dir(pack, resume=False)
    assert n == 1
    assert (staging_dir(pack) / "shard_00000.xrsh").is_file()
    assert shard_file_version((staging_dir(pack) / "shard_00000.xrsh").read_bytes()[:8]) == XRSH_V5
    assert verify_dir(pack)
    # 源 v3 分片应仍在
    assert shard_file_version((pack / "shard_00000.xrsh").read_bytes()[:8]) == 3


def test_finalize_replaces_in_place(tmp_path: Path) -> None:
    pack = tmp_path / "xrsh"
    pack.mkdir()
    _make_v3_shard(pack)
    stage_dir(pack, resume=False)
    assert verify_dir(pack)
    finalize_dir(pack, in_place=True, force=False)
    assert not staging_dir(pack).exists()
    assert shard_file_version((pack / "shard_00000.xrsh").read_bytes()[:8]) == XRSH_V5
    meta = json.loads((pack / "pack_meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "xrsh_v5"


@pytest.mark.skipif(
    not Path(r"C:\projects\77xiangqi_engine\data\xrsh_train\shard_00000.xrsh").is_file(),
    reason="本地大师数据不存在",
)
def test_verify_real_shard_00000_smoke() -> None:
    src = Path(r"C:\projects\77xiangqi_engine\data\xrsh_train")
    staging = staging_dir(src)
    v3 = src / "shard_00000.xrsh"
    v5 = staging / "shard_00000.xrsh"
    if not v5.is_file():
        pytest.skip("staging 尚无 shard_00000")
    assert verify_shard_pair(v3, v5) == []
