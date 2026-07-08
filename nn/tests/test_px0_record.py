from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from nn.px0_record import (
    PX0_PLANES,
    PX0_POLICY_SIZE,
    V6_RECORD_SIZE,
    V6_STRUCT,
    iter_px0_chunk_file,
    parse_v6_record,
)


def _official_like_decode(
    record: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        version,
        input_format,
        probs_raw,
        packed_planes,
        stm,
        rule50_count,
        invariance_info,
        _dep_result,
        _root_q,
        best_q,
        _root_d,
        best_d,
        _root_m,
        _best_m,
        plies_left,
        result_q,
        result_d,
        _played_q,
        _played_d,
        _played_m,
        _orig_q,
        _orig_d,
        _orig_m,
        visits,
        _played_idx,
        _best_idx,
        policy_kld,
        _reserved,
    ) = V6_STRUCT.unpack(record)
    assert version == 6
    assert input_format == 1

    planes = np.unpackbits(np.frombuffer(packed_planes, dtype=np.uint8), bitorder="little")
    planes = planes.reshape((-1, 128))[:, :90].reshape((-1, 10, 9)).astype(np.float32)
    stm_plane = np.full((1, 10, 9), float(stm), dtype=np.float32)
    rule50_plane = np.full((1, 10, 9), float(rule50_count), dtype=np.float32)
    aux_plane = np.zeros((1, 10, 9), dtype=np.float32)
    if invariance_info >= 128:
        aux_plane.fill(1.0)
    edge_plane = np.ones((1, 10, 9), dtype=np.float32)
    full_planes = np.concatenate([planes, stm_plane, rule50_plane, aux_plane, edge_plane], axis=0)

    winner = np.asarray(
        [
            0.5 * (1.0 - result_d + result_q),
            result_d,
            0.5 * (1.0 - result_d - result_q),
        ],
        dtype=np.float32,
    )
    best = np.asarray(
        [
            0.5 * (1.0 - best_d + best_q),
            best_d,
            0.5 * (1.0 - best_d - best_q),
        ],
        dtype=np.float32,
    )
    policy = np.frombuffer(probs_raw, dtype=np.float32).copy()
    return (
        full_planes,
        policy,
        winner,
        best,
        np.asarray([float(visits)], dtype=np.float32),
        np.asarray([float(policy_kld)], dtype=np.float32),
        np.asarray([float(plies_left)], dtype=np.float32),
    )


def _fake_v6_record() -> bytes:
    probs = np.zeros(PX0_POLICY_SIZE, dtype=np.float32)
    probs[7] = 1.0
    packed_planes = bytes(1920)
    return V6_STRUCT.pack(
        6,
        1,
        probs.tobytes(),
        packed_planes,
        1,
        12,
        0,
        0,
        0.0,
        0.4,
        0.0,
        0.1,
        0.0,
        0.0,
        23.0,
        0.8,
        0.0,
        0.0,
        0.0,
        0.0,
        float("nan"),
        float("nan"),
        float("nan"),
        128,
        0,
        7,
        0.75,
        0,
    )


def test_parse_v6_record_shapes() -> None:
    sample = parse_v6_record(_fake_v6_record())
    assert sample.planes.shape == (PX0_PLANES, 10, 9)
    assert sample.policy.shape == (PX0_POLICY_SIZE,)
    assert sample.winner_q.shape == (1,)
    assert sample.winner_wdl.shape == (3,)
    assert sample.search_q.shape == (1,)
    assert sample.search_wdl.shape == (3,)
    assert sample.search_visits.shape == (1,)
    assert sample.policy_kld.shape == (1,)
    assert sample.plies_left.shape == (1,)
    assert sample.winner_wdl.sum() == pytest.approx(1.0)
    assert sample.search_wdl.sum() == pytest.approx(1.0)
    assert sample.search_wdl[0] - sample.search_wdl[2] == pytest.approx(sample.search_q[0])


def test_parse_v6_record_matches_official_chunkparser_semantics() -> None:
    record = _fake_v6_record()
    sample = parse_v6_record(record)
    planes, policy, winner, best, visits, policy_kld, plies_left = _official_like_decode(record)
    np.testing.assert_allclose(sample.planes, planes)
    np.testing.assert_allclose(sample.policy, policy)
    np.testing.assert_allclose(sample.winner_wdl, winner)
    np.testing.assert_allclose(sample.search_wdl, best)
    np.testing.assert_allclose(sample.search_visits, visits)
    np.testing.assert_allclose(sample.policy_kld, policy_kld)
    np.testing.assert_allclose(sample.plies_left, plies_left)


def test_iter_px0_chunk_file_reads_gzip(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk_000.gz"
    with gzip.open(chunk, "wb") as fh:
        fh.write(_fake_v6_record())
    items = list(iter_px0_chunk_file(chunk))
    assert len(items) == 1
    assert items[0].policy[7] == 1.0
    assert V6_RECORD_SIZE == struct.calcsize("<ii8248s1920sBBBb15fIHHfI")


def test_parse_v6_record_rejects_non_classical_input() -> None:
    record = bytearray(_fake_v6_record())
    struct.pack_into("<i", record, 4, 3)
    try:
        parse_v6_record(bytes(record))
    except ValueError as exc:
        assert "input_format=1" in str(exc)
    else:
        raise AssertionError("expected non-classical input to be rejected")
