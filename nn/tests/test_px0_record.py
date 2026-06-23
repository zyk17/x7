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
        0,
        0,
        0,
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
    assert sample.plies_left.shape == (1,)
    assert sample.winner_wdl.sum() == pytest.approx(1.0)
    assert sample.search_wdl.sum() == pytest.approx(1.0)
    assert sample.search_wdl[0] - sample.search_wdl[2] == pytest.approx(sample.search_q[0])


def test_iter_px0_chunk_file_reads_gzip(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk_000.gz"
    with gzip.open(chunk, "wb") as fh:
        fh.write(_fake_v6_record())
    items = list(iter_px0_chunk_file(chunk))
    assert len(items) == 1
    assert items[0].policy[7] == 1.0
    assert V6_RECORD_SIZE == struct.calcsize("<ii8248s1920sBBBb15fIHH4H")


def test_parse_v6_record_rejects_non_classical_input() -> None:
    record = bytearray(_fake_v6_record())
    struct.pack_into("<i", record, 4, 3)
    try:
        parse_v6_record(bytes(record))
    except ValueError as exc:
        assert "input_format=1" in str(exc)
    else:
        raise AssertionError("expected non-classical input to be rejected")
