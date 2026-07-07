"""PX0 v6 chunk 解析。

当前只支持最小主线需要的字段：

- 124 x 10 x 9 输入平面
- 2062 维 policy 概率
- winner WDL
- search q WDL（来自 best_q / best_d）
- plies_left

这里不试图兼容 lc0/px0 全部历史版本，也不引入 proto 依赖。
"""

from __future__ import annotations

import gzip
import glob
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

V6_VERSION = 6
CLASSICAL_INPUT = 1
V6_STRUCT = struct.Struct("<ii8248s1920sBBBb15fIHH4H")
V6_RECORD_SIZE = V6_STRUCT.size
PX0_PLANES = 124
PX0_ROWS = 10
PX0_COLS = 9
PX0_POLICY_SIZE = 2062


@dataclass(frozen=True)
class Px0Sample:
    planes: np.ndarray
    policy: np.ndarray
    winner_q: np.ndarray
    winner_wdl: np.ndarray
    search_q: np.ndarray
    search_wdl: np.ndarray
    plies_left: np.ndarray


def _winner_wdl(result_q: float, result_d: float) -> np.ndarray:
    win = 0.5 * (1.0 - result_d + result_q)
    loss = 0.5 * (1.0 - result_d - result_q)
    return np.asarray([win, result_d, loss], dtype=np.float32)


def _search_wdl(best_q: float, best_d: float) -> np.ndarray:
    win = 0.5 * (1.0 - best_d + best_q)
    loss = 0.5 * (1.0 - best_d - best_q)
    return np.asarray([win, best_d, loss], dtype=np.float32)


def _decode_planes(
    packed_planes: bytes,
    *,
    stm: int,
    rule50_count: int,
    invariance_info: int,
    input_format: int,
) -> np.ndarray:
    planes = np.unpackbits(
        np.frombuffer(packed_planes, dtype=np.uint8),
        bitorder="little",
    ).reshape((-1, 128))[:, : (PX0_ROWS * PX0_COLS)]
    planes = planes.reshape((-1, PX0_ROWS, PX0_COLS)).astype(np.float32)

    rule50_plane = np.full(
        (1, PX0_ROWS, PX0_COLS),
        float(rule50_count) / 120.0 if input_format > 3 else float(rule50_count),
        dtype=np.float32,
    )

    aux_plane = np.zeros((1, PX0_ROWS, PX0_COLS), dtype=np.float32)
    if input_format in (132, 133) and invariance_info >= 128:
        aux_plane.fill(1.0)

    edge_plane = np.ones((1, PX0_ROWS, PX0_COLS), dtype=np.float32)

    if input_format != CLASSICAL_INPUT:
        raise ValueError(
            f"当前仅支持 px0 classical input_format=1, got input_format={input_format}"
        )
    stm_planes = np.full((1, PX0_ROWS, PX0_COLS), float(stm), dtype=np.float32)

    out = np.concatenate([planes, stm_planes, rule50_plane, aux_plane, edge_plane], axis=0)
    if out.shape != (PX0_PLANES, PX0_ROWS, PX0_COLS):
        raise ValueError(f"unexpected planes shape: {out.shape}")
    return out


def parse_v6_record(record: bytes) -> Px0Sample:
    if len(record) != V6_RECORD_SIZE:
        raise ValueError(f"record size mismatch: {len(record)} != {V6_RECORD_SIZE}")

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
        _visits,
        _played_idx,
        _best_idx,
        _reserved1,
        _reserved2,
        _reserved3,
        _reserved4,
    ) = V6_STRUCT.unpack(record)

    if version != V6_VERSION:
        raise ValueError(f"unsupported v{version}; only v6 is supported")

    policy = np.frombuffer(probs_raw, dtype=np.float32).copy()
    if policy.shape != (PX0_POLICY_SIZE,):
        raise ValueError(f"unexpected policy shape: {policy.shape}")

    planes = _decode_planes(
        packed_planes,
        stm=int(stm),
        rule50_count=int(rule50_count),
        invariance_info=int(invariance_info),
        input_format=int(input_format),
    )
    winner = _winner_wdl(float(result_q), float(result_d))
    search = _search_wdl(float(best_q), float(best_d))
    plies = np.asarray([float(plies_left)], dtype=np.float32)
    return Px0Sample(
        planes=planes,
        policy=policy,
        winner_q=np.asarray([float(result_q)], dtype=np.float32),
        winner_wdl=winner,
        search_q=np.asarray([float(best_q)], dtype=np.float32),
        search_wdl=search,
        plies_left=plies,
    )


def iter_px0_chunk_file(path: Path | str) -> Iterator[Px0Sample]:
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        version_raw = fh.read(4)
        if len(version_raw) != 4:
            return
        version = struct.unpack("<i", version_raw)[0]
        if version != V6_VERSION:
            raise ValueError(f"{path} version={version}; 当前仅支持 v6 gz chunks")
        fh.seek(0)
        while True:
            record = fh.read(V6_RECORD_SIZE)
            if not record:
                return
            if len(record) != V6_RECORD_SIZE:
                raise ValueError(f"{path} contains truncated record")
            yield parse_v6_record(record)


def expand_chunk_globs(patterns: list[str], *, max_files: int = 0) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p).resolve() for p in glob.glob(pattern, recursive=True))
    unique = sorted({p.resolve() for p in files if p.is_file()})
    if max_files > 0:
        return unique[:max_files]
    return unique
