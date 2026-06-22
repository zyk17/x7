"""XRSH v5 训练主线 IO（仅 v5 二进制）。"""

from __future__ import annotations

import hashlib
import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIC = b"XRSH"
HEADER_SIZE = 64
XRSH_V5 = 5


@dataclass(frozen=True)
class XrshRowRef:
    shard_index: int
    row_offset: int
    fen_key: int
    game_group: int
    ply: int
    game_result_red: int
    ply_total: int


@dataclass(frozen=True)
class XrshTrainRow:
    game_group: int
    fen: str
    legal_idx: list[int]
    target_idx: int
    ply: int
    search_q: float
    search_visits: int
    search_counts: list[int]


def read_str_u16(buf: bytes | mmap.mmap, off: int) -> tuple[str, int]:
    ln = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    return buf[off : off + ln].decode("utf-8"), off + ln


def read_prefix_list(buf: bytes | mmap.mmap, off: int) -> tuple[list[str], int]:
    n = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    out: list[str] = []
    for _ in range(n):
        bl = buf[off]
        off += 1
        out.append(buf[off : off + bl].decode("utf-8"))
        off += bl
    return out, off


def _skip_str_u16(buf: bytes | mmap.mmap, off: int) -> int:
    ln = int.from_bytes(buf[off : off + 2], "little")
    return off + 2 + ln


def _skip_prefix_list(buf: bytes | mmap.mmap, off: int) -> int:
    n = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    for _ in range(n):
        bl = buf[off]
        off += 1 + bl
    return off


def fen_key64(fen: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(fen.encode("utf-8"), digest_size=8).digest(),
        "little",
    )


def shard_file_version(buf: bytes) -> int:
    if len(buf) < 8 or buf[0:4] != MAGIC:
        raise ValueError("魔数非 XRSH")
    return int.from_bytes(buf[4:8], "little")


def assert_shard_binary_v5(path: Path | str) -> None:
    p = Path(path)
    with open(p, "rb") as fh:
        header = fh.read(8)
    if len(header) < 8 or header[0:4] != MAGIC:
        raise ValueError(f"非 XRSH 分片: {p}")
    ver = int.from_bytes(header[4:8], "little")
    if ver != XRSH_V5:
        raise ValueError(
            f"训练主线仅接受 XRSH v5 二进制分片，{p.name} 为 v{ver}；"
            f"请先完成 migrate verify + finalize"
        )


def parse_shard_bytes(buf: bytes) -> tuple[list[dict[str, Any]], bytes]:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"XRSH 分片过短: {len(buf)}")
    if buf[0:4] != MAGIC:
        raise ValueError("魔数非 XRSH")
    if shard_file_version(buf) != XRSH_V5:
        raise ValueError(f"训练主线仅支持 XRSH v5，当前版本: {shard_file_version(buf)}")
    vocab_hash = bytes(buf[8:40])
    n_games = int.from_bytes(buf[40:44], "little")
    off = HEADER_SIZE
    all_samples: list[dict[str, Any]] = []
    for _ in range(n_games):
        gid, off = read_str_u16(buf, off)
        n_rows = int.from_bytes(buf[off : off + 4], "little")
        off += 4
        for _ in range(n_rows):
            fen, off = read_str_u16(buf, off)
            root_fen, off = read_str_u16(buf, off)
            prefix, off = read_prefix_list(buf, off)
            target_idx = int.from_bytes(buf[off : off + 4], "little", signed=True)
            off += 4
            n_leg = int.from_bytes(buf[off : off + 2], "little")
            off += 2
            legal: list[int] = []
            for _ in range(n_leg):
                j = int.from_bytes(buf[off : off + 4], "little", signed=True)
                off += 4
                legal.append(j)
            ply = int.from_bytes(buf[off : off + 2], "little")
            off += 2
            if off + 11 + 2 * n_leg > len(buf):
                raise ValueError("XRSH v5 样本缺少 result/search 字段")
            gr, pt = struct.unpack_from("<bH", buf, off)
            off += 3
            search_q = struct.unpack_from("<f", buf, off)[0]
            off += 4
            search_visits = int.from_bytes(buf[off : off + 4], "little")
            off += 4
            search_counts: list[int] = []
            for _ in range(n_leg):
                search_counts.append(int.from_bytes(buf[off : off + 2], "little"))
                off += 2
            all_samples.append(
                {
                    "game_id": gid,
                    "fen": fen,
                    "root_fen": root_fen,
                    "uci_prefix": prefix,
                    "legal_idx": legal,
                    "target_idx": target_idx,
                    "ply": int(ply),
                    "game_result_red": int(gr),
                    "ply_total": int(pt),
                    "search_q": float(search_q),
                    "search_visits": search_visits,
                    "search_counts": search_counts,
                }
            )
    if off != len(buf):
        raise ValueError(f"XRSH 尾部长度不匹配: 解析到 {off} 总长 {len(buf)}")
    return all_samples, vocab_hash


def read_shard_file(path: Path | str) -> tuple[list[dict[str, Any]], bytes]:
    return parse_shard_bytes(Path(path).read_bytes())


def scan_shard_file(
    path: Path | str,
    *,
    shard_index: int,
    start_game_group: int = 0,
) -> tuple[list[XrshRowRef], bytes, int]:
    buf = Path(path).read_bytes()
    if shard_file_version(buf) != XRSH_V5:
        raise ValueError(f"训练主线仅支持 XRSH v5: {Path(path).name}")
    vocab_hash = bytes(buf[8:40])
    n_games = int.from_bytes(buf[40:44], "little")
    off = HEADER_SIZE
    refs: list[XrshRowRef] = []
    game_group = start_game_group
    for _ in range(n_games):
        _, off = read_str_u16(buf, off)
        n_rows = int.from_bytes(buf[off : off + 4], "little")
        off += 4
        for _ in range(n_rows):
            row_offset = off
            fen, off = read_str_u16(buf, off)
            off = _skip_str_u16(buf, off)
            off = _skip_prefix_list(buf, off)
            off += 4
            n_leg = int.from_bytes(buf[off : off + 2], "little")
            off += 2 + 4 * n_leg
            ply = int.from_bytes(buf[off : off + 2], "little")
            off += 2
            if off + 11 + 2 * n_leg > len(buf):
                raise ValueError("XRSH v5 样本缺少 result/search 字段")
            gr, pt = struct.unpack_from("<bH", buf, off)
            off += 3 + 4 + 4 + 2 * n_leg
            refs.append(
                XrshRowRef(
                    shard_index=shard_index,
                    row_offset=row_offset,
                    fen_key=fen_key64(fen),
                    game_group=game_group,
                    ply=int(ply),
                    game_result_red=int(gr),
                    ply_total=int(pt),
                )
            )
        game_group += 1
    if off != len(buf):
        raise ValueError(f"XRSH 尾部长度不匹配: 扫描到 {off} 总长 {len(buf)}")
    return refs, vocab_hash, game_group


def scan_shard_train_rows(
    path: Path | str,
    *,
    start_game_group: int = 0,
) -> tuple[list[XrshTrainRow], bytes, int]:
    buf = Path(path).read_bytes()
    if shard_file_version(buf) != XRSH_V5:
        raise ValueError(f"训练主线仅支持 XRSH v5: {Path(path).name}")
    vocab_hash = bytes(buf[8:40])
    n_games = int.from_bytes(buf[40:44], "little")
    off = HEADER_SIZE
    rows: list[XrshTrainRow] = []
    game_group = start_game_group
    for _ in range(n_games):
        off = _skip_str_u16(buf, off)
        n_rows = int.from_bytes(buf[off : off + 4], "little")
        off += 4
        for _ in range(n_rows):
            fen, legal_idx, target_idx, ply, search_q, search_visits, search_counts, off = (
                _read_row_train_fields(buf, off)
            )
            rows.append(
                XrshTrainRow(
                    game_group=game_group,
                    fen=fen,
                    legal_idx=legal_idx,
                    target_idx=target_idx,
                    ply=ply,
                    search_q=search_q,
                    search_visits=search_visits,
                    search_counts=search_counts,
                )
            )
        game_group += 1
    if off != len(buf):
        raise ValueError(f"XRSH 尾部长度不匹配: 扫描到 {off} 总长 {len(buf)}")
    return rows, vocab_hash, game_group


def read_row_at(buf: bytes | mmap.mmap, row_offset: int) -> dict[str, Any]:
    off = row_offset
    fen, off = read_str_u16(buf, off)
    root_fen, off = read_str_u16(buf, off)
    prefix, off = read_prefix_list(buf, off)
    target_idx = int.from_bytes(buf[off : off + 4], "little", signed=True)
    off += 4
    n_leg = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    legal: list[int] = []
    for _ in range(n_leg):
        j = int.from_bytes(buf[off : off + 4], "little", signed=True)
        off += 4
        legal.append(j)
    ply = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    gr, pt = struct.unpack_from("<bH", buf, off)
    off += 3
    search_q = struct.unpack_from("<f", buf, off)[0]
    off += 4
    search_visits = int.from_bytes(buf[off : off + 4], "little")
    off += 4
    search_counts: list[int] = []
    for _ in range(n_leg):
        search_counts.append(int.from_bytes(buf[off : off + 2], "little"))
        off += 2
    return {
        "fen": fen,
        "root_fen": root_fen,
        "uci_prefix": prefix,
        "legal_idx": legal,
        "target_idx": target_idx,
        "ply": int(ply),
        "game_result_red": int(gr),
        "ply_total": int(pt),
        "search_q": float(search_q),
        "search_visits": search_visits,
        "search_counts": search_counts,
    }


def _read_row_train_fields(
    buf: bytes | mmap.mmap,
    row_offset: int,
) -> tuple[str, list[int], int, int, float, int, list[int], int]:
    off = row_offset
    fen, off = read_str_u16(buf, off)
    off = _skip_str_u16(buf, off)
    off = _skip_prefix_list(buf, off)
    target_idx = int.from_bytes(buf[off : off + 4], "little", signed=True)
    off += 4
    n_leg = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    legal_idx: list[int] = []
    for _ in range(n_leg):
        legal_idx.append(int.from_bytes(buf[off : off + 4], "little", signed=True))
        off += 4
    ply = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    if off + 11 + 2 * n_leg > len(buf):
        raise ValueError("XRSH v5 样本缺少 result/search 字段")
    off += 3
    search_q = struct.unpack_from("<f", buf, off)[0]
    off += 4
    search_visits = int.from_bytes(buf[off : off + 4], "little")
    off += 4
    search_counts: list[int] = []
    for _ in range(n_leg):
        search_counts.append(int.from_bytes(buf[off : off + 2], "little"))
        off += 2
    return (
        fen,
        legal_idx,
        target_idx,
        ply,
        float(search_q),
        search_visits,
        search_counts,
        off,
    )


def read_row_train_at(
    buf: bytes | mmap.mmap,
    row_offset: int,
) -> tuple[str, list[int], int, float, int, list[int]]:
    fen, legal_idx, target_idx, _ply, search_q, search_visits, search_counts, _off = (
        _read_row_train_fields(buf, row_offset)
    )
    return (
        fen,
        legal_idx,
        target_idx,
        search_q,
        search_visits,
        search_counts,
    )


def load_pack_meta(pack_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(pack_dir) / "pack_meta.json").read_text(encoding="utf-8"))


def xrsh_dir_is_complete(pack_dir: Path | str) -> bool:
    d = Path(pack_dir)
    return (d / "pack_meta.json").is_file() and any(d.glob("shard_*.xrsh"))
