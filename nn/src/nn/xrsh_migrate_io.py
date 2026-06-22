"""XRSH v3→v5 迁移专用 IO（不进训练热路径）。"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from nn.xrsh_io import (
    HEADER_SIZE,
    MAGIC,
    read_prefix_list,
    read_str_u16,
    shard_file_version,
)

XRSH_V3 = 3
XRSH_V5 = 5

__all__ = [
    "XRSH_V3",
    "XRSH_V5",
    "encode_shard_v5",
    "parse_shard_bytes",
    "samples_to_games",
    "shard_file_version",
    "write_pack_meta",
]


def _write_str_u16(s: str) -> bytes:
    b = s.encode("utf-8")
    return len(b).to_bytes(2, "little") + b


def _write_prefix_list(pfx: list[str]) -> bytes:
    out = len(pfx).to_bytes(2, "little")
    for s in pfx:
        bb = s.encode("utf-8")
        out += bytes([len(bb)]) + bb
    return out


def _write_row_v5(row: dict[str, Any]) -> bytes:
    legal = [int(x) for x in row["legal_idx"]]
    counts = [int(x) for x in row.get("search_counts") or [0] * len(legal)]
    if len(counts) != len(legal):
        raise ValueError("search_counts 与 legal_idx 长度不一致")
    body = (
        _write_str_u16(str(row["fen"]))
        + _write_str_u16(str(row["root_fen"]))
        + _write_prefix_list(list(row.get("uci_prefix") or []))
        + int(row["target_idx"]).to_bytes(4, "little", signed=True)
        + len(legal).to_bytes(2, "little")
    )
    for idx in legal:
        body += int(idx).to_bytes(4, "little", signed=True)
    body += int(row.get("ply", 0)).to_bytes(2, "little")
    body += struct.pack(
        "<bHfI",
        int(row.get("game_result_red", 2)),
        int(row.get("ply_total", 0)),
        float(row.get("search_q", 0.0) or 0.0),
        int(row.get("search_visits", 0) or 0),
    )
    for count in counts:
        body += int(count).to_bytes(2, "little")
    return body


def encode_shard_v5(
    games: list[tuple[str, list[dict[str, Any]]]],
    *,
    vocab_hash: bytes,
) -> bytes:
    if len(vocab_hash) != 32:
        raise ValueError("vocab_hash 须为 32 字节")
    header = (
        MAGIC
        + int(XRSH_V5).to_bytes(4, "little")
        + vocab_hash
        + len(games).to_bytes(4, "little")
        + bytes(20)
    )
    body = b""
    for gid, rows in games:
        body += _write_str_u16(gid)
        body += len(rows).to_bytes(4, "little")
        for row in rows:
            body += _write_row_v5(row)
    return header + body


def _parse_shard_v3(buf: bytes) -> tuple[list[dict[str, Any]], bytes]:
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
            if off + 15 > len(buf):
                raise ValueError("XRSH v3 样本缺少 aux/result 等 15 字节")
            off += 12
            gr, pt = struct.unpack_from("<bH", buf, off)
            off += 3
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
                    "search_q": 0.0,
                    "search_visits": 0,
                    "search_counts": [0] * len(legal),
                }
            )
    if off != len(buf):
        raise ValueError(f"XRSH 尾部长度不匹配: 解析到 {off} 总长 {len(buf)}")
    return all_samples, vocab_hash


def parse_shard_bytes(buf: bytes) -> tuple[list[dict[str, Any]], bytes]:
    if len(buf) < HEADER_SIZE or buf[0:4] != MAGIC:
        raise ValueError("魔数非 XRSH")
    ver = shard_file_version(buf)
    if ver == XRSH_V5:
        from nn.xrsh_io import parse_shard_bytes as parse_v5_only

        return parse_v5_only(buf)
    if ver == XRSH_V3:
        return _parse_shard_v3(buf)
    raise ValueError(f"不支持的 XRSH 文件版本: {ver}")


def samples_to_games(samples: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in samples:
        gid = str(row.get("game_id", ""))
        if gid not in grouped:
            order.append(gid)
        grouped[gid].append(row)
    return [(gid, grouped[gid]) for gid in order]


def write_pack_meta(
    out_dir: Path | str,
    *,
    vocab_hash: bytes,
    shard_count: int,
    source: str,
) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": "xrsh_v5",
        "format_version": XRSH_V5,
        "vocab_sha256": vocab_hash.hex(),
        "shard_count": shard_count,
        "source": source,
    }
    (root / "pack_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
