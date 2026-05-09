"""读取 Rust ``xiangqi_dataset`` 写入的 XRSH 分片（``shard_*.xrsh``）。

二进制布局须与 ``crates/xiangqi_dataset/src/shard.rs`` 一致。
支持 **文件版本 1**（无辅助伪标签）与 **版本 2**（每样本末尾 3×float32）。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

MAGIC = b"XRSH"
# 魔数 4 + version 4 + vocab 32 + n_games 4 + 保留 20 = 64
HEADER_SIZE = 64


def read_str_u16(buf: bytes, off: int) -> tuple[str, int]:
    ln = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    return buf[off : off + ln].decode("utf-8"), off + ln


def read_prefix_list(buf: bytes, off: int) -> tuple[list[str], int]:
    n = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    out: list[str] = []
    for _ in range(n):
        bl = buf[off]
        off += 1
        out.append(buf[off : off + bl].decode("utf-8"))
        off += bl
    return out, off


def parse_shard_bytes(buf: bytes) -> tuple[list[dict[str, Any]], bytes]:
    """解析单个分片文件全部字节 → ``(flat_samples, vocab_hash_32bytes)``。"""
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"XRSH 分片过短: {len(buf)}")
    if buf[0:4] != MAGIC:
        raise ValueError("魔数非 XRSH（非本仓库 review shard 格式）")
    file_ver = int.from_bytes(buf[4:8], "little")
    if file_ver not in (1, 2):
        raise ValueError(
            f"不支持的 XRSH 文件版本: {file_ver}（支持 1 无辅助标签、2 含 aux）"
        )
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
            _root_fen, off = read_str_u16(buf, off)
            _prefix, off = read_prefix_list(buf, off)
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
            sample: dict[str, Any] = {
                "game_id": gid,
                "fen": fen,
                "root_fen": _root_fen,
                "uci_prefix": _prefix,
                "legal_idx": legal,
                "target_idx": target_idx,
                "ply": ply,
            }
            if file_ver >= 2:
                if off + 12 > len(buf):
                    raise ValueError("XRSH v2 样本缺少 aux 12 字节")
                atk, dan, tac = struct.unpack_from("<fff", buf, off)
                off += 12
                sample["aux_attack"] = atk
                sample["aux_danger"] = dan
                sample["aux_tactical"] = tac
            all_samples.append(sample)
    if off != len(buf):
        raise ValueError(f"XRSH 尾部长度不匹配: 解析到 {off} 总长 {len(buf)}")
    return all_samples, vocab_hash


def read_shard_file(path: Path | str) -> tuple[list[dict[str, Any]], bytes]:
    return parse_shard_bytes(Path(path).read_bytes())


def load_pack_meta(pack_dir: Path | str) -> dict[str, Any]:
    p = Path(pack_dir) / "pack_meta.json"
    return json.loads(p.read_text(encoding="utf-8"))


def xrsh_dir_is_complete(pack_dir: Path | str) -> bool:
    d = Path(pack_dir)
    if not (d / "pack_meta.json").is_file():
        return False
    return any(d.glob("shard_*.xrsh"))
