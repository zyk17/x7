"""读取 Rust ``xiangqi_dataset`` 写入的 XRSH v3 分片（``shard_*.xrsh``）。

二进制布局须与 ``crates/xiangqi_dataset/src/shard.rs`` 一致。
当前训练主线仅支持 **v3**：``aux_attack / aux_danger / aux_tactical`` +
``game_result_red`` + ``ply_total``。
"""

from __future__ import annotations

import hashlib
import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIC = b"XRSH"
# 魔数 4 + version 4 + vocab 32 + n_games 4 + 保留 20 = 64
HEADER_SIZE = 64


@dataclass(frozen=True)
class XrshRowRef:
    shard_index: int
    row_offset: int
    fen_key: int
    game_group: int
    ply: int
    game_result_red: int
    ply_total: int


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
        hashlib.blake2b(fen.encode("utf-8"), digest_size=8).digest(), "little"
    )


def parse_shard_bytes(buf: bytes) -> tuple[list[dict[str, Any]], bytes]:
    """解析单个分片文件全部字节 → ``(flat_samples, vocab_hash_32bytes)``。"""
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"XRSH 分片过短: {len(buf)}")
    if buf[0:4] != MAGIC:
        raise ValueError("魔数非 XRSH（非本仓库 review shard 格式）")
    file_ver = int.from_bytes(buf[4:8], "little")
    if file_ver != 3:
        raise ValueError(f"仅支持 XRSH v3，当前文件版本: {file_ver}")
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
            if off + 15 > len(buf):
                raise ValueError("XRSH v3 样本缺少 aux/result 共 15 字节")
            atk, dan, tac = struct.unpack_from("<fff", buf, off)
            off += 12
            gr, pt = struct.unpack_from("<bH", buf, off)
            off += 3
            sample["aux_attack"] = atk
            sample["aux_danger"] = dan
            sample["aux_tactical"] = tac
            sample["game_result_red"] = int(gr)
            sample["ply_total"] = int(pt)
            all_samples.append(sample)
    if off != len(buf):
        raise ValueError(f"XRSH 尾部长度不匹配: 解析到 {off} 总长 {len(buf)}")
    return all_samples, vocab_hash


def read_shard_file(path: Path | str) -> tuple[list[dict[str, Any]], bytes]:
    return parse_shard_bytes(Path(path).read_bytes())


def scan_shard_file(
    path: Path | str, *, shard_index: int, start_game_group: int = 0
) -> tuple[list[XrshRowRef], bytes, int]:
    """仅扫描分片元数据，返回行引用，不展开为 Python dict。"""
    buf = Path(path).read_bytes()
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"XRSH 分片过短: {len(buf)}")
    if buf[0:4] != MAGIC:
        raise ValueError("魔数非 XRSH（非本仓库 review shard 格式）")
    file_ver = int.from_bytes(buf[4:8], "little")
    if file_ver != 3:
        raise ValueError(f"仅支持 XRSH v3，当前文件版本: {file_ver}")
    vocab_hash = bytes(buf[8:40])
    n_games = int.from_bytes(buf[40:44], "little")
    off = HEADER_SIZE
    refs: list[XrshRowRef] = []
    game_group = start_game_group
    for _ in range(n_games):
        _gid, off = read_str_u16(buf, off)
        n_rows = int.from_bytes(buf[off : off + 4], "little")
        off += 4
        for _ in range(n_rows):
            row_offset = off
            fen, off = read_str_u16(buf, off)
            off = _skip_str_u16(buf, off)  # root_fen
            off = _skip_prefix_list(buf, off)
            off += 4  # target_idx
            n_leg = int.from_bytes(buf[off : off + 2], "little")
            off += 2 + 4 * n_leg
            ply = int.from_bytes(buf[off : off + 2], "little")
            off += 2
            if off + 15 > len(buf):
                raise ValueError("XRSH v3 样本缺少 aux/result 共 15 字节")
            off += 12  # aux triples
            gr, pt = struct.unpack_from("<bH", buf, off)
            off += 3
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


def read_row_at(
    buf: bytes | mmap.mmap, row_offset: int
) -> dict[str, Any]:
    """从分片缓冲区按 offset 解析单条样本。"""
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
    atk, dan, tac = struct.unpack_from("<fff", buf, off)
    off += 12
    gr, pt = struct.unpack_from("<bH", buf, off)
    return {
        "fen": fen,
        "root_fen": root_fen,
        "uci_prefix": prefix,
        "legal_idx": legal,
        "target_idx": target_idx,
        "ply": int(ply),
        "aux_attack": atk,
        "aux_danger": dan,
        "aux_tactical": tac,
        "game_result_red": int(gr),
        "ply_total": int(pt),
    }


def read_row_train_at(
    buf: bytes | mmap.mmap, row_offset: int
) -> tuple[str, list[int], int, float, float, float]:
    """训练热路径最小解析：只取 ``fen / legal_idx / target_idx / aux_*``。"""
    off = row_offset
    fen, off = read_str_u16(buf, off)
    off = _skip_str_u16(buf, off)  # root_fen
    off = _skip_prefix_list(buf, off)
    target_idx = int.from_bytes(buf[off : off + 4], "little", signed=True)
    off += 4
    n_leg = int.from_bytes(buf[off : off + 2], "little")
    off += 2
    legal: list[int] = []
    for _ in range(n_leg):
        j = int.from_bytes(buf[off : off + 4], "little", signed=True)
        off += 4
        legal.append(j)
    off += 2  # ply
    atk, dan, tac = struct.unpack_from("<fff", buf, off)
    return fen, legal, target_idx, atk, dan, tac


def load_pack_meta(pack_dir: Path | str) -> dict[str, Any]:
    p = Path(pack_dir) / "pack_meta.json"
    return json.loads(p.read_text(encoding="utf-8"))


def xrsh_dir_is_complete(pack_dir: Path | str) -> bool:
    d = Path(pack_dir)
    if not (d / "pack_meta.json").is_file():
        return False
    return any(d.glob("shard_*.xrsh"))
