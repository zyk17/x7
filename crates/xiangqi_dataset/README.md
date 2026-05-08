# xiangqi_dataset

维护者工具：**PGN / JSONL → 二进制分片**（格式 **`XQB` v1**），PGN 路径下按对局 **Rayon 并行**。

## 依赖

- 词表：`nn/scripts/data_pgn/build_vocab.py` 生成的 `move_vocab.json`（`{ "moves": [...] }`）。
- 规则与合法着：`xiangqi_core`（与 pyffish 坐标 **1～10** 纵标一致）。

## 命令

```bash
# 从 PGN / .pgns 生成 shards（--jobs 0 为默认线程数）
cargo run -p xiangqi_dataset -- pgn-shards --pgn data/foo.pgns --vocab data/move_vocab.json --out-dir data/xqb --jobs 8 --games-per-shard 500

# 从已有 extract_rows JSONL 生成（每行需含 game_id / fen / root_fen / uci_prefix / human_move_pyffish / ply）
cargo run -p xiangqi_dataset -- jsonl-shards --jsonl data/train.jsonl --vocab data/move_vocab.json --out-dir data/xqb_j
```

输出：`shard_00000.xqb`、…、`pack_meta.json`（含 `vocab_sha256` 与 Python `policy_pack` 指纹算法一致）。

## 格式说明

见仓库根目录 **`ARCHITECTURE.md`**（数据管线 / `xqb_v1`）。P2 将由 Python 或本仓库增加 **读取 XQB → 训练** 的加载器。
