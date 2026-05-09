# xiangqi_dataset

维护者工具：**PGN / JSONL → 二进制分片**（格式 **XRSH v2**（`pack_meta.format: xrsh_v2`），扩展名 **`.xrsh`**）。**PGN** 按局并行；**JSONL** 按行并行（Rayon，`--jobs`，`0` = 本机并行度）。编码时为每样本写入 **`xiangqi_core`** 预计算的 **attack/danger/tactical** 三 float（与 `nn.aux_pseudo_labels` 对齐）。

## 依赖

- 词表：`nn/scripts/vocab/build_vocab.py` 生成的 `move_vocab.json`（`{ "moves": [...] }`）。
- 规则与合法着：`xiangqi_core`（与 pyffish 坐标 **1～10** 纵标一致）。

## 命令

```bash
# 从 PGN / .pgns 生成 shards（--jobs 0 = 本机可用并行度，按局并行）
cargo run -p xiangqi_dataset -- pgn-shards --pgn data/foo.pgns --vocab data/move_vocab.json --out-dir data/xrsh_out --jobs 8 --games-per-shard 500

# 从已有训练 JSONL 生成（每行需含 game_id / fen / root_fen / uci_prefix / human_move_pyffish / ply）
# 超大 JSONL：务必加 --jobs（0 即用满 CPU）；按行并行解析与编码（Rayon）
cargo run -p xiangqi_dataset -- jsonl-shards --jsonl data/train.jsonl --vocab data/move_vocab.json --out-dir data/xrsh_from_jsonl --jobs 0
```

说明：**整文件会先读入内存**后再并行处理单行；若单行数极大导致内存吃紧，可先在外部按字节切成多个 JSONL 分批运行。

输出：`shard_00000.xrsh`、…、`pack_meta.json`（`format: xrsh_v2`，含 `vocab_sha256`，与 Python `policy_pack` 指纹算法一致）。

## 格式说明

见仓库根目录 **`ARCHITECTURE.md`**（数据管线 / **XRSH v1↔v2**）。Python 训练侧使用 **`nn.dataset_xrsh.PolicyXrshDataset`**（`train_policy.py --train-xrsh-dir`）。

## 测试

```bash
cargo test -p xiangqi_dataset
```

含：**单元测试**（ICCS、PGN、`vocab` 哈希）、**集成冒烟**（`tests/jsonl_smoke.rs`）、**JSONL 并行一致性**（`tests/jsonl_parallel.rs`：`jobs=1` 与多线程输出分片字节一致）。夹具说明见 **`tests/fixtures/README.md`**。
