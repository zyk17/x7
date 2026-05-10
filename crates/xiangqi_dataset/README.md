# xiangqi_dataset



维护者工具：**canonical 词表 + PGN → 二进制分片**（格式 **XRSH v3**（`pack_meta.format: xrsh_v3`），扩展名 **`.xrsh`**）。**PGN** 按局并行（Rayon，`--jobs`，`0` = 本机并行度）。编码时为每样本写入 **`xiangqi_core`** 预计算的 **attack/danger/tactical** 三 float，以及 PGN **`[Result]`** 与总局数 **`ply_total`**（与 `nn` value 监督对齐）。



## 依赖



- 词表：`move_vocab.json`（`{ "moves": [...] }`）。主线使用本 crate 子命令 **`vocab-enum`** 直接生成 **canonical** 词表；不再把“从 PGN 扫词表”作为常规流程。
- 若需 **XRSH v3 的 value 字段**：棋谱须有 **`[Result "1-0"]` / `[Result "0-1"]` / `[Result "1/2-1/2"]`**（标签内可多写空格，解析时会压缩）；`*` 或未写 `Result` 则结局未知。

- 规则与合法着：`xiangqi_core`（UCI 与 Pikafish 一致：**纵坐标 0～9**）。



## 命令



```bash

# 直接枚举 canonical 词表 → move_vocab.json

cargo run --release -p xiangqi_dataset -- vocab-enum --out data/move_vocab.json



# 从 PGN / .pgns 生成 shards（--jobs 0 = 本机可用并行度，按局并行）

cargo run --release -p xiangqi_dataset -- pgn-shards --pgn data/corpus.pgns --vocab data/move_vocab.json --out-dir data/xrsh_out --jobs 8 --games-per-shard 500

```



输出：`shard_00000.xrsh`、…、`pack_meta.json`（`format: xrsh_v3`，含 `vocab_sha256`，与 Python `policy_pack` 指纹算法一致）。



## 格式说明



见仓库根目录 **`ARCHITECTURE.md`**（数据管线 / **XRSH v3**）。Python 训练侧使用 **`nn.dataset_xrsh.PolicyXrshDataset`**（`train_policy.py --train-xrsh-dir`）。



## 测试



```bash

cargo test -p xiangqi_dataset

```



含：**单元测试**（ICCS、PGN、`vocab` 哈希）、**集成冒烟**（`tests/pgn_xrsh_smoke.rs`）、**PGN 并行一致性**（`tests/pgn_shards_parallel.rs`：`jobs=1` 与多线程输出分片字节一致）。夹具说明见 **`tests/fixtures/README.md`**。
