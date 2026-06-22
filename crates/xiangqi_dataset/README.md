# xiangqi_dataset

维护者侧数据工具，只保留第一阶段主线能力：

- 生成 canonical `move_vocab.json`
- 把 PGN 转成 `XRSH v5`
- 对人类局面跑 MCTS，并把搜索标注继续写成 `XRSH v5`

## 主要命令

```bash
cargo run --release -p xiangqi_dataset -- vocab-enum --out data/move_vocab.json

cargo run --release -p xiangqi_dataset -- pgn-shards \
  --pgn data/corpus.pgns \
  --vocab data/move_vocab.json \
  --out-dir data/xrsh_train \
  --jobs 0 \
  --games-per-shard 500

cargo run --release -p xiangqi_dataset -- search-label-pgn \
  --pgn data/corpus.pgns \
  --vocab data/move_vocab.json \
  --out-dir data/xrsh_search \
  --onnx data/policy.onnx \
  --playouts 256
```

## XRSH v5

当前格式只保留主线字段：

- `target_idx`
- `legal_idx`
- `game_result_red`
- `ply_total`
- `search_q`
- `search_visits`
- `search_counts`

不再写入辅助语义头字段。

## 测试

```bash
cargo test -p xiangqi_dataset
```
