# 测试夹具说明

- **`pgn_xrsh_smoke.rs`**：运行时构造极小 **PGN**（两回合合法 UCI），再 **`collect_vocab_moves_from_pgn`** → **`run_pgn_shards`**，校验分片头与 `pack_meta`。
- 无需提交大块棋谱。手工跑 CLI 时在**仓库根**执行，例如：

```bash
cargo run --release -p xiangqi_dataset -- vocab-enum --out data/move_vocab.json
cargo run --release -p xiangqi_dataset -- pgn-shards --pgn data/foo.pgn --vocab data/move_vocab.json --out-dir data/xrsh_out --jobs 0
```
