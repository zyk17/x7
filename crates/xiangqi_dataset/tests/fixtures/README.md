# 测试夹具说明

- **`jsonl_smoke.rs`** 冒烟测试在运行时生成：
  - 临时目录中的 **move_vocab.json**（内容为起始局面全部合法 UCI，已排序）；
  - 单行 **JSONL**（与训练管线 JSONL 字段一致，见根目录 **`ARCHITECTURE.md`**）。
- 无需提交大块棋谱；若要手工跑 CLI，可先在同一目录写好 `vocab.json` 与 `*.jsonl`，再：

```bash
cargo run -p xiangqi_dataset -- jsonl-shards --jsonl train.jsonl --vocab move_vocab.json --out-dir out_xrsh
```
