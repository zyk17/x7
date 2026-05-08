# 里程碑（P0–P3）

与 **`ARCHITECTURE.md`** 中的产品与分层说明配套；**执行勾选**见根目录 **`TODO.md`**。

| 阶段 | 目标 | 主要交付 / Crate |
|------|------|------------------|
| **P0** | **完整象棋规则 + 合法 UCI**，与 **pikafish-rust / Pikafish** 语义对齐 | `crates/xiangqi_core`：已从 pikafish-rust 迁入 `types`/`board`/`movegen`/`misc`；`legal_moves_uci`、perft 测试；可选：与 pyffish 抽样对拍 |
| **P1** | **数据管线**：PGN / JSONL → **二进制 shards**（**XRSH** `.xrsh`），**按局并行** | `crates/xiangqi_dataset`：**`xrsh_v1`**、`pack_meta.json`（`vocab_sha256`）；CLI 见 crate `README.md`；Python 读取见 **`nn.dataset_xrsh`** |
| **P2** | **Python 训练**接入二进制数据包 + **多头网络** | `nn/`：`Dataset`/loader、损失与 ONNX 契约扩展（policy + 辅助头）；标签管线与 P1 输出衔接 |
| **P3** | **引擎**：搜索 + UCI | `crates/engin`：**Alpha-Beta**、**TT**、**move ordering**、**UCI 协议**；挂接 `xiangqi_core` + ONNX |

## 依赖顺序

```
P0 (core) → P1 (dataset CLI) → P2 (train) ─┐
                                          ├→ 完整产品闭环
P0 (core) ───────────────────→ P3 (engin) ┘
```

P3 可与 P2 **部分并行**（规则与 MoveGen 来自 P0），但 **实用引擎**通常需 **policy ONNX**（P2 导出）后再深度联调。

## 参考实现（本地）

- `c:\projects\pikafish-rust` — 规则、位棋盘、`movegen` 分层优先对齐此处。
- `c:\projects\Pikafish` — 行为与边界用例的最终参照。
