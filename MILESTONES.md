# 里程碑（P0–P7）

与 **`ARCHITECTURE.md`** 中的产品与分层说明配套（**人类认知驱动的搜索**：网络学人类特征与剪枝先验，**机器侧战术与评估** 主要由搜索承担）；中期目标见根目录 **`工程目标.md`**；**执行勾选**与**近期推荐顺序**见根目录 **`TODO.md`**（文首「推进顺序」表）。

| 阶段 | 目标 | 主要交付 / Crate |
|------|------|------------------|
| **P0** | **完整象棋规则 + 合法 UCI**，与 **pikafish-rust / Pikafish** 语义对齐 | `crates/xiangqi_core`：已从 pikafish-rust 迁入 `types`/`board`/`movegen`/`misc`；`legal_moves_uci`、perft 测试；可选：与 pyffish 抽样对拍 |
| **P1** | **数据管线**：**canonical 词表 + PGN → 二进制 shards**（**XRSH v3** `.xrsh`），**按局并行** | `crates/xiangqi_dataset`：**`vocab-enum`**、**`pgn-shards`**；`pack_meta.json`（`format=xrsh_v3`、`vocab_sha256`）；Python 读取见 **`nn.dataset_xrsh`** |
| **P2** | **Python 训练**接入二进制数据包 + **多头网络**（人类棋谱 policy + 语义辅助头） | `nn/`：`Dataset`/loader、损失与 ONNX 契约扩展（policy + 辅助头）；标签管线与 P1 输出衔接 |
| **P3** | **引擎**：**搜索承担战术深度**；UCI 闭环 | `crates/engin`：**Alpha-Beta**、**TT**、**move ordering**、**UCI 协议**；挂接 `xiangqi_core` + ONNX（policy/语义先验，非单独扛「引擎真理」） |
| **P4** | **人类局面感 Value（可选）**：服务剪枝与志向，**非**引擎静态评估的全量替代 | `nn/`：value head 契约；标签侧重 **人类局面理解**（或文档约定的伪标/Teacher，见 TODO）；`engin`：可选消费接口 |
| **P5** | **Search-aware 语义头**：直接驱动搜索调度 | `nn/`：danger / volatility / forcing / mobility_tension 等；`xiangqi_dataset`：Rust 统一标注 |
| **P6** | **搜索注意力蒸馏（可选）**：学习何处值得算 | 数据/训练：visit count、search distribution 等；**补充**人类 policy，而非取代人类风格主线 |
| **P7** | **Dynamic search**：**人类语义 + 算力** 协同调度 | `crates/engin`：根据多头信号控制 extension / pruning / LMR / top-k |

## 依赖顺序

```
P0 (core) → P1 (dataset CLI) → P2 (train) ─┐
                                          ├→ 完整产品闭环
P0 (core) ───────────────────→ P3 (engin) ┘
```

P3 可与 P2 **部分并行**（规则与 MoveGen 来自 P0），但 **实用引擎**通常需 **policy ONNX**（P2 导出）后再深度联调。

P4–P7 依赖顺序：

```text
P2 + P3 → P4
P4 → P5
P3 + P5 → P6
P4 + P5 + P6 → P7
```

## 参考实现（本地）

- `c:\projects\pikafish-rust` — 规则、位棋盘、`movegen` 分层优先对齐此处。
- `c:\projects\Pikafish` — 行为与边界用例的最终参照。
