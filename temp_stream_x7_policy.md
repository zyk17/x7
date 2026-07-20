# X7 stream policy（临时，未合并）

独立记录。**不要**与 `NextStep.md` / `TODO.md` / `temp.md` 混写；合并文档另开。

## 定位

- Stream 仍是 **classic 语义的 tree MCTS**：选边、出招先对照本仓库 classic / px0。
- 变的是 **表示与并发**（分片 repository + owned event + Gather/Eval/Backprop），便于 Rust 安全实现。
- 首版 **tree key**，不做 DAG/TT。
- **UCI 本阶段不切 stream**（S3 deferred）。对拍用 `stream_compare`。

标注：**X7 stream policy**（项目批准的自创对照 classic；不伪称 LC3 源码翻译）。

## 已确认

| 项 | 选择 |
|----|------|
| Final move | 根上 completed `N` → `Q` → `P`；terminal 已证明胜压败；同档比 `m`；未访问子排序用 `Q=0` |
| Selection | 对齐 classic FPU + `ComputeCpuct`；edge `N` 含 in-flight；publish 后按 P 排序 |
| Q / 正负号 | 对齐 px0：NN `wl = -eval.wl` 入库；edge 存 child/mover 视角；parent backup 再翻号；`draw_score=0` |
| Draw | 无 contempt/`draw_score`，但 WDL 的 `d` 仍存于 node 并在 `root_stats` 暴露 |
| searchmoves | 与 px0 相同：UCI `go searchmoves` → 内部 `root_move_filter`；选边与 bestmove 共用（空=不限制） |
| 将死/无边根 | `best_move = Move::NULL`（`a0a0`）；0 visit 时回退 filter/首边 |
| Two-fold | 对齐 classic 早停（`two_fold_draws`） |
| S3 UCI | 不做；库内 + `stream_compare` 出 bestmove/PV |
| MultiPV / TB / multivisit 批量 / 时间剪枝 | 暂不做（并发模型可不同，但自身须 settled/正确） |

## 参考锚点

- px0 正负号：`search.cc:2129`（NN 取反）、`2175-2257`（backup `v=-v`）、`node.h` mover 视角 `wl_`
- px0 searchmoves：`search.cc:53-58,721-724`；`wrapper.cc` 解析 go params
- classic 选边：`crates/engin/src/search/classic/uct.rs`
- classic 出招：`search.rs` `best_child_edge`（px0 `search.cc:705-808`）
- LC3 架构（非公式）：https://lczero.org/dev/lc0/search/lc3/overview/

## 对拍工具

```powershell
cargo run -p engin --bin stream_behavior_compare
cargo run -p engin --bin stream_compare -- --uniform --playouts 128
```

## 明确不做

- 切换 UCI 到 stream  
- 从 classic 搬共享 mutable tree / task-worker 模型  
- 宣称正式 LC3 selection/output  
- 为对齐 classic 而改 LC3 式 edge/node 字段布局  
- stream 路径使用 contempt / 非零 `draw_score`  
