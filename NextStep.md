# NextStep

## 当前阶段：重建 lc0 classic 搜索核心

旧 `crates/engin/src/mcts/` 已整体删除。当前不接受“修旧搜索”或以旧接口为约束的实现。

唯一主参考：`C:\Users\Administrator\projects\lc0`。

## 逐函数移植顺序

| 阶段 | lc0 参考 | 本仓库目标 | 完成条件 |
|---|---|---|---|
| S0 | `src/search/classic/search.h:50-203` | 新建最小 `mcts/` 数据结构：`Node`、`Edge`、`Search`、`SearchWorker` | 无 UCI 接线；节点字段可逐项对照 |
| S1 | `search.cc:921-1047`、`search.h:254-300` | 单线程 `RunBlocking` 及 iteration 7 阶段调度 | 无 ONNX 时能按 budget 结束；root 不预展开 |
| S2 | `search.cc:1507-1920`、`search.h:407-416` | `PickNodesToExtend` / `PickNodesToExtendTask` | root 与子节点同一路径；in-flight 可回滚 |
| S3 | `search.cc:1921-2149` | `ExtendNode` | 合法着、象棋终局、rule60、重复规则只在此处接入 |
| S4 | `search.cc:2151-2216` | `FetchMinibatchResults` | ONNX 在树锁外；无 ONNX fallback 明确可测 |
| S5 | `search.cc:2217-2373` | `DoBackupUpdateSingleNode` / PV / 计数 | nodes、playouts、visits、in_flight 逐项不变量测试 |
| S6 | `search.cc:261-396,617-646,896-920` | UCI info、stop、线程生命周期 | `go nodes` / `go movetime` 正确；再恢复 UCI `go` |
| S7 | `search.cc:1060-1490`、`search.h:209-419` | batch、task workers、OOO eval 与 shared tree | 只在 S0-S6 trace 对齐后开始 |

## 中国象棋最小偏离点

| 内容 | 参考 | 本仓库 |
|---|---|---|
| 合法着 / do-undo | lc0 `chess/board.*` 的接口位置 | `crates/xiangqi_core/src/board.rs`、`movegen.rs` |
| history / repetition | lc0 `chess/position_history.*` | `crates/engin/src/history.rs` |
| 重复裁决、长将长捉、rule60 | 需先核对 px0 象棋规则后接入 | `crates/xiangqi_core/src/rule.rs` |
| 网络输入 | lc0 NN 输入接口，仅替换为 px0 124 planes | `crates/engin/src/fen_tensor.rs` |

没有找到 lc0 对照位置时，不实现，先记录到 `TODO.md`。
