# NextStep

## 当前阶段：P4 单 worker 搜索与 UCI 生命周期已接线；完整碰撞/task workers/weights 配置未闭合

P0–P3 规则、UCI、搜索树均已通过。P4 **worker 七阶段 + 异步 `ClassicSearch`** 已接入 UCI：

- 测试用 `UniformBackend` 下，`go nodes` / `go movetime` / `go wtime` / `go infinite`+`stop` 可返回 `bestmove`
- 主 UCI 不再使用 Uniform fallback；尚未翻译 `WeightsFile` 配置时会明确返回不可搜索状态
- UniformBackend NN cache 子集；固定 nodes 确定性 trace（Rust stub，非 px0 二进制）

当前唯一工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

| 阶段 | px0 参考 | Rust 目标 | 完成条件 |
|---|---|---|---|
| P0 | `src/chess/types.h`、`bitboard.h`、`board.h/.cc` | types、bitboard、ChessBoard、FEN、走子、合法着 | `board_test.cc` 与 legal move-set 对拍 |
| P1 | `src/chess/position.h/.cc` | Position、PositionHistory、重复、rule60、RuleJudge | `position_test.cc` 逐项移植 |
| P2 | `src/chess/gamestate.*`、`uciloop.*` | UCI 局面/history 语义 | `position ... moves ...` 与 px0 一致 |
| P3 | `src/search` | Node、Tree、Search、worker、选择/扩展/回传 | 单线程固定 FEN/budget trace 对照 |
| P4 | `src/search`、`src/neural` | minibatch、NN cache、prefetch、并发、tree reuse | 同局面统计和 bestmove 对照 |
| P6 | `pxzero-training` | 数据字段、训练/导出契约 | 训练与 ONNX I/O 对照 |

每个 Rust 函数必须标注 px0 文件和连续行区间；找不到参考不实现。

### P4 已落地

- `engin/src/search/classic/backend.rs`：`UniformBackendComputation` + cache（`backend.h:67-78`）
- `engin/src/search/classic/worker.rs`：七阶段 + `nodes_budget` + OOO 子集
- `engin/src/search/classic/node.rs`：`EdgeAndNode` Q/U/NStarted 代理；worker 的
  单 worker `PickNodesToExtendTask` 显式 workspace/path-backtrack 翻译
  （`search.cc:1551-1827`，task split 尚未翻译）
- `engin/src/search/classic/worker.rs`：递归 `PrefetchIntoCache`（`search.cc:1989-2099`）
- `engin/src/search/classic/search.rs`：异步 `StartThreads`、`ClassicSearch` 取代 `SearchSession` 主路径
- `engin/src/search/classic/stoppers/*`：Visits/Playouts/TimeLimit/wtime 预算
- `engin/src/search/classic/uct.rs`：PUCT/FPU 共享辅助
- `engin/tests/p4_*` + `uci_search_test` + `search_trace_test` 全绿

### P4 已完成的真实网络入口

- `engin/src/neural/mod.rs`：真实 `PositionHistory -> 124x10x9`，对应
  `src/neural/encoder.cc:118-217`；`encoder.cc:229-481` 的 2062 policy 表由
  源码机械提取为 `px0_policy_moves.txt`。
- `engin/src/neural/onnx.rs`：`OnnxBackend` / batch computation，逐项翻译
  `src/neural/wrapper.cc:49-172`。本地 `data/x7.onnx` 冒烟已通过。
- `engin/src/neural/backend.rs`、`onnx.rs`：`BackendComputation` 以 task-safe
  内部状态承载并发 `AddInput`，对应 `src/neural/backend.h:75-87` 与
  `src/search/classic/search.cc:1423-1462`；NN compute 期间不持有 batch 状态锁。
- `engin/src/search/classic/worker.rs`：`PickTask`、`PickTaskQueue` 与 worker
  生命周期的 `ResetTasks` 已翻译，对应 `src/search/classic/search.h:367-445`、
  `search.cc:1069-1140,1464-1508`；task dispatch/split 尚未接线。

### P4 下一入口

- `search.cc:1828-1897`、`search.h:367-448`：task worker split 与任务队列
- `classic/node.h:127-339`、`search.cc:1494-1508`：稳定 node 存储与 task
  selection 的树访问边界；当前 `Vec<Node>` + 整轮 `Mutex` 不能直接承载 px0 子树并发
- `search.cc:2103-2364`：释放树锁后的 NN compute/fetch/backup 分阶段并发
- `engine.cc:153-167`、`neural/shared_params.*`：`WeightsFile` 到真实 ONNX backend 的 UCI 配置
- px0 二进制 fixed-nodes trace 对拍
