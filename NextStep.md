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
- `PickNodesToExtendTask` 现在显式写入 caller receiver，对应
  `src/search/classic/search.h:401-406`；下一步可直接将 gathering task 的结果写入
  `PickTask.results`。
- `PickNodesToExtend` 在主选择完成后等待并汇合各 `PickTask.results`，对应
  `src/search/classic/search.cc:1501-1507`；task split/dispatch 尚未接线。
- `PickNodesToExtendTask` 的 DFS state 改为显式 `TaskWorkspace` 参数，对应
  `src/search/classic/search.h:401-406,425-434`、`search.cc:1551-1827`；主 worker
  仍持有自己的 workspace，后续 gathering task 可各自持有独立 workspace。
- `RunTasks` 的领取、按 gathering/processing 分派和完成回写已落地，对应
  `src/search/classic/search.cc:1069-1140`；目前由主 worker 同步消费队列，task split 与
  常驻 task worker 尚未接线。
- gathering split 已按 px0 `MinimumPickingWork=1`、`MinimumRemainingPickingWork=20`、
  `MAX_TASKS=100` 及 passed-off/completed-visits 条件翻译，对应
  `src/search/classic/params.cc:604-612`、`search.cc:1828-1864`；常驻 task worker
  和并发树访问边界尚未接线。
- processing split 已按 px0 `MinimumProcessingWork=20`、`MinimumPerTaskProcessing=8`
  将前段交给 `PickTask::Processing`、主 worker 保留尾段，对应
  `src/search/classic/params.cc:604-612`、`search.cc:1322-1347`；当前同步执行，
  常驻 task worker 生命周期尚未接线。
- `TaskWorkers=-1` 已按 px0 GPU 硬件并发启发式解析（每个 search worker 最多 4 个；CPU 为 0），
  对应 `src/search/classic/search.h:205-233`；当前仍同步消费任务队列，不能在整树锁下伪造常驻
  task thread。
- collision 的 `maxvisit` 扩容、祖先 `NInFlight` 更新与 collision-budget 停止条件已翻译，
  对应 `src/search/classic/search.cc:1400-1419`；不再以“本轮没有叶子”提前返回。
- `InitializeIteration` 现在在创建新 computation 前释放上一轮 computation，对应
  `src/search/classic/search.cc:1233-1240`，避免后端缓存和分配生命周期漂移。
- `ResetTasks` 每轮清空后保留 `MAX_TASKS=100` 容量，对应
  `src/search/classic/search.cc:1464-1473`；这是后续 task worker 持有任务稳定地址的前提。
- `ClassicSearch` 现在每次搜索线程只构建一个 `SearchWorker`，并通过
  `SearchWorker::RunBlocking` 持久执行 iteration，对应
  `src/search/classic/search.h:235-249`；为 task workspace/NN computation 的跨 iteration
  所有权建立了 px0 一致的生命周期。
- `time_since_first_batch` 改由第一个完成 backup 的 worker 写入共享状态，watchdog 读取该状态，
  对齐 px0 worker/watchdog 分离的统计时序（`src/search/classic/search.cc:2158-2173,2331-2364`）。
- `PickTaskQueue` 已支持 px0 的阻塞领取、condition-variable 唤醒和 `task_count=-1` 退出语义，
  对应 `src/search/classic/search.cc:1069-1124`、`search.h:225-233`；尚未把实际 task thread
  接到树的子树并发访问。
- `SearchWorker` 在构造时按 task worker 数分配独立 `TaskWorkspace`，同步 task dispatch
  也使用该 workspace，并在 worker 退出时关闭队列，对应
  `src/search/classic/search.h:205-233,357-364`；实际常驻 task thread 尚未接线。
- `SearchWorker::DoBackupUpdateSingleNode` 已补齐 sticky-endgame 的 bounds 传播、终局
  平均值修正与强制终局父节点标记，对应
  `src/search/classic/search.cc:2175-2289`、`src/search/classic/node.cc:300-392`；
  root best-edge 缓存与 `MakeSolid` 内存布局仍待按 Rust arena 访问边界单独翻译。
- `TaskWorkspace` 已恢复 px0 的 256-slot selection scratch 数组及选中边 score 的增量
  更新，对应 `src/search/classic/search.h:348-365`、`search.cc:1575-1825`；
  `Node::CreateEdges` 同步保留 px0 `uint8_t num_edges_` 的 255 条上限。

### P4 下一入口

- `search.cc:1828-1897`、`search.h:367-448`：task worker split 与任务队列
- `classic/node.h:127-339`、`search.cc:1494-1508`：稳定 node 存储与 task
  selection 的树访问边界；当前 `Vec<Node>` + 整轮 `Mutex` 不能直接承载 px0 子树并发
- `search.cc:2103-2364`：root best-edge 缓存、`MakeSolid` 与释放树锁后的
  NN compute/fetch/backup 分阶段并发
- `engine.cc:153-167`、`neural/shared_params.*`：`WeightsFile` 到真实 ONNX backend 的 UCI 配置
- px0 二进制 fixed-nodes trace 对拍
