# NextStep

## 当前阶段：P4 单 worker 搜索、collision、prefetch 与常驻 task workers 已接线；多 SearchWorker tree boundary 未闭合

P0–P3 规则、UCI、搜索树均已通过。P4 **worker 七阶段 + 异步 `ClassicSearch`** 已接入 UCI：

- 测试用 `UniformBackend` 下，`go nodes` / `go movetime` / `go wtime` / `go infinite`+`stop` 可返回 `bestmove`
- 主 UCI 不再使用 Uniform fallback；`WeightsFile` 可在下一条 `position` 前加载正式 ONNX backend
- UniformBackend NN cache 子集；fixed-nodes stopper trace（Rust stub，非 px0 二进制）

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
- `engin/src/search/classic/worker.rs`：七阶段 + px0 `VisitsStopper` + OOO 子集
- `engin/src/search/classic/node.rs`：`EdgeAndNode` Q/U/NStarted 代理；worker 的
  单 worker `PickNodesToExtendTask` 显式 workspace/path-backtrack 翻译
  （`search.cc:1551-1827`，task split 已接线）
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
- `engin/src/search/classic/worker.rs`：`PickTask`、`PickTaskQueue`、`RunTasks` 与 worker
  生命周期的 `ResetTasks` 已翻译，对应 `src/search/classic/search.h:205-249,367-445`、
  `search.cc:1069-1140,1322-1347,1464-1508,1828-1864`；主 worker 在对应阶段等待
  常驻 task workers 完成。
- `PickNodesToExtendTask` 现在显式写入 caller receiver，对应
  `src/search/classic/search.h:401-406`；下一步可直接将 gathering task 的结果写入
  `PickTask.results`。
- `PickNodesToExtend` 在主选择完成后等待并汇合各 `PickTask.results`，对应
  `src/search/classic/search.cc:1494-1508`；task split/dispatch 已接线。
- `PickNodesToExtendTask` 的 DFS state 改为显式 `TaskWorkspace` 参数，对应
  `src/search/classic/search.h:401-406,425-434`、`search.cc:1551-1827`；主 worker
  仍持有自己的 workspace，后续 gathering task 可各自持有独立 workspace。
- `RunTasks` 的领取、按 gathering/processing 分派和完成回写，以及 `RunBlocking`
  内的常驻 task-worker 生命周期已落地，对应 `src/search/classic/search.h:205-249`、
  `search.cc:1069-1140`；直接单元测试入口仍保留同步 bridge。
- gathering split 已按 px0 `MinimumPickingWork=1`、`MinimumRemainingPickingWork=20`、
  `MAX_TASKS=100` 及 passed-off/completed-visits 条件翻译，对应
  `src/search/classic/params.cc:604-612`、`search.cc:1828-1864`；常驻 task worker
  已接线，多 SearchWorker 的并发树访问边界尚未接线。
- processing split 已按 px0 `MinimumProcessingWork=20`、`MinimumPerTaskProcessing=8`
  将前段交给 `PickTask::Processing`、主 worker 保留尾段，对应
  `src/search/classic/params.cc:604-612`、`search.cc:1322-1347`；正常搜索由
  常驻 task workers 执行，直接单元测试使用同步 bridge。
- `TaskWorkers=-1` 已按 px0 GPU 硬件并发启发式解析（每个 search worker 最多 4 个；CPU 为 0），
  对应 `src/search/classic/search.h:205-233`；常驻 task worker 会在 `RunBlocking` 生命周期内
  启动、阻塞领取任务，并在退出前关闭和 join。
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
  对应 `src/search/classic/search.cc:1069-1124`、`search.h:225-233`；实际 task thread 已接到
  单个 SearchWorker 的拆分子树/processing range。
- `SearchWorker` 在构造时按 task worker 数分配独立 `TaskWorkspace`，正常搜索由每个
  常驻 task thread 独占一个 workspace，并在 worker 退出时关闭队列、join 全部 task threads；
  对应 `src/search/classic/search.h:205-249,357-364`。
- `WorkerSearchState::pending_searchers` 与 `SearchWorker::ExecuteOneIteration` 现翻译
  px0 `MaxConcurrentSearchers`：slot 覆盖 gather/collision/prefetch，NN compute 前归还；
  `SearchSpinBackoff=false` 保持 hard-spin 默认值，对应
  `src/search/classic/params.cc:399-404,604-604,525-526,632-632`、
  `src/search/classic/search.cc:1142-1195`。当前单 SearchWorker 下行为不变，作为后续多 worker
  tree boundary 的先决共享状态。
- `WorkerSearchState::backend_waiting_counter`、`IdlingMinimumWork=0` 与
  `ThreadIdlingThreshold=1` 已翻译：存在其他 search worker 时，当前 worker 会在 backend
  idle 前提前结束 gather；计数覆盖 collision/prefetch/NN compute，对应
  `src/search/classic/params.cc:498-505,628-629`、
  `src/search/classic/search.cc:1187-1199,1290-1301`。当前单 SearchWorker 下无行为变化。
- `SearchWorker` 已分出直接测试 tree 与生产 shared tree 两种持有方式；生产路径仅在
  initialize、gather/collision/prefetch、fetch/backup 三个 px0 tree phase 持有 `NodeTree` mutex，
  NN compute 不持有树锁。`ClassicSearch::StartThreads` 已允许多个 SearchWorker，固定 nodes 的
  两 worker UniformBackend 回归通过，对应 `src/search/classic/search.cc:1088-1140,1142-1211,
  1979-2008,2161-2174`。两 SearchWorker + 每 worker 一个 task worker 的 fixed-nodes 压力回归
  也通过，并验证 root `NInFlight` 在结束时归零。
- `SearchWorker::DoBackupUpdateSingleNode` 已补齐 sticky-endgame 的 bounds 传播、终局
  平均值修正与强制终局父节点标记，对应
  `src/search/classic/search.cc:2175-2289`、`src/search/classic/node.cc:300-392`；
  root best-edge 缓存与 `MakeSolid` 内存布局仍待按 Rust arena 访问边界单独翻译。
- `TaskWorkspace` 已恢复 px0 的 256-slot selection scratch 数组及选中边 score 的增量
  更新，对应 `src/search/classic/search.h:348-365`、`search.cc:1575-1825`；
  `Node::CreateEdges` 同步保留 px0 `uint8_t num_edges_` 的 255 条上限。
- `PickNodesToExtendTask` 已恢复 px0 receiver 在容量不足 30 时的按需预留，主 minibatch
  与 gathering task result 可跨 iteration 复用容量（`src/search/classic/search.cc:1570-1573`）。
- `ProcessPickedTask` 现在在每个非 terminal leaf 扩展后立即 `AddInput`，再执行
  out-of-order fetch（`src/search/classic/search.cc:1423-1462`）；不再通过临时输入列表改变
  cache-hit 的回传时序。
- `UpdateCounters` 现在直接调用共享 `VisitsStopper`，并删除 Rust 私有的
  `nodes_budget` 硬截断（`src/search/classic/search.cc:596-620,2331-2334`，
  `src/search/classic/stoppers/stoppers.cc:59-70`）。因此 `go nodes N` 是 px0 的
  completed-iteration 下限，不承诺 root visits 严格等于 `N`。
- `StoppersHints` 现按 px0 每次 stopper pass 前 reset、大上限初始化、min-only
  更新，并将最新 remaining playouts 回写给下一轮 gather（`src/search/classic/
  search.cc:596-610`、`src/search/classic/stoppers/timemgr.cc:35-66`）。
- root `current_best_edge`、无温度 best-child 排序（terminal/tablebase、visits、Q、prior）
  和 remaining-playouts root smart pruning 已翻译（`src/search/classic/search.cc:705-808,
  1584-1588,1726-1742,2241-2249`）。`MakeSolid` 仍明确阻塞于稳定并发 node 存储：
  px0 `node.cc:245-289` 会转换 sibling 链及 pointer 所有权，当前 Rust arena 不能伪造。
- `WeightsFile -> OnnxBackend` 的 UCI/engine 子集已翻译：`setoption` 保存配置，`set_position`
  停止旧搜索后更新 backend，再构造新 `GameState`（`src/neural/shared_params.cc:43-80`、
  `src/engine.cc:153-167,187-197`、`src/search/search.h:48-55`）。本项目只接受 ONNX，未翻译
  px0 的 backend registry、protobuf 权重和 autodiscover。
- `Search::SendUciInfo` 的单 PV 无温度选边、完整 PV 构造、平均/选择深度、首次 batch 后的
  NPS/EPS、root inherited visits 和 WDL 整数化已翻译（`src/search/classic/search.cc:239-270,
  324-350`）。`MultiPV`、`PerPVCounters` 的 UCI 参数与多行 root 排序已接线
  （`src/search/classic/params.cc:360-368,585-586`、`search.cc:239-246,705-808`）。当前只在
  搜索结束时发送；`ScoreType` 的全部 choice 和零 contempt 默认下的 `WDL_mu` 公式已翻译
  （`src/search/classic/params.cc:587-595`、`search.cc:206-236,275-336`）。worker 中的实时
  responder 回调及可变 contempt/WDL calibration OptionsDict 仍待翻译。
- `engin/src/utils/fastmath.rs` 已逐式翻译 px0 `FastLog2/FastExp2/FastLog/FastExp/FastLogistic`
  （`src/utils/fastmath.h:42-92`）；classic `ComputeCpuct` 已改用 `FastLog`
  （`src/search/classic/search.cc:426-433`），不再以 Rust libm 改变 PUCT 数值路径。

### P4 下一入口

- 已校正 `SearchParams` 的 px0 默认值与 root PUCT 三元组：`FpuValueAtRoot=1.0`、
  `MaxCollisionVisitsScalingEnd=145000`，以及 `CpuctBase/FactorAtRoot` 在
  `RootHasOwnCpuctParams` 下的取值；对应 `src/search/classic/params.h:58-65`、
  `params.cc:543-583,644-655`。

- 已完成 `go searchmoves` 的合法根着法解析和选择/PV/输出共用过滤；对应
  `src/search/classic/wrapper.cc:78-100`、`search.cc:721-724,1668-1740`。下一项是
  多 SearchWorker 的稳定 node 存储边界，不以假并行替代。
- root 尚无已评估 child 的 early-stop/terminal fallback 同样不得逃逸 `searchmoves`；对应
  `src/search/classic/wrapper.cc:78-100`、`search.cc:721-724`。

- `classic/node.h:127-339`、`search.cc:1494-1508`：稳定 node 存储与多 SearchWorker
  selection 的树访问边界；当前 `Vec<Node>` + 整轮 `Mutex` 不能直接承载 px0 多 worker 并发
- `node.cc:245-289`：随稳定 node 存储边界翻译 `MakeSolid`；不能在当前 arena 上伪造
- `search.cc:2103-2364`：释放树锁后的 NN compute/fetch/backup 分阶段并发
- `search.cc:357-368`、`params.h:107-128`：实时 `MaybeOutputInfo` responder 回调，及
  可变 contempt/WDL calibration OptionsDict；结束时 MultiPV/ScoreType 已完成
- px0 二进制 fixed-nodes trace 对拍
