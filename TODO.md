# TODO

> 当前只做 px0 1:1 Rust 翻译。旧实现、lc0 和 KataGo 都不是当前兼容或优化目标。

## P0：棋规与棋盘

### 规则验收（对拍通过）

- [x] ChessBoard 主路径：FEN、走子、攻击、伪/合法着、长捉、杀子力
- [x] magic bitboard 走 **NO_PEXT** 路径（有意不实现 PEXT）
- [x] 移植 `board_test.cc`：perft d1–d5、FEN 校验、合法着集合

- [x] `ChessBoard::hash()`、`startpos_board()`（px0 `hashcat.h`、`board.cc:58`）
- [x] `bitboard::count_few` 稀疏路径（px0 `bitboard.h:76-88` NO_POPCNT 语义）
- [x] `types.h` 对应标量 API：File 默认无效、File/Rank/Square/Piece/Move 文本与翻转辅助（`types.h:31-222`）
- [x] FEN 一致性校验：保留，因为 px0 `board_test.cc:30-34,265-282` 明确要求非法兵/子力布局抛错

## P1：局面历史与重复规则

- [x] 翻译 `position.h`、`position.cc`（px0 `position.h:38-155`、`position.cc:31-197`）。
- [x] 移植 `PositionHistory`、重复计数、rule60、`RuleJudge`。
- [x] 移植 `position_test.cc:28-260`。

## P2：引擎外围

- [x] 删除旧 `engin` 的 history、ONNX、policy vocabulary、UCI 和旧 core API 调用。
- [x] 建立 `GameState` 骨架：`xiangqi_core/src/gamestate.rs` 对应 `gamestate.h:38-47`、`gamestate.cc:35-55`。
- [x] 建立 `GoParams`、`UciResponder`、`EngineController`、`UciLoop` 骨架：`engin/src/uci_loop.rs` 对应 `uciloop.h:42-116`。
- [x] 翻译 `GameState::CurrentPosition`、`GetPositions`（`gamestate.cc:35-55`），并添加逐步 moves 的位置序列对照测试。
- [x] 翻译 `ParseCommand`、`GetOrEmpty`、`GetNumeric`、`ContainsKey`（`uciloop.cc:81-168`）。
- [x] 翻译 `UciLoop::DispatchCommand`、`ProcessLine`（`uciloop.cc:178-261`）；`position ... moves ...` 保留完整 moves。
- [x] 翻译 String/Stdout responder 格式化（`uciloop.cc:263-337`，依赖 `callbacks.h:42-148`）。
- [x] 建立 stdin/stdout UCI 入口；P3 前 `go` 不返回伪搜索结果。

## P3：搜索

- [x] 建立 `SearchBase`、`ClassicSearch`、`Node`、`Edge`、`SearchParams` 文件与类型骨架。
- [x] `SearchBase` + `ClassicEngine` 取代 P2 RecordingEngine（`engine.rs`、`main.rs`）。
- [x] `Edge` prior、Node N/Q/in-flight、terminal、backup、tree reuse（`node.h/.cc`、`search.cc` 子集）。
- [x] `SearchParams` 默认参数子集（`params.h/.cc` 关键项）。
- [x] 单线程 gather → stub NN → backup，支持 `go nodes` / `go movetime`。
- [x] 固定 nodes、movetime、绝对 UCI move/ponder、tree reuse 回归。

## P4：并发与训练

- [x] 建立 `BackendAttributes`、`EvalPosition`、`BackendComputation` 与 P4
  `SearchWorker` 七阶段函数骨架（px0 `src/neural/backend.h:45-138`、
  `src/search/classic/search.h:201-448`、`search.cc:1142-1231,1268-2364`）。
- [x] `UniformBackendComputation` + 单线程 worker 七阶段流水线
  （`InitializeIteration` → `GatherMinibatch` → `CollectCollisions` →
  `RunNNComputation` → `FetchMinibatchResults` → `DoBackupUpdate` →
  `UpdateCounters`）；`p4_skeleton_test` / `worker::tests` 通过。
- [x] `ClassicSearch` 异步 `StartThreads` + `SearchWorker` 接线；`go nodes` /
  `go movetime` / `go wtime` / `go infinite`+`stop` UCI 路径可用（`uci_search_test`、
  `p4_async_search_test`）。
- [x] `stoppers/*` 子集：`Visits` / `Playouts` / `TimeLimit` / `wtime` 预算 /
  `ChainedSearchStopper`；`UniformBackend` NN cache 子集。
- [x] 固定 FEN + px0 `VisitsStopper` trace（`p4_trace_test`；UniformBackend，非 px0 二进制）。
- [x] 翻译 `NetworkAsBackendComputation`：真实 history 编码、2062 policy 索引、
  ONNX batch、WDL 与合法着 softmax（`src/neural/encoder.cc:118-217,229-481`，
   `src/neural/wrapper.cc:49-172`）。主 UCI 在 weights 配置翻译前明确拒绝搜索，
   不再回退到 `UniformBackend`。
- [x] 翻译单 worker `PickNodesToExtendTask` 的 workspace/path-backtrack、碰撞访问分配与
  递归 `PrefetchIntoCache`（`search.cc:1551-1827,1989-2099`）。
- [x] 翻译 two-fold terminal 生成与 tree reuse 回退
  （`search.cc:1510-1550,1899-1959`）。
- [x] 将 `BackendComputation` 收为 task-safe 内部状态，允许并发 `AddInput`，并在
  ONNX compute 期间释放 batch-state 锁（`src/neural/backend.h:75-87`、
  `src/search/classic/search.cc:1423-1462`）。
- [x] 翻译 `PickTask`、队列领取/完成/等待与每轮 `ResetTasks` 生命周期
  （`src/search/classic/search.h:367-445`、`search.cc:1069-1140,1464-1508`）。
- [x] 将 `PickNodesToExtendTask` 改为显式 receiver，供主 minibatch 与 gathering
  task 共用（`src/search/classic/search.h:401-406`）。
- [x] 主选择后等待并汇合 gathering `PickTask.results`
  （`src/search/classic/search.cc:1501-1507`）。
- [x] 将 `PickNodesToExtendTask` 的 DFS state 显式参数化为 `TaskWorkspace`，使主 worker
  与 gathering task 不共享 path state（`src/search/classic/search.h:401-406,425-434`、
  `search.cc:1551-1827`）。
- [x] 翻译 `RunTasks` 的领取、gathering/processing 分派与完成回写，以及 `RunBlocking`
  内常驻 task-worker 的启动、阻塞领取、关闭和 join（`src/search/classic/search.h:205-249`、
  `search.cc:1069-1140`）；直接单元测试入口保留同步 bridge。
- [x] 翻译 gathering split 的 px0 work-size 参数、100-task reservation 与
  passed-off/completed-visits 条件（`src/search/classic/params.cc:604-612`、
  `search.cc:1828-1864`）。
- [x] 翻译 processing split 的 20/8 work-size 参数和前段 task / 尾段主 worker 范围划分
  （`src/search/classic/params.cc:604-612`、`search.cc:1322-1347`）。
- [x] 翻译 `TaskWorkers=-1` 的 CPU/GPU 硬件并发启发式（`src/search/classic/search.h:205-233`）。
- [x] 翻译 collision `maxvisit` 扩容、祖先 in-flight 更新与预算停止
  （`src/search/classic/search.cc:1400-1419`）。
- [x] 翻译 iteration computation 的先释放后创建生命周期
  （`src/search/classic/search.cc:1233-1240`）。
- [x] 翻译 `ResetTasks` 的 100-task 稳定容量预留（`src/search/classic/search.cc:1464-1473`）。
- [x] 翻译 `SearchWorker::RunBlocking` 的每次搜索持久 worker 生命周期
  （`src/search/classic/search.h:235-249`）。
- [x] 将 first-batch 计时移入 worker backup 后的共享统计状态
  （`src/search/classic/search.cc:2158-2173,2331-2364`）。
- [x] 翻译 task queue 的阻塞领取、条件变量唤醒和 `-1` 退出生命周期
  （`src/search/classic/search.cc:1069-1124`、`search.h:225-233`）。
- [x] 翻译构造期 per-task `TaskWorkspace` 分配及 worker 退出队列关闭
  （`src/search/classic/search.h:205-233,357-364`）。
- [x] 翻译 `MaxConcurrentSearchers` 的 `pending_searchers_` acquire/release 时序及
  `SearchSpinBackoff=false` 默认值：slot 仅覆盖 gather/collision/prefetch，并在 NN compute 前归还
  （`src/search/classic/params.cc:399-404,525-526,604-604,632-632`、
  `src/search/classic/search.cc:1142-1195`）。
- [x] 翻译 `backend_waiting_counter_`、`IdlingMinimumWork=0` 与
  `ThreadIdlingThreshold=1`：worker 在 NN compute 前后维护 backend 等待计数，且仅在多
  SearchWorker 时使用 gather early-yield（`src/search/classic/params.cc:498-505,628-629`、
  `src/search/classic/search.cc:1187-1199,1290-1301`）。
- [x] 将 production `SearchWorker` 改为 px0 分阶段 shared tree access：只在 initialize、
  gather/collision/prefetch、fetch/backup 持有 tree mutex，NN compute 不持有；`StartThreads`
  允许多个 SearchWorker，已覆盖两 worker fixed-nodes 回归
  （`src/search/classic/search.cc:1088-1140,1142-1211,1979-2008,2161-2174`）。
- [x] 覆盖两 SearchWorker 与每 worker 一个 task worker 同时运行的 bounded fixed-nodes 路径，
  验证完成后 root `NInFlight=0`（`src/search/classic/search.cc:1088-1211,1322-1347,
  1494-1508,1828-1864`）。
- [x] 翻译 sticky-endgame 的 `MaybeSetBounds`、强制终局父 bounds 传播与
  `AdjustForTerminal` 统计修正（`src/search/classic/search.cc:2175-2289`、
  `src/search/classic/node.cc:300-392`）。
- [x] 翻译 `TaskWorkspace` 的 256-slot selection scratch、policy-prefix score cache
  与 `Node::num_edges_` 上限（`src/search/classic/search.h:348-365`、
  `src/search/classic/search.cc:1575-1825`、`src/search/classic/node.h:320-321`）。
- [x] 翻译 `PickNodesToExtendTask` receiver 的 30-item 条件预留
  （`src/search/classic/search.cc:1570-1573`）。
- [x] 翻译 `ProcessPickedTask` 的逐 leaf `AddInput` 后 out-of-order fetch 顺序，及
  `UpdateCounters -> MaybeTriggerStop` 的 worker stopper 调用；删除私有 exact
  `nodes_budget` 截断（`src/search/classic/search.cc:1423-1462,2331-2334`，
  `src/search/classic/stoppers/stoppers.cc:59-70`）。
- [x] 对齐 `ProcessPickedTask` 对 collision 的入口早退：终局 collision 不能进入 OOO
  fetch/backup，保持其仅由 gather/backup 取消或共享（`src/search/classic/search.cc:1429-1460`）。
- [x] 覆盖 OOO cache-hit 的 `AddInput -> immediate fetch -> Gather backup/remove -> next leaf
  reservation` 顺序（`src/search/classic/search.cc:1370-1393,1423-1462`）。
- [x] 翻译 `StoppersHints` reset/min-update 及 `MaybeTriggerStop` 向下一轮 worker
  回写 remaining playouts（`src/search/classic/search.cc:596-610`、
  `src/search/classic/stoppers/timemgr.cc:35-66`）。
- [x] 翻译 root `current_best_edge` 缓存更新、无温度 best-child 比较及
  remaining-playouts smart pruning（`src/search/classic/search.cc:705-808,1584-1588,
  1726-1742,2241-2249`）。
- [x] 翻译 `SendUciInfo` 的单 PV、平均/选择深度、NPS/EPS、继承 visits 与 WDL 整数化；
  不在缺失 `ScoreType/WDL_mu` 配置时伪造 cp 分数
  （`src/search/classic/search.cc:239-270,324-350`）。
- [x] 翻译 `MultiPV` / `PerPVCounters` UCI 参数、共享 best-child 排序与多行 root PV 输出
  （`src/search/classic/params.cc:360-368,585-586`、`search.cc:239-246,705-808`）。
- [x] 翻译 px0 FastMath，并令 classic `ComputeCpuct` 使用 `FastLog`
  （`src/utils/fastmath.h:42-92`、`src/search/classic/search.cc:426-433`）。
- [x] 翻译 `ScoreType` 选择、默认零 contempt `WDL_mu` 重标定与 `SendUciInfo` score/mate
  输出（`src/search/classic/params.cc:587-620`、`search.cc:206-236,275-336`）。
- [x] 翻译 `go searchmoves` 的合法根着法过滤，并在 root policy prefix、PUCT selection、
  current-best cache、MultiPV/PV/bestmove 共用该过滤表（`src/search/classic/wrapper.cc:78-100`、
  `search.cc:721-724,1668-1740`）。
- [x] 覆盖 root 尚无 child 的 early-stop fallback，保证其不逃逸 `searchmoves`
  （`src/search/classic/wrapper.cc:78-100`、`search.cc:721-724`）。
- [x] 对齐 `SearchParams` root PUCT 三元组与默认值：`CpuctBaseAtRoot`、
  `CpuctFactorAtRoot`、`FpuValueAtRoot=1.0`、collision scaling end `145000`
  （`src/search/classic/params.h:58-65`、`params.cc:543-583,644-655`）。
- [x] 翻译 task worker split、任务队列与常驻 task-worker 生命周期
  （`search.h:205-249,367-448`，`search.cc:1069-1140,1268-1508,1828-1897`）。
- [x] 对齐常驻 task worker 的 idle 与析构退出：`GatherMinibatch` 后 `task_count=-1` 仅休眠，
  `exiting` 才终止 worker；覆盖下一轮 `ResetTasks` 复用（`search.cc:1069-1140,1182-1185,
  1464-1492`）。
- [ ] 补齐 px0 完整 out-of-order 及其多 SearchWorker 交互
  （`search.cc:1268-1508,2109-2331`）。
- [x] 用 stable boxed arena 翻译 px0 `MakeSolid` 的 child ownership 语义：在满足
  leaf/terminal in-flight 条件时补齐每个 edge 的 child slot，并在 backup 的同一时序刷新
  root best-edge cache（`src/search/classic/node.cc:245-289,394-405`、
  `src/search/classic/search.cc:2211-2217`）。
- [x] 提供 px0 / engin fixed-nodes 原始 UCI transcript 采集脚本；由于本机 px0
  使用 `pb.gz` 而 engin 使用 ONNX，记录用于人工对照 nodes/PV/bestmove，不做不成立的
  数值相等断言（px0 `src/chess/uciloop.cc:178-254`、
  `src/search/classic/wrapper.cc:53-141`）。
- [x] 翻译 px0 `Engine::UciPonderForwarder` 的非 owning responder 注册边界，并令
  `ClassicSearch` 在 worker 生命周期内持有 thread-safe callback；注销前先停止/join 搜索
  （`src/engine.cc:81-136,238-250`、`src/search/search.h:45-99`）。
- [x] 删除 Engine 对搜索结果的手工 drain，最终 `info` / `bestmove` 仅通过 SearchBase
  responder 输出，避免 worker/watchdog 接入后双发（`src/engine.cc:85-116,204-208`、
  `src/search/classic/search.cc:596-620`）。
- [x] 翻译 px0 `StartThreads` 的 watchdog 拓扑：`go` 非阻塞、watchdog 驱动 stopper 和
  最终完整 MultiPV/bestmove 输出、`wait` 只 join、`stop` 只发停止请求；覆盖
  `go infinite -> stop -> wait` 的唯一 bestmove（`src/search/classic/search.cc:595-620,
  874-1041`）。
- [x] 翻译 watchdog 内 `MaybeOutputInfo` 的 root-edge/depth/seldepth/5 秒频率判断，且保持
  tree -> current-best 的同步顺序（`src/search/classic/search.cc:51,357-382,2211-2249`）。
- [ ] 逐函数翻译 px0 minibatch、prefetch、tree reuse 与多 task worker 并发路径。
- [x] 对 fixed-nodes 覆盖 solid node 与多 SearchWorker/task split 的 selection 交互，核对
  solid child 不会造成 collision/in-flight 泄漏或重复扩展
  （`src/search/classic/node.h:451-523,556-626`、`search.cc:1494-1864`）。
- [x] 翻译 worker/watchdog 的 `MaybeOutputInfo` 实时 responder 回调与 collision-only
  iteration 的 10ms 退让（`src/search/classic/search.cc:357-382,2331-2364`）。
- [ ] 翻译可变 contempt/WDL calibration `OptionsDict`，使 WDL 重标定在 NN value 进入
  backup 前使用 px0 的完整参数计算（`src/search/classic/params.cc:57-174,688-703`、
  `search.cc:206-236,2117-2154`）。
- [x] 翻译 `WeightsFile` 到正式 `OnnxBackend` 的 UCI/engine 配置子集；保持 px0 配置与
  `SetPosition` 时序，不实现多 backend registry、protobuf weight 或 autodiscover
  （`src/neural/shared_params.cc:43-80`、`src/engine.cc:153-167,187-197`、
  `src/search/search.h:48-55`）。

## 后续才允许做

- [ ] 记录与 lc0/KataGo 的明确差异，再决定是否吸收优化。
