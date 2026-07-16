# NextStep

## 当前状态

当前唯一工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

P0-P3 的规则、历史、UCI、classic tree/worker 基础已建立。P4 的正式 ONNX 路径使用
`124x10x9 -> 2062 + WDL`，Windows 实测后端为 DirectML。

## P1-P3 进入 P4 前复核（2026-07-15）

- P1：Rust `xiangqi_core` 的 release 全量对拍已通过：px0 移植的 `board_test.cc` depth-5 perft、
  `position_test.cc` 的 history/repetition/RuleJudge 与 FEN/hash 用例均为绿。参考
  `src/chess/board_test.cc:70-232`、`position.cc:41-205`。
- P2：`position ... moves ...` 保存完整 history；每次 `go`、`position`、`ucinewgame` 都先
  `Abort + Wait` 上一搜索。px0 默认 legacy clock manager 已翻译；未翻译的 `depth/mate/ponder`
  仍明确拒绝，不能静默执行。
  参考 `src/engine.cc:148-235`、`src/search/classic/wrapper.cc:100-150`、
  `src/search/classic/stoppers/common.cc:118-186`。
- P3：Node 的 edge policy 编码、`MakeSolid`、terminal visit 反转、tree reuse 与 UCT/参数默认值已有
  对应单测，且 `cargo test -p engin --lib` 通过。参考 `src/search/classic/node.cc:161-543`、
  `params.cc:543-640`、`search.cc:408-433`。

结论：P1-P3 没有阻断 P4 的已知语义偏差。P4 只能继续翻译明确列出的 px0 连续区间；不得重新开放
未翻译的 UCI 预算或用本地启发式替代 px0 时间管理。

`nn/` 的训练入口已完成独立收口：参考 pxzero-training 的 YAML 布局，但不移植其旧 TensorFlow
兼容层；当前唯一启动方式是 `train_px0.py --config <yaml>`。这不是 P4 搜索任务的一部分。

`ClassicEngine::new()` 现在是正式 UCI 构造：它在 `WeightsFile` 被下一次 `position` 成功加载前
不创建搜索对象，不再用 `UniformBackend` 伪装可搜索状态。参考 px0
`src/engine.cc:137-197,206-219` 与 `src/neural/shared_params.cc:43-80`。

集成测试通过 `engine.search().expect(...)` 明确区分两类构造：`ClassicEngine::uniform()` 的测试
后端必定有 search，正式 `ClassicEngine::new()` 则允许在权重未加载时没有 search。这样测试不会
重新引入已废止的 `unavailable()` 占位构造，也不改变 px0 的延迟 backend 生命周期。

## P4：单 worker 搜索主线

### P4 task-worker 重译前置约束（2026-07-16）

px0 的 task-worker 不是独立的普通队列：主 worker 在持有 `nodes_mutex_` 的同一 tree phase 内，
让 task thread 直接执行 `PickNodesToExtendTask` 或 `ProcessPickedTask`。参考
`src/search/classic/search.cc:1069-1140,1423-1462,1485-1508,1551-1897` 与
`search.h:348-445`。`Node` 的 visits、children 与 edge 存储均不是原子；px0 的正确性依赖 task split
后的子树/`minibatch_` range 不重叠，不能用“每个任务各拿一把整树锁”替代。

重新翻译必须按以下顺序完成：

1. 将 Rust `SearchWorker` 的 task 所需状态按 px0 字段划分：task 独占 `TaskWorkspace`、`PickTask`
   副本与 gathering results；主 worker 独占 `minibatch`、`BackendComputation`、iteration counters 与
   UCI/stop 生命周期。不得跨线程共享 `&mut SearchWorker`。
2. 只在 `PickNodesToExtend` 的 px0 tree phase 内向 task 公开任务；`WaitForTasks` 返回前，主 worker
   不得进入 `CollectCollisions`、NN、fetch 或 backup。task range 的不重叠条件必须由入队处
   `search.cc:1329-1362,1828-1897` 直接证明，而不是由调用者约定。
3. 在任何并行实现前补回归：同一未扩展叶只能成功一次 `TryStartScoreUpdate`/`ExtendNode`；每轮
   `WaitForTasks` 后 `NInFlight == 0`；固定 nodes、`go movetime`、`stop -> wait` 和
   `position ... moves ...` 都要在真实 ONNX/DirectML 下通过。
4. 当前保留 `task_workers=0`。只有上述结构与回归完成后，才可评估是否存在最小的、局限于 tree phase
   的 Rust `unsafe`；不得恢复 `*mut SearchWorker`、`unsafe impl Send` 或 raw-pointer bridge。

collision-only `sleep(10ms)` 保持与 px0 一致：它位于 `SearchWorker::UpdateCounters`
(`src/search/classic/search.cc:2337-2351`)，只在一次 iteration 完全没有非 collision 工作时退避。

watchdog 已按 px0 `Search::WatchdogThread` / `FireStopInternal` 的等待边界收口：`meta` mutex
对应 `counters_mutex_`，stopper 触发或 UCI `stop/abort` 都通知 `watchdog_cv`，空闲等待按
`estimated_remaining_time_ms` 限制为 `1..=100ms`。参考
`src/search/classic/search.cc:981-1034`。这只替换固定 1ms polling，不改变时间分配或 stopper 策略。

`nps_start_time` 同样归 watchdog/controller 所有：首次观测到完成 playout 后才开始计时，UCI
`nps/eps` 与 `NodesPerSecondLimit` 都使用该时钟；在时钟尚未建立时，限速回退到 move-start。参考
`src/search/classic/search.cc:249-264,393-398,908-918,1213-1231`。不再由 worker 在首个 backend
batch 回写时私自设定时间原点。

`StoppersHints` 的所有权已对齐 px0：每个 `SearchWorker` 持有自己的
`latest_time_manager_hints_`，watchdog 另持一份，仅通过同一个 stopper 更新；`go nodes` 的真实
剩余预算在第一轮完成后由各自的 stopper pass 用 `total_nodes` 发布。参考
`src/search/classic/search.h:368-369, search.cc:596-610,908-922,981-1017,1268-1284`。不能把它放进
共享 worker state，否则多个 worker 会互相覆盖 gather 提示。

因此 `WorkerSearchState` 只保存 px0 `Search` 共享计数和同步状态，不再暴露没有 px0 对应物的
`remaining_playouts` 构造参数。

stopper 现在也遵守 px0 的 root-first-visit gate：root 尚无访问时只等待，不执行 budget stopper。
参考 `src/search/classic/search.cc:596-610`。

已对齐 tree reuse 的 stopper 统计：`total_nodes` 必须是本轮 `total_playouts + initial_visits`，而
`nodes_since_movestart` 只统计本轮 playouts。参考 px0 `search.cc:908-922`。

`WorkerTree` 已收为显式 tree-phase 借用：direct 单测树和共享生产树均在 `with_tree` /
`with_tree_read` 中临时借出 `NodeTree`，selection、processing、fetch、backup 都把该借用逐层传入。

真实 ONNX 已由 `CachingBackend` 包裹，对照 px0 `src/neural/memcache.h:34-45` 与
`memcache.cc:38-190`：缓存 key 是当前局面 hash，合法着数量保护碰撞，cache miss 仅在
`ComputeBlocking` 后回填，`Engine::NewGame` 清 cache。通用缓存容器已改为 `quick_cache` 分片 S3-FIFO，
value 是 `Arc<EvalResult>`，避免 cache hit 复制 policy 向量；这刻意不保留 px0 `HashKeyedCache` 的严格
FIFO 淘汰（`src/utils/cache.h:35-57,69-105,214-230`），但仍原子地保留首个 completed value、hash 与
合法着数量 guard。默认和 UCI `NNCacheSize` 均为 px0 的 `2000000`
（`src/neural/shared_params.cc:63-82`）。
这对应 px0 `nodes_mutex_` 的单 worker phase 边界（`src/search/classic/search.cc:1142-1211,1494-1508`）。

共享 `NodeTree` 的通用 `RwLock` 已采用 `parking_lot`，替代标准库会 poison 的锁接口；这只承担锁
机制，不承载任何搜索策略或节点语义。参考 px0 的 `nodes_mutex_` 使用边界
`src/search/classic/search.cc:1142-1211,1494-1508`。

P4 的单 worker/minibatch/OOO/cache/stopper 主线可用。GPU task-worker 的 scoped raw-pointer 结构仍保留为
翻译对象，但真实 ONNX 的 `go infinite -> stop` 回归已复现重复 `ExtendNode`；因此运行时激活门固定为
`active_task_workers=0`，而 UCI/px0 的 `TaskWorkers` 解析仍保留作诊断。不能把当前后台路径称为完成对齐。
下一步必须逐行核实 `PickNodesToExtendTask` 的 in-flight/collision 互斥，参考 px0
`search.cc:1551-1897`，并用真实 ONNX 覆盖固定 visits、movetime 与 stop/wait，回归稳定前不得解除门控。

第一步已完成：Rust 的 `IterationState` 已独占封装 px0 `minibatch_`、`computation_` 与
`number_out_of_order_`（`src/search/classic/search.h:419-427`），其生命周期限定为
`InitializeIteration -> UpdateCounters`，仍只由主 worker 写入。下一步不是启用线程，而是把
`PickTask` 的输入、workspace 与结果变成可移动的独占值，再证明 px0 processing range 不重叠。

第二步已完成：`PickTaskQueue` 领取 task 时移动其唯一所有权，完成后把同一个 task（含 gathering
结果）回填 slot；不再克隆 `moves_to_base` 或临时 results。该 Rust 所有权映射对应 px0
`RunTasks` 的领取/完成区间（`src/search/classic/search.cc:1069-1140`）和 `PickTask::results` 的合并区间
（`1494-1508`）。workspace 和 tree 仍未可安全移动到后台线程。

第三步已完成：生产 `PickTaskQueue` 只保留 `task_count / claim / complete / wait` 的同步 phase；px0
`RunTasks` 的 sleep/exit 条件仅作为测试状态机保留（`src/search/classic/search.cc:1069-1124`）。在没有真实
task owner 前，不把该测试机制伪装成后台 task worker，也不让它影响 `task_workers_=0` 搜索路径。

第四步已完成：`PickNodesToExtendTask` 的 Rust 入口只接收 `SelectionContext`、tree、task-owned workspace
和结果 receiver。`SelectionContext` 精确收纳 px0 selection 所读的 params、root filter、best-edge、stopper
hints 与 task queue（`src/search/classic/search.cc:1551-1897`）；它不含 minibatch/backend/其他 workspace。
twofold 回退改为纯 tree helper，对应 `search.cc:1510-1550`。

第五步已完成：`ExtendNode` 只接收复制的 `ExtendContext`、tree 与 task workspace history，不再借用可变
`SearchWorker`。该 context 仅含 px0 `ExtendNode` 使用的 root-history 长度和 twofold 开关，参考
`src/search/classic/search.cc:1899-1974`；这为 processing range 的独占传递消除了一个 worker 别名来源。

第六步已完成：`ProcessPickedTask`/`FetchSingleNodeResult` 已收为 `ProcessingContext + &mut [NodeToProcess]`
的 range 操作；computation 以共享 `BackendComputation` 引用进入，task 只能回写自身 item。参考 px0
`src/search/classic/search.cc:1423-1462,2109-2156`。tree 写入范围仍须在启用后台 task 前证明不重叠。

第七步已完成：`GatherMinibatch` 的 processing split 已成为纯 range 规划，严格对应 px0
`src/search/classic/search.cc:1322-1362`。回归验证混合 collision batch 的 queued ranges 互不重叠，且均在
main suffix 之前；这只证明 minibatch 所有权，未证明 gathering task 的 tree 子树写入范围。

第八步已完成：px0 的 `task_workspaces_ / main_workspace_` 已映射为 Rust `TaskRunner` 的独占 workspace，
参考 `src/search/classic/search.h:348-365,441-445`。当前仅有 main runner；未来每个后台 task runner 必须自带
workspace，不能共享主 runner 的 history/path scratch。

第九步已完成：正式 x7 ONNX 没有 moves-left head，因此 selection 按 px0 在
`backend_attributes_.has_mlh == false` 时构造禁用的 `MEvaluator()`；访问过与未访问 child 的 M utility
均为零，不能留下隐式的 moves-left 启发式。参考 `src/search/classic/search.cc:60-114,1596,1680-1692`。

第十步已完成：gathering 子树 handoff 已收为独立 helper。它只在 task 成功发布后清零父 DFS 的该 edge
visit budget；队列满、未扩展或 terminal child 必须继续由父任务处理。参考
`src/search/classic/search.cc:1828-1864`。回归覆盖 task 的 root/path/depth/budget 与父边去重；这仍不表示
后台 task-worker 已安全可用。

第十一步已完成：px0 `task_workspaces_[tid]` 的 gathering/processing 分支已映射为 `TaskRunner` 方法；
runner 只拥有自己的 DFS/history scratch，gathering task 保留自己的结果 vector，processing 只接收已拆分的
minibatch slice。参考 `src/search/classic/search.cc:1116-1129,1322-1362`。当前仍由 main runner 同步调用，
不能据此开启 `task_workers`。

第十二步已完成：核实 px0 在 `PickNodesToExtend` 内由主 worker 持有 `nodes_mutex_`，同时 task threads 直接
读写 tree；其 C++ 接口以 `NO_THREAD_SAFETY_ANALYSIS` 明确绕过静态别名检查。参考
`src/search/classic/search.cc:1485-1508,1551-1897`。Rust `NodeTree` 当前是普通可变 arena，不能安全地按该
别名模型启动线程；本轮只补齐 `TaskRunner` processing 独占 workspace 回归，不以全树锁伪造并发。

第十三步已完成：进一步核实 px0 `Node` 的 `wl_/d_/m_/n_/n_in_flight_`、child sibling 链和 edge 排序都不是
atomic；`Edge_Iterator` 明确要求外部同步却允许 sibling 在 iterator 调用间插入。参考
`src/search/classic/node.h:132-260,423-525`、`node.cc:245-373`。因此 Rust 不能把当前普通 arena 的 `&mut
NodeTree` 借给多个 runner 后仍声称是安全的逐行翻译。P4 后台 task-worker 的下一步是明确阻塞项：必须先给出
不依赖 raw-pointer alias、且不以整树锁串行化的 Rust 所有权/并发表示，并逐段对照上述 px0 行；在此之前维持
`task_workers=0`。

第十四步已完成：`TaskWorkers` 已恢复 px0 构造期解析：显式非负值原样保留，`-1` 时 CPU 为 0，GPU 按
`min(hardware_concurrency / max(search_threads - 1, 1) - 1, 4)` 推导。参考
`src/search/classic/search.h:205-224`。当前已按该数值规划 gather/processing split，但 `WaitForTasks` 仍由
owner 同步 drain；下一步才将该 drain 替换为 scoped task threads。

第十五步已完成：task queue 已补齐 scoped phase 的 `publish -> wait -> seal -> exhaust` 协议；worker 在主
DFS 尚可发布 gathering task 时等待，seal 后只在已领取任务全部耗尽才退出。参考
`src/search/classic/search.cc:1069-1140,1485-1508`。当前同步 drain 也复用该协议，因此行为不变；下一步可在
同一协议下替换为真实 scoped task threads。

第十六步已完成：已接入 scoped task threads。每个 thread 独占 `TaskRunner`，从 queue 领取 task；tree 与
processing range 通过仅在 scope 内有效的 raw pointer 传入，主 worker 在 join 前不再访问它们。参考
`src/search/classic/search.cc:1069-1140,1322-1362,1485-1508`。当前 scope 在 producer seal 后启动，因此已
验证 task 执行/回填生命周期，但尚未达到 px0 “主 DFS 发布时 task 已并行消费”的持续流水线；下一步只能在
固定 ONNX 回归后把 spawn 前移到 gathering/processing producer phase。

第十七步已完成：gathering scope 已前移到主 `PickNodesToExtendTask` 的 producer phase。task runners 先等待
queue，主 DFS 可持续发布 gathering 子树，随后 seal/join 并按 px0 顺序合并 results。参考
`src/search/classic/search.cc:1069-1140,1485-1508,1828-1864`。processing split 仍在主 suffix 结束后才
启动 task scope，是下一步唯一未对齐的 pipeline 时序。

第十八步已完成：processing scope 已前移到 px0 的 split 点：后台 runners 消费已发布的非重叠 range，同时
main runner 处理最终 suffix，随后 seal/join。参考 `src/search/classic/search.cc:1322-1362,1423-1462`。至此
P4 的 gather/processing task 时序、queue lifecycle、独立 workspace、task result merge 与 raw tree-phase
边界均已翻译；下一步只做真实 ONNX/DirectML 固定 visits、movetime、stop/wait 与 `NInFlight==0` 回归，
不再继续扩结构。

第十九步（回归门控）已完成：真实 ONNX 的 `go infinite -> stop` 触发 `Node::create_edges` 的重复扩展断言，
说明当前 scoped gathering 对普通 Rust `NodeTree` 的并发写入尚未满足 px0 的 in-flight/collision 时序。运行时
`active_task_workers` 因此固定为 0；保留 `TaskWorkers` 的 px0 解析和 scoped 实现，仅供逐行诊断，不能作为
生产并发搜索。下一步先对照 `src/search/classic/search.cc:1551-1897` 和
`src/search/classic/node.cc:245-373` 建立该回归的最小复现与唯一扩展证明。

第二十步（并发语义核对）已完成：px0 `PickNodesToExtend` 在主线程持有 `nodes_mutex_` 时启动 task，task
直接执行 `PickNodesToExtendTask`/`ProcessPickedTask`，见 `src/search/classic/search.cc:1069-1140,1485-1508`；
该 mutex 的实现是 `std::shared_timed_mutex` 包装，见 `src/utils/mutex.h:93-119`，并不把这把锁交给 task。
px0 依赖 `Node::TryStartScoreUpdate` 的 `n_ == 0 && n_in_flight_ > 0` 排他规则，见
`src/search/classic/node.cc:348-365`。现有 Rust raw pointer 直接并发访问普通 arena，会令这组非原子读写发生
数据竞争，真实 ONNX 的重复 `ExtendNode` 正是其表现。后续不得以“节点已有 edges 则跳过”掩盖；必须先给出能
保持该 in-flight 原子性的 Rust 节点存储/同步边界，才能重启 task worker。

UCI 时间管理已翻译 px0 工厂默认 `legacy` 的连续区间：`stoppers/factory.cc:44-115`、
`legacy.cc:43-174`、`stoppers.cc:39-129`、`common.cc:118-165`。正式支持
`MoveOverheadMs`、`Slowmover` 与 `go wtime/btime/winc/binc/movestogo`；`TimeManager` 的其他 px0 变体
(`simple/smooth/alphazero`) 仍不暴露，避免把未翻译配置伪装成可用功能。

完整 task-worker 是明确缺口，不得把已删除的 `TaskTreeBridge` / `TaskWorkerRunner` 当作可启用实现；后续只能
从 px0 `search.h:205-244,348-445`、`search.cc:1069-1508` 重新建立无别名的数据所有权边界。此前的
`*mut SearchWorker` / `unsafe impl Send` bridge 在真实 ONNX 正时间搜索会触发重复 `ExtendNode`，已
从代码删除；不得恢复或以全树锁掩盖该错误。

这不是仅把 `Vec<Box<Node>>` 改成线程安全容器就能解决的问题。px0 自己声明 `Edge_Iterator` 和
`VisitedNode_Iterator` 非线程安全（`src/search/classic/node.h:423-436,547-551`），而
`Edge_Iterator::Actualize` 又明确允许其他 task 在 iterator 调用间隙创建 sibling
（`src/search/classic/node.h:485-525`）。此外 twofold 修正会沿 parent 链回写
（`src/search/classic/search.cc:1510-1550`）。未来 Rust 端必须先证明 task 的节点范围和 workspace
所有权不重叠；不得以每节点锁或 raw-pointer alias 假定算法已经等价。

## 验收

```powershell
cargo fmt --check
cargo test -p engin --lib
cargo build --release -p engin
```

DirectML 本地冒烟：

```powershell
@('uci', 'isready', 'position startpos', 'go nodes 1000', 'wait', 'quit') |
  .\target\release\engin.exe
```

P4 的单 worker 子目标已可验收；P4 只有在真实 ONNX/DirectML 下安全 task-worker 生命周期、固定
nodes、`stop`、`position ... moves ...` 和 tree in-flight 清理均对照 px0 通过后才能完全关闭。
