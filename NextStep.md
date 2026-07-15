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
  `Abort + Wait` 上一搜索。未翻译的 `depth/mate/ponder/clock manager` 已明确拒绝，不能静默执行。
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

## P4：task-worker 生命周期已收口

此前的 Rust 实现把同一个可变 `SearchWorker` 借给 scoped task thread，导致 node index 越界和
poisoned tree lock。现已改为受限 `TaskTreeBridge`：只有 scoped task thread 可通过 active phase
访问 `NodeTree`，主 worker 在 `WaitForTasks` 前持续持有对应 tree phase。参考 px0
`src/search/classic/search.h:205-244,435-445`、`search.cc:1069-1140,1485-1508,1828-1897`。

### 当前决策

CPU backend 保持 px0 `task_workers_=0` 分支；GPU backend 在 `TaskWorkersPerSearchWorker=-1` 时按
px0 `search.h:210-224` 解析为每个 SearchWorker 最多四个 helper，并以 scoped thread 生命周期运行。
现阶段优先验证 task split、多个 `SearchWorker`、minibatch、cache、prefetch 与 backend computation
的组合时序。

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

task worker 不直接执行 GPU 推理。它在 px0 中并行执行 gathering/processing，减少 selection、node
extend 和 `BackendComputation::AddInput` 的 CPU 准备间隔；这能帮助持续向 GPU 提交输入，但不是 GPU
吞吐的唯一或首要来源。持续喂卡首先依赖多个搜索 worker、真实 batch、共享 backend computation 与
backend 的异步调度。task worker 已接通，但 DirectML/ONNX 的实际吞吐仍须单独验收。

GPU task-worker 测试已确认 helper 实际领取任务并在固定 visits 后清空 root `NInFlight`；这不是
同步 split。默认 task-worker 数和 CPU fallback 均按 px0 `search.h:210-224`。

`WorkerTree` 已收为显式 tree-phase 借用：direct 单测树和共享生产树均在 `with_tree` /
`with_tree_read` 中临时借出 `NodeTree`，selection、processing、fetch、backup 都把该借用逐层传入。

真实 ONNX 已由 `CachingBackend` 包裹，对照 px0 `src/neural/memcache.h:34-45` 与
`memcache.cc:38-190`：缓存 key 是当前局面 hash，合法着数量保护碰撞，cache miss 仅在
`ComputeBlocking` 后回填，`Engine::NewGame` 清 cache。FIFO 容器保留 px0 `HashKeyedCache` 的
“不替换已有 key、按插入顺序淘汰”语义（`src/utils/cache.h:35-57,69-105,214-230`）；默认和 UCI
`NNCacheSize` 均为 px0 的 `2000000`（`src/neural/shared_params.cc:63-82`）。
这对应 px0 `nodes_mutex_` 的 phase 边界（`src/search/classic/search.cc:1142-1211,1494-1508`），并删除了
此前 `active: *mut NodeTree` 的无边界访问桥接；当前 `TaskTreeBridge` 将 raw pointer 限定在 active
phase + scoped task-thread 内，不改变 px0 的 selection、in-flight 或 backup 算法。

共享 `NodeTree` 的通用 `RwLock` 已采用 `parking_lot`，替代标准库会 poison 的锁接口；这只承担锁
机制，不承载任何搜索策略或节点语义。参考 px0 的 `nodes_mutex_` 使用边界
`src/search/classic/search.cc:1142-1211,1494-1508`。

P4 已按下列区间收口；不再为了时钟管理改变 task-worker 或 backend 流水线。
完整 `wtime/btime` 需先整体翻译 px0 `stoppers/factory.cc:44-115` 的 TimeManager 选择与默认
`legacy`，再翻译被选择的 manager，不能只拿 `simple.cc` 替代默认行为。这是后续独立 UCI
完整性任务，不属于 P4 搜索并发基建。

P4 已完成的逐函数翻译点：

1. 已完成队列原子状态机：`src/search/classic/search.h:435-445`、
   `src/search/classic/search.cc:1069-1119,1464-1483`。
   - `task_taking_started`、task claim、idle、wake、close 与重用已在
     `crates/engin/src/search/classic/worker.rs` 对照实现并有多线程领取回归。
2. `src/search/classic/search.cc:1142-1231,1977-2008,2109-2334`
   - 多 SearchWorker + scoped task worker 的 fixed-visits shared-tree 回归已通过；继续对照
     `MaxConcurrentSearchers`、out-of-order backup 与 counter 时序。
3. DirectML release UCI 已验证固定 nodes、`go infinite -> stop -> wait`、`position ... moves ...`、
   backend reload，以及 `go infinite -> go nodes` / `go infinite -> position ... -> go nodes` 的旧搜索
   静默回收。参考 `src/engine.cc:148-224`、`src/search/classic/wrapper.cc:100-140`。后续只补可观测的
   root `NInFlight=0` 断言，不重复改变 UCI 生命周期。

每次只翻译一个连续参考区间，补对应回归，再提交。raw pointer / `unsafe impl Send` 仅允许保留在
`TaskTreeBridge`、`TaskWorkerRunner` 两个有 px0 行号和 scoped-lifetime 注释的内部类型。

### 已确认的前置缺口

px0 `Node` 本身不是原子对象；`PickNodesToExtend` 在 `search.cc:1494-1501` 持有
`nodes_mutex_`，task thread 在该锁保护的约定下执行 `PickNodesToExtendTask`。`SharedMutex::Lock`
本身只是 `std::unique_lock<std::shared_timed_mutex>` 的包装（`src/utils/mutex.h:93-125`）：px0 依赖
task split 后的逻辑不重叠，让 task thread 在主线程持锁期间直接修改普通 `Node`。Rust 当前
`NodeTree` 同样是普通可变对象，且 `WorkerTree` 只能在一个 worker 的 tree phase 中临时激活。
因此，真实 task thread 已通过“主线程独占 phase + 已切分子任务”的受限 bridge 接线。它使用
`*mut SearchWorker` 和 `unsafe impl Send`，但只存在于 scoped thread 生命周期，且 `WaitForTasks`
完成前不清空 active tree pointer；不得扩大该例外或改为每次任务全树锁。

这不是仅把 `Vec<Box<Node>>` 改成线程安全容器就能解决的问题。px0 自己声明 `Edge_Iterator` 和
`VisitedNode_Iterator` 非线程安全（`src/search/classic/node.h:423-436,547-551`），而
`Edge_Iterator::Actualize` 又明确允许其他 task 在 iterator 调用间隙创建 sibling
（`src/search/classic/node.h:485-525`）。此外 twofold 修正会沿 parent 链回写
（`src/search/classic/search.cc:1510-1550`）。Rust 端以 active tree phase 保留该外部同步契约；
不得以每节点锁替换后假定算法已经等价。

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

P4 只有在真实 ONNX/DirectML 下 task worker 生命周期、固定 nodes、`stop`、`position ... moves ...`
和 tree in-flight 清理均对照 px0 通过后才能关闭。
