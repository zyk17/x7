# NextStep

## 当前状态

当前唯一工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

P0-P3 的规则、历史、UCI、classic tree/worker 基础已建立。P4 的正式 ONNX 路径使用
`124x10x9 -> 2062 + WDL`，Windows 实测后端为 DirectML。

`nn/` 的训练入口已完成独立收口：参考 pxzero-training 的 YAML 布局，但不移植其旧 TensorFlow
兼容层；当前唯一启动方式是 `train_px0.py --config <yaml>`。这不是 P4 搜索任务的一部分。

`ClassicEngine::new()` 现在是正式 UCI 构造：它在 `WeightsFile` 被下一次 `position` 成功加载前
不创建搜索对象，不再用 `UniformBackend` 伪装可搜索状态。参考 px0
`src/engine.cc:137-197,206-219` 与 `src/neural/shared_params.cc:43-80`。

## P4 未完成：按 px0 重建 task-worker 生命周期

此前的 Rust 实现把同一个可变 `SearchWorker` 借给 scoped task thread，导致 node index 越界和
poisoned tree lock。该路径已移除，正式搜索当前禁用 task split，不能再称为 px0
task-worker 完成。

### 当前决策

当前保持安全基线：正式搜索继续禁用 task split（等价于 px0 CPU backend 的 `task_workers_=0`
分支），优先验证多个 `SearchWorker`、minibatch、cache、prefetch 与 backend computation 的主线。

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

stopper 现在也遵守 px0 的 root-first-visit gate：root 尚无访问时只等待，不执行 budget stopper。
参考 `src/search/classic/search.cc:596-610`。

已对齐 tree reuse 的 stopper 统计：`total_nodes` 必须是本轮 `total_playouts + initial_visits`，而
`nodes_since_movestart` 只统计本轮 playouts。参考 px0 `search.cc:908-922`。

task worker 不直接执行 GPU 推理。它在 px0 中并行执行 gathering/processing，减少 selection、node
extend 和 `BackendComputation::AddInput` 的 CPU 准备间隔；这能帮助持续向 GPU 提交输入，但不是 GPU
吞吐的唯一或首要来源。持续喂卡首先依赖多个搜索 worker、真实 batch、共享 backend computation 与
backend 的异步调度。安全 tree-phase 重构完成前，不以 task worker 作为当前性能验收前提。

当前测试不再把 `TaskWorkersPerSearchWorker` 的请求误称为已执行的同步 split；运行时会明确解析为
`0`，直到安全 task-worker 生命周期落地。

下一步必须逐函数翻译，不补写并发捷径：

1. 已完成队列原子状态机：`src/search/classic/search.h:435-445`、
   `src/search/classic/search.cc:1069-1119,1464-1483`。
   - `task_taking_started`、task claim、idle、wake、close 与重用已在
     `crates/engin/src/search/classic/worker.rs` 对照实现并有多线程领取回归。
2. `src/search/classic/search.h:205-249,357-448`、`search.cc:1122-1140,1268-1508`
   - 继续翻译一个 `SearchWorker` + 每 task thread 一个 `TaskWorkspace` 的执行关系；不得共享
     Rust `&mut SearchWorker`，也不得把一个 workspace 交给多个 task thread。
3. `src/search/classic/search.cc:1828-1897`
   - 翻译 task split 和 `ResetTasks`；`task_count=-1` 只表示 idle，`exiting` 才关闭线程。
4. `src/search/classic/search.cc:1142-1231,1977-2008,2109-2334`
   - 对照 `MaxConcurrentSearchers`、tree phase、out-of-order backup 与 counter 时序。

每次只翻译一个连续参考区间，补对应回归，再提交。禁止重新引入原始指针或 `unsafe impl Send` 来
跨线程共享一个 `SearchWorker`。

### 已确认的前置缺口

px0 `Node` 本身不是原子对象；`PickNodesToExtend` 在 `search.cc:1494-1501` 持有
`nodes_mutex_`，task thread 在该锁保护的约定下执行 `PickNodesToExtendTask`。Rust 当前
`NodeTree` 同样是普通可变对象，且 `WorkerTree` 只能在一个 worker 的 tree phase 中临时激活。
因此，真实 task thread 接线前必须先逐函数建立能表达这一所有权的安全 tree-phase 边界；不能通过
`*mut SearchWorker`、`unsafe impl Send` 或把任务同步执行来伪造 px0 task-worker。

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
