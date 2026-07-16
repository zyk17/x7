# TODO

> 当前只做 px0 1:1 Rust 翻译。旧实现、lc0 和 KataGo 都不是当前兼容或优化目标。

## 已验证基建

- P0/P1：`src/chess/types.h`、`bitboard.h`、`board.h/.cc`、`position.h/.cc` 的 Rust 基础与
  legal move/history/rule60/RuleJudge 回归。
- P2：`src/chess/gamestate.*`、`uciloop.*`；`position ... moves ...` 保留完整历史。
- P3：`src/search/classic/node.*`、`params.*`、基本 selection/extend/backup/tree reuse。
- P4 已接通的部分：`src/neural/backend.h:45-138`、`src/neural/encoder.cc:118-217,229-481`、
  `src/neural/wrapper.cc:49-172`、`src/neural/memcache.h:34-45,memcache.cc:38-190`、
  `src/search/classic/search.cc:1142-1231,1268-1508,1551-1827,1977-2334` 的 ONNX/MemCache、
  单 worker/minibatch/prefetch/OOO/shared-tree 子集。
- P4 legacy 时钟预算：`src/search/classic/stoppers/factory.cc:44-115`、`legacy.cc:43-174`、
  `stoppers.cc:39-129`、`common.cc:118-165`；已支持 `MoveOverheadMs`、`Slowmover` 与
  `go wtime/btime/winc/binc/movestogo`，不暴露未翻译的其他 time manager。
- 正式 x7 ONNX 不含 moves-left head；selection 因此严格走 px0 禁用 `MEvaluator()` 的 WDL-only
  分支，M utility 为零，参考 `src/search/classic/search.cc:60-114,1596,1680-1692`。
- 正式 UCI `WeightsFile` 生命周期：`src/engine.cc:137-197,206-219`，没有权重时明确拒绝搜索，
  不回退到 `UniformBackend`。
- NN 训练入口：参考 `pxzero-training/tf/train.py:110-126`、`tf/configs/example.yaml:4-31`，已收为
  单一 `dataset / model / training` YAML；当前不移植其 TensorFlow 兼容层或旧数据管道。
- P1-P3 进入 P4 前复核：`cargo test --release -p xiangqi_core`（22 项 px0 规则/history 对拍）、
  `cargo test --release -p engin --lib`（88 项 P2/P3/controller/tree/UCT/P4 回归）与真实 ONNX UCI
  生命周期/legacy clock 冒烟已通过。

## P4：单 worker 搜索流水线可用；task-worker 待重构

单 worker/minibatch/OOO/cache/stop 与真实 ONNX/DirectML 时序已有回归和 release 冒烟。GPU
task split 不可用：此前 Rust raw-pointer 版本会让两个 task 重复扩展同一未扩展节点，现已删除并统一退回
`task_workers_=0`，不能作为 px0 对齐完成项。

- [x] 翻译 px0 `src/neural/memcache.cc:38-190`、`memcache.h:34-45` 为正式 ONNX 的
  `CachingBackend` wrapper：当前局面 hash 为 key、合法着数量防碰撞、cache miss 仅在
  `ComputeBlocking` 后写入、`ucinewgame` 清 cache；`NNCacheSize` 默认/范围为
  `src/neural/shared_params.cc:63-82` 的 `2000000` / `0..999999999`。

- [x] 按 `src/search/classic/search.cc:981-1034` 翻译 watchdog 的 counters-mutex/condition-variable
  等待与 `FireStopInternal` 唤醒；不再固定 1ms polling。
- [x] 按 `src/search/classic/search.cc:249-264,393-398,908-918,1213-1231` 翻译
  `nps_start_time_` 的 watchdog 初始化、UCI nps/eps 与 NPS limit 时钟归属。
- [x] 按 `src/search/classic/search.h:368-369, search.cc:596-610,908-922,981-1017,1268-1284`
  将 `latest_time_manager_hints_` 收为 SearchWorker-local、watchdog-local 两份；不得跨 worker
  共享 remaining-playouts hint。
- [x] 按 `src/search/classic/search.cc:596-610` 加入 root-first-visit stopper gate；未扩展根节点
  不能被 budget stopper 提前结束。
- [x] 按 `src/search/classic/stoppers/factory.cc:44-115`、`legacy.cc:43-174`、
  `stoppers.cc:39-129`、`common.cc:118-165` 翻译 factory 默认 legacy time manager：
  `MoveOverheadMs`、`Slowmover` 与 `go wtime/btime/winc/binc/movestogo`；不暴露
  `simple/smooth/alphazero`。

- [x] 在 DirectML/ONNX 下完成固定 nodes、`go infinite -> stop -> wait`、`position ... moves ...`、
  backend reload 的 release UCI 冒烟。另验证 `go infinite -> go nodes` 与
  `go infinite -> position ... -> go nodes`：旧搜索静默回收，只有最后一次 `go` 输出 `bestmove`。
  对照 `src/engine.cc:148-224`、`src/search/classic/wrapper.cc:100-140`。

## 后续：P4 task-worker 的安全所有权翻译

已完成前置收口：`IterationState` 独占保存主 worker 的 minibatch/computation/out-of-order 计数，参考
`src/search/classic/search.h:419-427`；这不改变 `task_workers_=0`，也尚未把任何搜索数据交给 task。
`PickTaskQueue` 现以移动所有权而非克隆形式领取、完成并合并 task/result，参考
`src/search/classic/search.cc:1069-1140,1494-1508`；task workspace/tree 的所有权拆分仍未完成。
生产队列只保留同步 phase；`RunTasks` 的 sleep/exit 仅保留为测试状态机，参考
`src/search/classic/search.cc:1069-1124`，不能误作为后台 worker 已启用。
已完成 gathering 输入拆分：`PickNodesToExtendTask` 只读取 `SelectionContext`，参考
`src/search/classic/search.cc:1510-1550,1551-1897`；processing 的 minibatch range/tree 所有权仍待拆分。
已完成 `ExtendNode` 输入拆分：仅使用 `ExtendContext + tree + history`，参考
`src/search/classic/search.cc:1899-1974`；下一步仅允许继续收窄 processing 的明确 range。
已完成 processing range 拆分：`ProcessingContext + &mut [NodeToProcess]`，参考
`src/search/classic/search.cc:1423-1462,2109-2156`；尚未证明 tree 写入不重叠，`task_workers` 必须保持 0。
已完成 processing range 规划与不重叠回归，参考 `src/search/classic/search.cc:1322-1362`；下一步仅处理
gathering task 的 subtree/tree 写入所有权证明，不能以全树锁替代。
已完成 workspace 所有权映射：`TaskRunner` 独占其 `TaskWorkspace`，参考
`src/search/classic/search.h:348-365,441-445`；仍未创建后台 runner。
已完成 gathering handoff 的父边去重：仅 `PickTask` 发布成功后才从主 DFS 清零该 child 的 visit budget，
参考 `src/search/classic/search.cc:1828-1864`；尚未证明后台 task 的 subtree/tree 写入不重叠。
已完成 px0 `task_workspaces_[tid]` 的执行边界：`TaskRunner` 独占 workspace，gathering task 自有结果，
processing 只接收已拆分 minibatch slice，参考 `src/search/classic/search.cc:1116-1129,1322-1362`；当前仍是
main runner 的同步调用。
已核实 px0 task threads 在主 worker 持有 `nodes_mutex_` 的 gather phase 内直接修改 tree，且
`PickNodesToExtendTask` 标记为 `NO_THREAD_SAFETY_ANALYSIS`，参考
`src/search/classic/search.cc:1485-1508,1551-1897`。Rust 不能对普通 `NodeTree` 复制该可变别名；不得用
全树锁或 raw pointer 启用 task worker。
已核实 px0 `Node` 的统计、child sibling 链与 edge 排序均非 atomic，`Edge_Iterator` 允许 sibling 在调用
间隙插入但要求外部同步，参考 `src/search/classic/node.h:132-260,423-525`、`node.cc:245-373`。这使后台
task-worker 的 Rust 所有权/并发表示成为明确阻塞项：未形成按上述代码逐段证明的 scoped tree-phase
方案前，配置值只能影响 split，不能启动后台线程；不得以全树锁或未经审计的 raw pointer 启用。
已恢复 px0 `TaskWorkers` 构造期解析，参考 `src/search/classic/search.h:205-224`：显式值保留，`-1` 按
CPU/GPU 与硬件线程公式推导。当前仅据此做 split，`WaitForTasks` 仍同步 drain；后续 scoped task threads
必须满足 `AGENTS.md` 的 raw-pointer tree phase 与真实 ONNX 回归约束。
已完成 scoped task queue phase 协议，参考 `src/search/classic/search.cc:1069-1140,1485-1508`：主 DFS 可持续
发布 task，seal 后 worker 才在队列耗尽时退出。当前同步 drain 复用该协议；下一步替换为 scoped task threads。

- [ ] 先完成 px0 task state 的 Rust 所有权拆分，对照 `src/search/classic/search.h:348-445`、
  `search.cc:1069-1140,1423-1462,1485-1508`：task 独占 workspace/task/result，主 worker 独占
  minibatch/computation/counters；禁止共享 `&mut SearchWorker`。
- [ ] 重新设计 Rust task 的所有权边界，对照 px0 `src/search/classic/search.h:205-244,348-445`、
  `search.cc:1069-1508`：task 只能拥有独立 workspace 与明确不重叠的 node/minibatch range，不能共享
  `&mut SearchWorker` 或通过 raw pointer 直接修改整棵树。
- [ ] 在固定 visits、`go movetime`、真实 ONNX/DirectML 下补重复 ExtendNode、`NInFlight==0`、stop/wait
  回归；只有该回归稳定后才能恢复 px0 GPU `task_workers_` 默认解析。

## 约束

- 每个 Rust 函数的注释或本文件必须标注 px0 文件和连续行区间。
- 找不到 px0 对应参考时，记录缺口，不实现。
- `UniformBackend` 仅限单元测试和对拍，正式 UCI 永不使用。
