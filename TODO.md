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

## P4：px0 classic task-worker 已启用

单 worker/minibatch/OOO/cache/stop 与 GPU task split 均已在真实 ONNX/DirectML 回归。`TaskWorkers`
按 px0 解析并实际启用；`NodeArena` allocation lock、first-extend CAS、child-slot reservation 共同保证
task phase 的最小并发边界。历史上的 duplicate `ExtendNode` 门控已解除。

- [x] 翻译 px0 `src/neural/memcache.cc:38-190`、`memcache.h:34-45` 为正式 ONNX 的
  `CachingBackend` wrapper：当前局面 hash 为 key、合法着数量防碰撞、cache miss 仅在
  `ComputeBlocking` 后写入、`ucinewgame` 清 cache；`NNCacheSize` 默认/范围为
  `src/neural/shared_params.cc:63-82` 的 `2000000` / `0..999999999`。通用存储已替换为
  `quick_cache` 分片 S3-FIFO，缓存 value 为 `Arc<EvalResult>`；这不是 px0 严格 FIFO 的逐项翻译，
  但不改变 key/collision guard/completed-only 回填语义。

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
已接入 scoped task threads：每线程独占 `TaskRunner`，task/result 留在 queue；tree 与不重叠 processing range
只在 scope 内以审计过的 raw pointer 访问，参考
`src/search/classic/search.cc:1069-1140,1322-1362,1485-1508`。当前在 producer seal 后才启动 scope，尚未
实现 px0 主 DFS 与 task 的持续并行消费；下一步需固定 ONNX 回归后将 spawn 前移。
已将 gathering scope 前移到主 `PickNodesToExtendTask` producer phase，参考
`src/search/classic/search.cc:1069-1140,1485-1508,1828-1864`：task runners 等待 queue，主 DFS 发布子树，
seal/join 后合并 results。
已将 processing scope 前移到 split 点，参考 `src/search/classic/search.cc:1322-1362,1423-1462`：后台 runners
消费不重叠 range，同时 main runner 处理最终 suffix，seal/join 后继续 OOO/backup。P4 pipeline 的结构翻译
至此收口；下一步只补真实 ONNX/DirectML 固定 visits、movetime、stop/wait、`NInFlight==0` 回归，不再扩结构。

已将 px0 gathering subtree 的隐含互斥收为 Rust queue 不变量，参考
`src/search/classic/search.cc:1828-1864`、`src/search/classic/node.h:234-239`：同级 sibling task root
可以共存，但 duplicate/ancestor/descendant root 必须拒绝，且 handoff 不得清零父 DFS budget。该回归只证明
task split，不让 `Vec<Box<Node>>` 或 `Node` 的非原子统计变成可并发访问。

已将 `Node::TryStartScoreUpdate` 的 first-extend gate 收为原子 CAS，参考
`src/search/classic/node.cc:346-365`；并发回归保证一个未扩展 node 只有一个 in-flight winner。WL/D/M
仍保持 px0 的 `WaitForTasks` 后主 worker backup 时序；`NodeArena` child allocation 尚未并发安全。

已将 edge child slot 改为 atomic `empty -> reserved -> published-index`，参考
`src/search/classic/node.h:468-525`；并发回归保证同一 edge 只会有一个 reservation winner。此项只解决
parent slot 重复创建，`NodeArena(Vec<Box<Node>>)` 的并发 allocation 仍是下一阻塞点。

- [x] 完成 px0 task state 的 Rust 所有权拆分：task 独占 workspace/task/result，主 worker 独占
  minibatch/computation/counters；禁止共享 `&mut SearchWorker`。参考
  `src/search/classic/search.h:348-445`、`search.cc:1069-1140,1423-1462,1485-1508`。
- [x] 建立 task 的 Rust 并发边界：gathering root claim、非重叠 minibatch range、稳定 arena allocation 与
  atomic node gate/child slot；不使用整树锁串行化。参考 `src/search/classic/search.h:205-244,348-445`、
  `search.cc:1069-1508`、`node.cc:346-365`。
- [x] 在固定 nodes、`go movetime`、真实 ONNX/DirectML 下完成 duplicate ExtendNode、stop/wait 与
  `NInFlight==0` 回归；已解除 `active_task_workers=0`。

## 约束

- 每个 Rust 函数的注释或本文件必须标注 px0 文件和连续行区间。
- 找不到 px0 对应参考时，记录缺口，不实现。
- `UniformBackend` 仅限单元测试和对拍，正式 UCI 永不使用。
