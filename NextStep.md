# NextStep

## 当前基线

唯一工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

P0-P3 已完成。P4 的 ONNX、minibatch、prefetch、collision、OOO、MemCache、tree reuse、watchdog、
legacy time manager 与 UCI 生命周期已接入。
Windows 发行构建只启用 DirectML，失败时明确回退 CPU。

## 当前搜索切换：classic 停在语义基线，重建 stream

classic 的 P4 T3 要求将 px0 `RunTasks()` 的共享 mutable `NodeTree` 翻译成 Rust；这与 Rust 的安全所有权
模型冲突，snapshot/replay 原型也已删除，不能继续在 classic 上打补丁。classic 只保留为当前 UCI 行为基线。

新模块是 `crates/engin/src/search/stream`。它按 LC3 的公开架构文档实现 streaming tree search：

- [Overview](https://lczero.org/dev/lc0/search/lc3/overview/)
- [Policy](https://lczero.org/dev/lc0/search/lc3/policy/)
- [Glossary](https://lczero.org/dev/lc0/search/lc3/glossary/)

本地 lc0 master 没有 LC3 源码，故这是架构实现，不是 1:1 源码翻译。第一版固定为 tree repository，禁止
提前加入 DAG/TT、KataGo graph、额外启发式或切换 UCI。

### S0：完成的基础层

- 独立 `NodeRepository`：分片 map，只锁查找/插入；节点/边统计不持有 whole-tree lock。
- `NodeKey = HashConcatenate(parent_key, move)`，明确是 tree key。
- `StreamEdge`：`started` 含 in-flight，completed Q 聚合局部锁保护；`EdgeReservation` 只能移动完成或取消。
- `NodeEvent`：拥有 generation、root history、variation 和 reservation；不得 clone reservation。

### S1：Gather -> Eval -> Backprop 串行语义

1. 已定义 LC3 policy 子集：edge selection、collision、`ValueDelta`、edge/node update；参考 LC3 Policy 的
   `GetNumEdgesToFetch`、`DistributeVisits`、`NodeEventToValueDelta`、`MergeNodeUpdates` 标题。
2. 已在单线程上实现 owned event 的 Gather、合法着/Eval、terminal 和 Backprop。失败 Eval 会撤销 reservation 并
   将 node 从 `Evaluating` 恢复为 `Unexpanded`。
3. 下一步为固定 visits 结构对拍：root edge `N/Q/P`、PV、bestmove，并补 terminal、two-fold、rule60、stop。

门槛：`NInFlight==0` 等价物为所有 edge `started == completed`；不接 UCI。

### S2：有界流水线与 NN

1. 已加入 bounded Gather/Eval/Backprop queues，generation gate 和 stop/drain；队列只传 owned event/result。
   当前为**协作式单 controller**，目的是先锁定队列背压与生命周期语义。
2. Eval 侧已接现有 `BackendComputation`：cache hit 直接取结果，miss 以 `eval_batch_size` 聚合并单次
   `compute_blocking()`；stream stats 记录实际 NN batch/evaluation 数。
3. S2b 已将 queue stage 移至常驻 worker：多个 Gather、一个 batch Eval、多个 Backprop；保持 event 所有权、
   generation gate 和 stop/join，不引入 classic shared mutable tree。已覆盖 worker fixed-playout 与
   根扩展后立即 stop/join 的 reservation drain。
4. 下一项是固定-visits 结构对拍和真实 ONNX 长 `movetime` 回归；Watchdog/UCI 只在两者通过后切换至 stream。

门槛：真实 ONNX `go movetime 30s` 持续推进；`position -> go -> stop -> position -> go` 无旧 generation
更新、无 reservation 泄漏、恰好一次 bestmove。

## NN：x7 v2 固定 trunk

训练模型已收敛为单一 x7 v2 架构，旧 `katago_gpool_value_aux_v1` checkpoint 不兼容且训练/导出入口会明确拒绝：

1. `3x3 stem: 124 -> 256`。
2. 12 个 pre-activation bottleneck：`BN/SiLU -> 1x1 256->112 -> 3x3 112->112 -> 3x3 112->112 -> 1x1 112->256 + identity`。
3. block 4、8 后各一次独立 Global Broadcast：`BN/SiLU -> 3x3 256->128 -> mean/max -> Linear(256->256) -> x + bias`。
4. policy 保留完整 `52x10x9 -> 2062` 空间输出；WDL 与 raw moves-left 共享 global readout，再分出 `WDL[3]` 和非负 `moves_left[1]`。

`width=256`、`blocks=12`、`bottleneck_channels=112` 是当前正式基准，参数为 `5,690,808`，FP32 权重约 `21.7 MiB`。v2 结构族允许
用 YAML 实验其他正偶数 width、不少于 3 个 blocks 与独立 bottleneck 中间宽度；两次 Global Broadcast 始终分布在三个 trunk stage
之间。改变 width/blocks/bottleneck_channels 后必须使用新 checkpoint，续训与 `init_from` 必须精确匹配 checkpoint 元数据；
项目版本仍为 v2。

`ContemptMode=play` 的无限搜索按 px0 `Search::Search`
(`src/search/classic/search.cc:156-170`) 在启动 worker 前降为 `none`；仅当 WDL rescale diff 非零时输出
对应 `info string` warning。旧 bridge 遗留的 `outputs` / `search_active` 状态已删除，不再作为搜索真相。
watchdog 也会在 stopper 第一次令搜索停止后继续执行 px0 `MaybeOutputInfo`
(`src/search/classic/search.cc:351-389,981-1017`)；不允许响应 `bestmove` 的无限搜索会得到无进展 warning。

Rust 的 `ClassicSearch` 跨多条 UCI `go` 复用，而 px0 每条 `go` 都销毁其 `Search`。因此 Rust 在所有
search worker join 后显式执行 px0 `Search::CancelSharedCollisions`，确保上一搜索未 backup 的 collision
virtual visit 不会带入下一次 `go`。参考 `src/search/classic/search.cc:1027-1064`。

## 当前优先级：先收敛主线语义

在继续处理 task-worker 的并发优化前，先逐项验收 px0 classic 的引擎语义、UCI 生命周期与搜索逻辑。
**不改变 px0 的 `SearchWorkers` 或 `TaskWorkers` 默认行为**：搜索 worker 数仍按 px0
`Search::StartThreads` (`src/search/classic/search.cc:874-897`) 由 backend 属性派生，`TaskWorkers=-1`
仍按 `src/search/classic/params.cc:478-481,622` 自动推导。

当前验收顺序：

1. UCI/controller：bare `go`、`position`/`go` 替换、`stop -> wait`、`ucinewgame`、有限搜索恰好一次
   `bestmove`。参考 px0 `src/engine.cc:187-235`、`src/chess/uciloop.cc:45-337`、
   `src/search/classic/wrapper.cc:100-152`。
2. Tree/search：root reuse、two-fold/rule60、terminal sticky evaluation、selection、NN 结果符号、backup、
   best-child/PV。参考 px0 `src/search/classic/node.cc:245-390,465-520` 与
   `src/search/classic/search.cc:705-808,1423-1508,1510-1974,2109-2235`。
3. 固定 FEN + 固定 visits 的结构对拍：记录 root child `N/WL/D/P`、terminal edge、PV、bestmove 与
   `NInFlight==0`；使用 UniformBackend 做确定性单测，真实 ONNX 只验 UCI 生命周期和不崩溃。

本轮语义门槛已完成：已逐段复核 controller/UCI、stopper、root reuse、two-fold/rule60、terminal、selection、
NN result、backup、best child/PV；真实 `x7.onnx` 已通过 `go nodes` 与
`go infinite -> stop -> wait -> position -> go nodes` 冒烟。当前未翻译的 `ponder`、`go depth/mate` 和额外
px0 可选项仍明确拒绝或未暴露，不能冒充为已支持功能；它们不改变默认 `nodes/movetime/infinite` 搜索主线。

已确认并修复：Rust terminal best-child 枚举排序曾与 px0
`src/search/classic/search.cc:705-808` 相反，可能使已证明的终局败着压过胜着；现已补回归。

已收口：two-fold correction 现在会按 px0 `search.cc:1510-1550` 同时扣除显示基线
`initial_visits_`；Rust 以 `WorkerSearchState::reverted_initial_visits` 记录 worker 侧原子增量，
stats 读取时从本次 root reuse 基线扣除，并有 repeated-tree 回归。

已收口：UCI 棋钟现在按 px0 `src/engine.cc:187-219` 在 `position` 启动，只有没有
`wtime/btime` 的 `go` 重置；带时钟预算保留 `position -> go` 的计时起点。

已收口：无合法着叶子现在按 px0 `SearchWorker::ExtendNode`
(`src/search/classic/search.cc:1913-1919`) 写入父边局部视角的 `WHITE_WON`，由 backup 翻转符号；不再按
绝对红黑方写 terminal WDL。`TimeLimitStopper` 也已按
`src/search/classic/stoppers/stoppers.cc:120-129` 只检查 elapsed time，因此复用树的 `go movetime 0`
不会被错误强制执行一次新 playout。

已收口：根无合法着仍按 px0 `search.cc:612-621`、`chess/uciloop.cc:279-287` 输出 null move
`bestmove a0a0`；Rust 不再因空 PV/info 把 UCI 完成响应吞掉。

已收口：stopper chain 现在按 px0 `stoppers/common.cc:133-145` 始终安装
`VisitsStopper(4_000_000_000)`；`go nodes` 缺省时也有与 px0 相同的树规模硬上限。
并且按 `chess/uciloop.cc:230-237` 保留原始 `go nodes` 数值，按
`stoppers/common.cc:123,147-151` 让 `go infinite movetime N` 忽略 `movetime`。

已收口：NN result policy 回填按 px0 `search.cc:2145-2153` 无条件写入 edge prior 后排序；删除了
无 px0 对应的 `node.N==0` 跳过分支，避免在 OOO/tree reuse 边界静默保留旧 policy 次序。

已收口：two-fold terminal 被 tree reuse 重开时，`Node::MakeNotTerminal` 现在按 px0
`node.cc:319-341` 保留当前 WDL 作为基础 visit 后再合并 child；不再把该 base WDL 预先清零。

已收口：best-child 的 `N == 0` child 现在按 px0 `EdgeAndNode::GetQ`
(`src/search/classic/node.h:375-377`) 返回传入默认 Q=0，而不是读取尚未完成 visit 的 placeholder WDL；
这保证非零 `DrawScore` 下的 PV/bestmove 仍按 visits、prior 的既定顺序排序。对应
`Search::GetBestChildrenNoTemperature` 的连续参考为 `search.cc:705-808`。

### 本轮主线审计证据

- 固定 visits 与合法 bestmove：`crates/engin/tests/search_trace_test.rs` 的
  `fixed_nodes_*`；验证 px0 `VisitsStopper` 的 batch 后停止语义，而非错误要求 exact nodes。
- 终局与 best-child：`search/classic/worker.rs` 的 no-legal-move、sticky terminal、two-fold 回归，以及
  `search/classic/search.rs` 的 terminal rank、`N == 0` Q-proxy 回归；连续参考为
  `search.cc:705-808,1913-1919,2175-2257`。
- tree reuse：`node.rs::reset_to_new_line_clears_unmatched_edges`、two-fold reopen 与
  `search.rs::reused_root_movetime_zero_returns_a_bestmove`；参考 `node.cc:465-520`、
  `search.cc:1510-1550`。
- stopper/UCI：`uci_search_test.rs` 与 `p4_async_search_test.rs` 覆盖 `nodes=0`、movetime、
  infinite/stop/wait、position replacement、searchmoves 和 exactly-one bestmove；参考
  `uciloop.cc:197-245`、`engine.cc:187-235`、`stoppers/common.cc:118-165`。
- 真实 ONNX：`local_x7_runs_mcts_with_cnn_if_present` 以及 release UCI transcript 只验证
  `WeightsFile -> position -> go/wait/stop` 主链不崩溃；棋力、PV 长度和 NPS 不属于本轮逻辑验收。

## P4 长任务：px0 TaskWorkers 所有权与生命周期翻译

目标不是先提升 NPS，也不是吸收 KataGo 的 NN queue 或图搜索；目标是让 `TaskWorkers > 0` 成为 px0
语义下真正运行的 task 线程，同时保留 Rust 的安全所有权边界。当前 `PickTaskQueue`、`TaskRunner` 与
owned processing input 已存在，但 `run_queued_tasks_in_tree()` 仍由 owner 同步 drain，故不得标记为已完成。

唯一连续参考：

- `px0/src/search/classic/search.h:205-244,348-445`
- `px0/src/search/classic/search.cc:1069-1150,1322-1515,1551-1905,2104-2257`

### T1：常驻 task 生命周期与无任务停机

1. 逐段翻译 `SearchWorker` 构造/析构的 `task_threads_`、`task_workspaces_`、`task_count_=-1`、
   `task_added_` 唤醒与 join（`search.h:205-244`）。
2. Rust task 线程只能持有 `Arc<TaskPhase>`、自己的 `TaskWorkspace` 和 owned result sender；不得持有
   `&mut SearchWorker`、`NodeTree`、minibatch、backend computation 或 raw pointer。
3. 严格翻译 `RunTasks` 的 take/idle/close 状态：析构或 abort 后每个线程必然退出；首次 `go`、连续
   `position -> go`、`stop -> wait` 均不得遗留线程或任务。

门槛：`TaskWorkers=0` 与 `TaskWorkers=1` 的 uniform fixed-visits trace 都可结束；`go infinite -> stop`
在 1 秒内 join，且无 `NInFlight` 泄漏。

### T2：processing task 真并行，owner 独占 commit

1. 翻译 processing split 的阈值、范围切分和 main-worker 尾段（`search.cc:1322-1362`）。
2. 后台 task 只运行 `ProcessPickedTask` 的 owned 规则/合法着/history 计算，返回 `ExtensionResult`；
   不直接写 node、edge、cache、backend computation 或 minibatch。
3. `WaitForTasks` 严格等待完成数量，再由 owner 单次合并 result、提交 backend、fetch 与 backup；
   连续参考为 `search.cc:1423-1462,1475-1508,2104-2257`。

门槛：真实 `x7.onnx`/DirectML 的 `go movetime 30s` 持续推进；固定 visits 下 `TaskWorkers=0/1` 的
root child `N/WDL/P`、bestmove 与 completed playout 语义一致；任何 task error 必须中断搜索并被 UCI 报告。

### T3：owned gathering delta，最后解除 gathering 门控

已验证的 px0 边界：`PickNodesToExtend()` 由主 worker 持有 `nodes_mutex_` 到 `WaitForTasks()` 返回，
而后台 `RunTasks()` 直接读写同一 `NodeTree`（`search.cc:1069-1140,1494-1508`）。C++ 以外部锁持有
约定表达这层别名；Rust 不能在不使用 raw pointer/unsafe 的前提下，将同一个 `&mut NodeTree` 同时交给
owner 与常驻线程。因此这里不能逐字照搬，也不能把 task 再套进整树锁后称为并行。

1. 先设计并测试 `PickNodesToExtendTask` 的 owned selection delta：task 返回 path、collision、
   multivisit 和新 leaf 描述，owner 才在 tree phase 提交变化。
2. 只有该 delta 能保留 `TryStartScoreUpdate` / `NInFlight` 的唯一扩展语义时，翻译 px0 gathering split
   （`search.cc:1485-1508,1551-1897`）。
3. 不能证明等价时，保持 owner gathering；不得用整树锁、raw pointer 或“已有 child 就跳过”伪造并发。

实现前置决策：必须先明确接受“owner 对 task delta 做 CAS/版本校验后重放”的 Rust 等价模型；它会保持
`TryStartScoreUpdate`/`NInFlight` 语义，但不是 px0 的字面共享树写入。若不接受该等价模型，T3 必须保持
同步 gathering，P4 的完成范围只包括 T1/T2 processing。

门槛：fixed visits、长 movetime、stop/wait、tree reuse、two-fold/rule60、root terminal 都通过；每次
搜索结束和下一条 `go` 前全树 `NInFlight==0`。完成后才允许宣称 P4 TaskWorkers 打通。

### 非目标

- 不吸收 KataGo 的 graph/DAG、central NN server 或 node-level lock 设计。
- 不修改 `SearchWorkers` / `TaskWorkers=-1` 的 px0 默认派生规则。
- 不在本长任务调 cpuct、batch、网络或 time manager。

## P4 并发边界（实施约束）

`TaskWorkers` 按 px0 `src/search/classic/search.h:205-224` 解析。当前 queue/owned task 边界已存在，
但 task 仍由 owner 同步 drain；T1-T3 是将其改为真正常驻 processing/gathering task worker 的唯一计划。

px0 的 `RunTasks` 让后台 task 在主 worker 持有 `nodes_mutex_` 的同一 tree phase 中直接访问普通
`Node`/`NodeTree`，见：

- `src/search/classic/search.h:205-244,348-445`
- `src/search/classic/search.cc:1069-1140,1322-1362,1423-1508,1551-1897`
- `src/search/classic/node.cc:245-373`

Rust 当前的 `NodeTree` 需要独占 `&mut` 借用，且 `Node` 的 WDL、terminal、bounds、children 和 edge
不是可并发字段。processing task 因此只消费 owned leaf path/root history，以私有 `TaskWorkspace` 计算
`ExtendNode` 的 rule/move result；owner 在 `WaitForTasks` 后独占写 node、提交 backend input、OOO 和 backup。
此前 raw-pointer scoped bridge 在真实 ONNX 的长时间搜索中重复扩展节点并停顿，已删除。

禁止：

- 恢复 raw pointer、`unsafe impl Send` 或共享 `&mut SearchWorker`
- 用整树锁把后台 task 串行化后宣称已经并行
- 仅凭 `n/n_in_flight` 原子化或 child slot 唯一创建解除门控

## 验收命令

```powershell
cargo fmt --check
cargo test -p engin --lib
cargo clippy -p engin --all-targets -- -D warnings
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1
```
