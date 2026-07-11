# NextStep

## 当前结论

当前主线只做搜索，并且只按 `lc0 classic` 抄。

**重要：搜索基建尚未完全对齐 lc0。** 当前只应把 **P4.1 MultiPV** 视为完成态；**并发细节（P4.2）仍未完成**，review 时不得误判为“搜索已 1:1 对齐 lc0”。

不再做的事情：

- 不先讨论模型
- 不先讨论中残局 value
- 不先参考 `px0` 改搜索结构
- 不先吸收 `KataGo` 的 graph / DAG / cache 主线

## 当前代码状态

当前 `engin` 已有：

1. `position / go / stop / ucinewgame / ponderhit`
2. `go wtime/btime/winc/binc/movestogo`
3. `searchmoves / mate / ponder`
4. `Threads=0 / MinibatchSize=0`
5. `shared-tree + batched eval + subtree reuse`
6. `policy + WDL` ONNX 主链路
7. **MultiPV** UCI + top-k PV + 多条 `info`（见 P4.1）
8. 基础回归护栏（MultiPV / 部分 UCI 长链路 / bench schema）

## P4 进度（分阶段，勿混为一谈）

### P4.1 MultiPV — 已完成

对齐 lc0 `SendUciInfo` **主输出形态**（不含 `PerPVCounters`）：

- UCI 公开 `MultiPV`（default 1，max 500）
- `GetBestChildrenNoTemperature` 等价：`worker.rs::get_best_children_no_temperature`
- 多条 `info`：共用 `depth/seldepth/time/nodes/nps`，独立 `score/multipv/pv`
- 默认每条 `multipv` 复用**总** `nodes`（与 lc0 默认、`PerPVCounters=false` 一致）
- **未做**：`PerPVCounters`（每条线独立 `nodes`）
- 参考：`lc0 search.cc:261-373,727-824`；`uciloop.cc:305-329`

### P4.2 并发对齐 lc0 — 进行中

| 项 | 状态 | lc0 参考 | 本仓库 |
|----|------|----------|--------|
| processing 区间拆分 | **已接线** | `search.cc:1353-1378` | `task_workers.rs::plan_processing_task_ranges` + `worker.rs::run_process_picked_phase` |
| `TaskWorkersPerSearchWorker` 解析 | **已接线** | `search.h:216-226` | `task_workers.rs::resolve_task_workers` |
| **Task worker 线程池** | **已接入（processing 子集）** | `search.cc:1091-1161,1382-1383` | `task_workers.rs::SearchWorkerTaskPool` + `RunTasks` / `WaitForTasks` |
| picking 阶段 task 拆分 | **未接入** | `search.cc:1507+` | 尚无 `PickNodesToExtendTask` 并行 picking |

**当前 processing task-workers 约束（勿误判为 lc0 SearchWorker 已对齐）：**

- 仅并行 **`ProcessPickedTask`**（processing 区间）；**未**并行 **`PickNodesToExtendTask`**（picking 仍单线程顺序走）。
- 并行 processing 仍持 **`Arc<Mutex<MctsTree>>` 大锁** 做树更新，不是 lc0 完整 task-worker 树访问模型。
- 共享 `computation_` 等价物 `SharedBackendComputation` 在 **`compute_blocking()` 后清空 slots**（一轮一 computation；对照 `search.cc:1255-1263` `InitializeIteration` reset）。

### P4.3 回归护栏 — 部分完成

已完成（围绕 MultiPV 与现有 UCI 主路径）：

- `tests/mcts_engine.rs`：MultiPV 引擎层、长链路、固定 FEN 范围断言
- `tests/p3_integration.rs`：`mcts_config.multi_pv` schema
- `commands.md`：`MultiPV` 说明与示例

未完成 / 待加强：

- 并发路径（多线程 + task workers 引入后）的 integration 回归
- 更紧的固定 FEN 数值快照（待模型与搜索稳定后再收紧）

## 下一步（固定顺序）

1. **picking 阶段 task 拆分**：`PickNodesToExtendTask`（`search.cc:1507+`）
2. **`go mate`** 改为 lc0 `mate_depth` 迭代 stopper 口径
3. 并发收口后再扩 P4.3 多线程回归
4. 模型训练保持可用即可，非当前搜索主线

## review 标准

后续 review 只看这几件事：

1. 是否一比一对齐 `lc0`（**含已知未对齐项清单**）
2. 是否又出现“名字像 lc0、行为不是 lc0”
3. 是否引入了不必要的新抽象
4. 是否破坏 `history / tree reuse / UCI` 语义闭合
5. `bench / UCI / GUI` 表现是否一致
