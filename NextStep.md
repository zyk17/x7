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

### P4.2 并发对齐 lc0 — 未完成

已审计、部分主路径一致，但 **整体不能标完成**：

| 项 | 状态 | lc0 参考 | 本仓库 |
|----|------|----------|--------|
| Watchdog + N workers | 一致 | `search.cc:896-922` | `search.rs::run_parallel_with_progress` |
| `Threads=0` → backend suggested | 一致 | backend attrs | `policy_onnx.rs::resolved_search_threads` |
| gather backend idle 早退 | 一致 | `search.cc:1319-1331` | `worker.rs::should_break_gather_for_thread_idling` |
| `backend_waiting` 计数 | 一致 | `search.cc:1328-1329` | `search.rs:507-527` |
| collision / OOO backup | 主路径一致 | `search.cc:1392-1421,2217-2373` | `worker.rs::apply_out_of_order_backups`, `do_backup_update` |
| **Task workers** picking/processing | **缺失** | `search.cc:1353-1384` | config 有 `minimum_work_*` 默认值，**未接 task 队列** |
| `go mate` stop 口径 | **未完全对齐** | iteration `mate_depth` | 看当前 best PV 的 `best_mate`（`search.rs` / `worker.rs::budget_exhausted`） |

**下一批搜索抄写应从这里继续**，而不是回到模型。

### P4.3 回归护栏 — 部分完成

已完成（围绕 MultiPV 与现有 UCI 主路径）：

- `tests/mcts_engine.rs`：MultiPV 引擎层、长链路、固定 FEN 范围断言
- `tests/p3_integration.rs`：`mcts_config.multi_pv` schema
- `commands.md`：`MultiPV` 说明与示例

未完成 / 待加强：

- 并发路径（多线程 + task workers 引入后）的 integration 回归
- 更紧的固定 FEN 数值快照（待模型与搜索稳定后再收紧）

## 下一步（固定顺序）

1. **继续 P4.2**：task workers picking/processing（`search.cc:1353-1384`）
2. **`go mate`** 改为 lc0 `mate_depth` 迭代 stopper 口径（`search.cc` PopulateCommonIterationStats / MateStopper）
3. 并发收口后再扩 P4.3 多线程回归
4. 模型训练保持可用即可，非当前搜索主线

## review 标准

后续 review 只看这几件事：

1. 是否一比一对齐 `lc0`（**含已知未对齐项清单**）
2. 是否又出现“名字像 lc0、行为不是 lc0”
3. 是否引入了不必要的新抽象
4. 是否破坏 `history / tree reuse / UCI` 语义闭合
5. `bench / UCI / GUI` 表现是否一致
