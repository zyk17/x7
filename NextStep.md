# NextStep

## 当前结论

当前主线只做搜索，并且按 `lc0 classic` 抄。

**搜索基建：功能已接入，并发回归尚未完全收口。** 本轮已修 `Threads=2` 提前 stop 导致 `bestmove (none)` 的路径；`MaybeTriggerStop` / `PerPVCounters` / UCI 口径仍待证明 1:1。

不再做的事情：

- 不先讨论模型
- 不先讨论中残局 value
- 不先参考 `px0` 改搜索结构
- 不先吸收 `KataGo` 的 graph / DAG / cache 主线

## lc0 classic 对齐清单（搜索基建）

| # | 项 | lc0 参考 | 本仓库落点 | 状态 |
|---|-----|----------|------------|------|
| 1 | task worker 收口 | `search.h:229,408,419` | `task_workers.rs` | 功能已接入 |
| 2 | `PopulateCommonIterationStats` | `search.cc:930-1001` | `iteration_stats.rs` | 集中层已建；字段/口径待对照 |
| 3 | `MaybeTriggerStop` | `search.cc:617-646,1009` | `iteration_stats.rs` | budget 链已收口；非完整 lc0（无 FireStop/OnSearchDone/SmartPruning） |
| 4 | `PerPVCounters` | `params.cc:367` | `config.rs` / `uci.rs` | option + 输出已接；同局面日志未核 |
| 5 | `RunBlocking` / worker 生命周期 | `search.cc:911-921` | `search.rs` | 单线程 do-while；并行 Watchdog |
| 6 | UCI info 统计口径 | `search.cc:930,1008,2375` | `uci.rs` | 冒烟有；lc0 同局面日志待核 |
| 7 | tree reuse / position moves | `node.cc:493-519` | `engine.rs` / `tree.rs` | 功能已接入；长链 UCI 待补 |
| 8 | **并发回归** | — | `search.rs` / `uci.rs` | **进行中**：stop 清 in-flight；Threads=2 全量测试 |

## 固定执行顺序

1. **并发回归通过**（`cargo test -p engin` 含 `threads_option_searches_without_hanging`）
2. UCI info / PerPVCounters 同局面 lc0 日志对照
3. `MaybeTriggerStop` 补 lc0 FireStop / time hints / SmartPruning 链
4. GPU + task_workers integration

## review 标准

1. 是否一比一对照 `lc0`（含上表「待核/子集」项）
2. 是否又出现“名字像 lc0、行为不是 lc0”
3. **`cargo test -p engin` 全绿**（含 Threads=2）
4. 是否破坏 `history / tree reuse / UCI` 语义闭合
