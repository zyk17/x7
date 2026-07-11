# TODO

> 只保留当前未完成事项。  
> 当前主线只做搜索。

## 已完成（功能接入，非 1:1 证明）

- [x] task worker picking/processing 拆分 + 死字段清理
- [x] `iteration_stats.rs` 集中 stats/stop 入口（**子集**，非完整 lc0）
- [x] `PerPVCounters` UCI option + 按 PV 输出 `nodes`（**未同局面核对**）
- [x] tree reuse：`ResetToPosition` / `TrimTreeAtHead` / `expand_root_at`
- [x] 单线程 `RunBlocking` do-while 形状

## 当前阻塞 — 必须先过

### 1. 并发回归

- [x] 并行 stop 路径 `clear_in_flight_in_tree`，避免 `ensure_tree_quiescent` → `bestmove (none)`
- [x] `threads_option_searches_without_hanging` 改为 `join_search`（不 sleep+stop）
- [x] `parallel_threads_two_completes_nodes_budget` 引擎级回归
- [ ] **`cargo test -p engin` 全套件稳定全绿**（含满负载）

落点：`search.rs`、`worker.rs`、`uci.rs`、`engine.rs`

## 当前未完成 — 对齐证明

### 2. `MaybeTriggerStop` 完整 lc0 语义

- [ ] FireStop / bestmove 发送 / `OnSearchDone` / time manager hints
- [ ] SmartPruningStopper / KldGainStopper

参考：`search.cc:617-646,1009`；`stoppers.cc`  
落点：`iteration_stats.rs`、`stoppers.rs`

### 3. UCI info + `PerPVCounters` 口径

- [ ] 固定 FEN 与 lc0 日志逐项对照 `depth/seldepth/nodes/nps/pv`

参考：`search.cc:930,1008,2375`；`params.cc:367`  
落点：`uci.rs`、`worker.rs`

### 4. 回归护栏

- [ ] GPU + task_workers integration

## 暂不做

- `KataGo` 路线
- 新 stop/stats/tree worker 抽象
- 模型训练主线
