# TODO

> 只保留当前未完成事项。  
> 当前主线只做搜索。  
> 每个改动前，先补 `lc0` 参考文件、行号、本仓库落点。

## 已完成（勿再当待办）

### P4.1 MultiPV

- [x] UCI 公开 `MultiPV` + `setoption` 接线
- [x] 多条 `info` / top-k PV（`GetBestChildrenNoTemperature`）
- [x] 默认总 `nodes` 口径（**不含** `PerPVCounters`）

## 当前未完成 — 搜索主线

### P4.2 并发细节对齐 lc0（进行中）

- [x] `TaskWorkersPerSearchWorker` 解析（`task_workers.rs::resolve_task_workers`）
- [x] gather processing 区间拆分（`plan_processing_task_ranges` + `run_process_picked_phase`）
- [x] **Task worker 线程池** 并行 `ProcessPickedTask`（`search.cc:1091-1161,1382-1383` → `SearchWorkerTaskPool`）
- [ ] picking 阶段 task 拆分（`search.cc:1507+`，GPU 路径 `PickNodesToExtendTask`）

**processing task-workers 当前边界（未完成项，不是“小尾巴”）：**

- 只并行 processing；picking 仍是单线程 `PickNodesToExtend` 语义。
- task worker 路径仍依赖 **共享树 `Mutex`**，未对齐 lc0 完整 SearchWorker task-worker 主线。
- 共享 backend：`SharedBackendComputation::compute_blocking()` 消费后 **清空 slots**（lc0 每轮 `computation_.reset()`）。

### lc0 已知未完全对齐项（记录，非当前必修）

- [ ] **`go mate`**：当前看 best PV 的 `best_mate` 提前停；lc0 用 iteration `mate_depth`（`search.rs` / `worker.rs::budget_exhausted`）
- [ ] **`PerPVCounters`**：当前未实现；`MultiPV>1` 时每条线仍共用总 `nodes`

### P4.3 回归护栏（部分完成，待并发收口后加强）

- [x] MultiPV UCI / 引擎层回归
- [x] `go nodes / movetime / ponder / stop / searchmoves` 长链路冒烟
- [x] bench JSON `multi_pv` + `commands.md` 同步
- [ ] 多线程 / task workers 引入后的 integration 回归
- [ ] 固定 FEN 数值快照（待搜索稳定后收紧）

参考：

- `C:\Users\Administrator\projects\lc0\src\search\classic\search.cc:896-922,1319-1331,1353-1384,2217-2373`
- `C:\projects\77xiangqi_engine\crates\engin\src\mcts\search.rs`
- `C:\projects\77xiangqi_engine\crates\engin\src\mcts\worker.rs`

## 暂不做

- [ ] 当前不把重点切回模型
- [ ] 当前不把中残局 value 作为主攻方向
- [ ] 当前不参考 `px0` 改搜索结构
- [ ] 当前不吸收 `KataGo` 的 graph / DAG / cache 主线
- [ ] 当前不做新搜索路线
- [ ] `PerPVCounters`（除非明确要抄 lc0 该 UCI 选项）
