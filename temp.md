# P4.2-D 树锁粒度审计（2026-07-11）

对照 lc0 `search.cc:1290-1439,2200-2373`；本仓库 `Arc<Mutex<MctsTree>>` 并行路径。

## 当前持锁阶段

| 阶段 | 单线程 | 并行 search worker | task worker |
|------|--------|-------------------|-------------|
| **picking** (`PickNodesToExtendTask`) | `&mut MctsTree` 全程 | 主：短锁；子 task：`Mutex` 独立加锁 | `PickingDispatch.tree` 加锁 |
| **processing** (`ProcessPickedTask`) | `&mut` 全程 | 主：短锁跑 main range；worker：`Mutex` | `ProcessingDispatch.tree` 加锁 |
| **fetch** (`FetchMinibatchResults`) | gather 外单线程持树 | iteration 后 `shared_tree.lock()` | 无（主/worker 路径外） |
| **backup** (`DoBackupUpdate`) | 同上 | iteration 后同一把 `Mutex` | 无 |
| **OOO 撤销/backup** | gather 内 `&mut` | gather 内 `TreeGatherAccess::with_tree` | 无 |

## 与 lc0 差异

- lc0：主线程在 `PickNodesToExtend` 持 `nodes_mutex_`，task worker **不再加锁**（NO_THREAD_SAFETY_ANALYSIS）。
- 本仓库：task worker **各自 `Mutex` 加锁**，主线程在 `WaitForTasks` 前**必须释放**树锁（已在 `pick_nodes_to_extend_parallel` / processing wait 路径落实）。

## 便利实现 vs 语义必需

| 锁区间 | 判定 |
|--------|------|
| 整个 gather 单把 `tree_guard`（旧并行路径） | **便利实现**；已改为 gather 内分段 lock/unlock |
| processing 时 `Arc<Mutex<SearchIteration>>` | **便利实现**；lc0 用 thread-local `TaskWorkspace` + 主 `minibatch_` |
| fetch/backup 与 gather 分离加锁 | **语义必需**（NN 在锁外） |
| 共享树 `Mutex` 作为唯一写入口 | **当前必需**（尚无 lc0 级 node lock） |

## 结论

- **暂不进一步细粒度改锁**；下一步优先 picking task 语义稳定性与 `go mate` stopper，而非 node-level lock。
- 若后续对齐 lc0「主线程持锁、worker 无锁 picking」，需引入 **显式树锁托管**（类似 lc0 `SharedMutex` + 主线程 guarantee），不能只在 worker 侧去掉 `Mutex`。
