# TODO

## P0 已完成

- [x] 收敛项目边界，只保留 `MCTS + policy/value`
- [x] 清掉 Alpha-Beta 主线
- [x] 打通最小 `UCI -> MCTS -> ONNX` 链路
- [x] 决定主训练数据路线切到 `px0 / lc0`
- [x] 固定当前主模型 I/O：`124x10x9 -> 2062 + WDL`
- [x] 把 value 语义切到 `WDL + qMix`
- [x] 去掉旧 `attention policy`，切到纯 CNN 主线
- [x] 把 `engin` 输入切到真实 history 主线
- [x] 保留孤立 `FEN -> fen_only` fallback
- [x] 打通最小 `shared-tree` 多 worker 搜索

## P1 已完成：搜索对齐 `px0 classic` 主干

### P1.1 对照

- [x] 列出 px0 / engin 主链路结构
- [x] 一一对应表：入口、worker、batch、backend、backup、stop、reuse
- [x] 标出 px0 有而我们没有的残余（见对照表 P2 栏）
- [x] 删除 engin 自定义启发式（`expanding` / spread / EvalCoordinator 等）

### P1.2 搜索驱动

- [x] `execute_one_iteration` 七步
- [x] `gather_minibatch` + collision multivisit
- [x] backend 喂数与 batch 形成
- [x] progress / info 输出（独立 Watchdog 读锁 + worker 短写锁）

### P1.3 树与边统计

- [x] `visits / in_flight / wl/d/m`
- [x] PUCT + `GetChildrenVisits` + `draw_score`
- [x] collision multivisit + `FinalizeScoreUpdate` / cancel
- [x] `CancelSharedCollisions` + `CalculateCollisionsLeft`
- [x] `pending_searchers` + `RwLock` 细粒度树锁（非整树 Mutex）
- [x] smart pruning（根节点）

### P1.4 对外统计

- [x] `go nodes` = playouts + initial_visits
- [x] `go movetime` / `go infinite` + `stop`
- [x] `nodes / nps / pv / depth / seldepth`
- [x] benchmark 同口径（threads / pv / moves 解析）
- [x] UCI 仅输出标准 `info depth ...`（已移除 `info string mcts` / `root_moves_top5`）

### P1.5 树复用

- [x] `advance_root` + 兄弟裁剪 + `make_not_terminal`
- [x] UCI `rebuild_search_engine` 保树
- [x] 增量 `position ... moves ...` 复用（`same_input_window` + append-path）
- [x] 完整 `ResetToPosition`（gamebegin 重放 + `TrimTreeAtHead`）

### P1.6 验收基建

- [x] 固定 FEN 集 + `scripts/search_regression.ps1`
- [ ] px0 可执行文件并排 diff 闭环（需本机 build px0）
- [ ] 逐项解释剩余 diff（模型 / 无 TB / 无 MLH 等）

## P2 暂缓（搜索残余 + KataGo 评估）

> 用户确认：P2 先不进行。下列项保留记录，恢复时再开。

- [x] `EnsureNodeTwoFoldCorrectForDepth`（边统计 revert 已补）
- [x] `go depth` stopper（seldepth 达标停止）
- [ ] px0 并排 diff 验收（需本机 build px0）
- [ ] 在 px0 classic 主干稳定后，再评估 KataGo 并发与 pipeline

## P3 模型与训练

- [ ] 保持 `124x10x9 -> 2062 + WDL` 主契约稳定
- [ ] 继续用 `px0` 数据训练 baseline
- [x] 把 value 标签混合收回到 `px0` 风格固定 `q_ratio`
- [ ] 后续再单独讨论如何更好利用 `px0` 的 `q / visits / policy_kld`

## P4 仓库收口

- [ ] 删除搜索里残留的旧注释和旧命名
- [ ] 保持 `NextStep.md / TODO.md / ARCHITECTURE.md` 与真实主线一致

## P5 当前明确不做

- [ ] 不先做 `KataGo` graph / DAG
- [ ] 不先做 heap+BFS 搜索改革
- [ ] 不先做新的 time manager
- [ ] 不先做 `MultiPV`
- [ ] 不先做新的搜索实验参数面板
