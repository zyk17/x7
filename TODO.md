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

## P2 进行中：吸收 KataGo 并发细节

> 当前 P2 的总原则：
> 框架继续对齐 `px0 classic`，并发/流水线细节参考 `KataGo`；
> 不引入 `graph / DAG / 传统 TT` 主线。

### P2.0 当前已落地，作为后续基线

- [x] `shared-tree` 多 worker 主框架
- [x] `collision / in_flight / multivisit` 基础语义
- [x] watchdog 独立 progress 输出
- [x] worker 在“无有效 playout”时支持持续重试，而不是直接退出
- [x] worker 在 backend 拥堵或连续空转时支持主动 `yield`
- [x] root 层 in-flight 参与选择时支持分数化折算，避免过度平均
- [x] eval cache 继续保持“仅缓存 `position/history -> NN output`”边界
- [x] benchmark 已可观测 eval cache 的 lookup / miss key 统计
- [x] 相关改动已过 `cargo check -p engin`
- [x] 相关改动已过 `cargo test -p engin --lib`

### P2.1 行为对齐前的硬边界

- [x] 不引入 `graphHash / useGraphSearch / DAG`
- [x] 不引入传统 alpha-beta 风格 `TT`
- [x] 不扩 `MultiPV`
- [x] 不先做新的 time manager
- [x] 不改现有模型 I/O 契约

### P2.2 worker 行为收敛

- [x] 明确区分“完成 playout”和“失败重试”
- [x] collision / in-flight 路径支持快速放弃并从 root 重进
- [x] 减少 worker 围绕单一路径反复自旋
- [x] 明确哪些失败应 `yield`，哪些失败应直接继续抢占
- [x] 给这部分补最小回归测试

参考代码：

- 我们：
  - `crates/engin/src/mcts/search.rs`
  - `crates/engin/src/mcts/worker.rs`
- px0：
  - `C:\Users\Administrator\projects\px0\src\search\classic\search.cc`
- KataGo：
  - `C:\Users\Administrator\projects\KataGo\cpp\search\search.cpp`
  - `C:\Users\Administrator\projects\KataGo\cpp\search\searchexplorehelpers.cpp`
  - `C:\Users\Administrator\projects\KataGo\cpp\search\searchnode.h`

验收：

- [ ] 固定 `go nodes` 下，PV 长度增长更稳定（待 P2.6 回归量化）
- [x] 单/并行在预算未耗尽时均不因 0 playout 提前退出；`go infinite` 有空转退让

### P2.3 gather / backend / backup 流水线

- [x] worker 更像持续供给器，而不是一轮一轮事务执行
- [x] gather 满足条件时尽快 flush 给 backend
- [x] backend 忙时减少无意义等待
- [x] backup 保持短路径、短锁、无额外统计分叉
- [x] 继续控制树写锁持有范围

参考代码：

- 我们：
  - `crates/engin/src/mcts/search.rs`
  - `crates/engin/src/mcts/worker.rs`
  - `crates/engin/src/mcts/coordinator.rs`
- px0：
  - `C:\Users\Administrator\projects\px0\src\search\classic\search.cc`
  - `C:\Users\Administrator\projects\px0\src\neural\backend.h`
- KataGo：
  - `C:\Users\Administrator\projects\KataGo\cpp\search\search.cpp`

验收：

- [ ] GPU 利用率更稳定，不再主要依赖偶发尖峰（待 P2.6 量化）
- [ ] `nps` 波动减小（待 P2.6 量化）
- [x] benchmark / UCI 统计口径不变

### P2.4 root 附近的访问分配

- [x] root 附近避免“过度平均”导致树处处都浅
- [x] 保持“宽而不乱”，不是退回纯 DFS
- [x] 检查 collision / virtual loss 是否真的把 worker 分流到有效分支
- [x] 固定对称开局和典型中局做对照

参考代码：

- 我们：
  - `crates/engin/src/mcts/worker.rs`
- px0：
  - `C:\Users\Administrator\projects\px0\src\search\classic\search.cc`
  - `C:\Users\Administrator\projects\px0\src\search\classic\stoppers\common.cc`
  - `C:\Users\Administrator\projects\px0\src\search\classic\stoppers\stoppers.cc`
- KataGo：
  - `C:\Users\Administrator\projects\KataGo\cpp\search\searchexplorehelpers.cpp`

验收：

- [ ] 相同 `go nodes` 下，PV 比当前更长（待 P2.6 对照）
- [ ] `seldepth` 与 `pv` 更匹配，不再出现“深度看着还行但 PV 很短”（待 P2.6 对照）

### P2.5 eval cache（轻量）

- [x] 明确只做 `position/history -> NN output` 缓存
- [x] 不在这一步合并搜索统计
- [x] 不引入 graph transposition 语义
- [x] cache 命中 / miss / 复用效果要可观测

参考代码：

- 我们：
  - `crates/engin/src/mcts/policy_value.rs`
- KataGo：
  - `C:\Users\Administrator\projects\KataGo\cpp\search\evalcache.h`
  - `C:\Users\Administrator\projects\KataGo\cpp\search\evalcache.cpp`
  - `C:\Users\Administrator\projects\KataGo\cpp\search\mutexpool.h`
  - `C:\Users\Administrator\projects\KataGo\cpp\search\mutexpool.cpp`

说明：

- 当前只吸收 `eval cache` 的思路
- 不吸收 `graphHash / useGraphSearch / DAG`

验收：

- [ ] 重复局面下重复推理次数下降（待 P2.6 量化）
- [x] 不影响 tree reuse / history 正确性

### P2.6 验收与对照

- [ ] 固定 FEN 集持续回归
- [ ] 与 px0 可执行文件继续并排 diff（bestmove / pv / nodes / nps）
- [ ] 明确哪些差异来自：
  - 模型不同
  - 无 TB
  - 无 MLH
  - 搜索并发细节尚未完全对齐

参考代码：

- px0 UCI / info 输出：
  - `C:\Users\Administrator\projects\px0\src\search\classic\search.cc`
  - `C:\Users\Administrator\projects\px0\src\chess\uciloop.cc`
- 我们：
  - `crates/engin/src/uci.rs`
  - `crates/engin/src/benchmark.rs`

### P2.7 当前已完成、纳入 P2 基线的项

- [x] `EnsureNodeTwoFoldCorrectForDepth`
- [x] `go depth` stopper（seldepth 达标停止）
- [x] `shared-tree` 多 worker 主框架
- [x] `collision / in_flight / multivisit` 基础语义
- [x] watchdog 独立 progress 输出
- [x] root 层 in-flight 分数化折算
- [x] 连续空转重试 + `yield`
- [x] eval cache 基础统计可观测

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
- [ ] 不先做传统 `TT`
- [ ] 不先做 heap+BFS 搜索改革
- [ ] 不先做新的 time manager
- [ ] 不先做 `MultiPV`
- [ ] 不先做新的搜索实验参数面板
