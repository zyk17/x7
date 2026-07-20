# TODO

> 只做 px0 1:1 Rust 翻译。没有连续 px0 参考位置，不实现。

## 已完成

- [x] P0/P1：`src/chess` 的 board、legal move、FEN、Position、history、RuleJudge 与 release 对拍。
- [x] P2：`GameState`、UCI/controller、`position ... moves ...` 的完整 history、`Abort + Wait` 生命周期。
- [x] P3：classic tree、UCT、selection、extend、backup、tree reuse、WDL display。
- [x] P4 安全主线：ONNX、`CachingBackend`、minibatch、prefetch、collision、OOO、watchdog、legacy clock manager。
- [x] P4 生命周期：每次 `Abort + Wait` 后按 `src/search/classic/search.cc:1027-1064` 清理所有 shared
  collision virtual visit，防止长生命周期 Rust wrapper 把 reservation 带入下一条 `go`。
- [x] P4 UCI 构造期：按 `src/search/classic/search.cc:156-170` 在无限搜索禁用 `play` contempt，并仅在
  WDL rescale diff 非零时输出 warning；删除无 px0 对应且无人读取的旧 bridge 输出状态。
- [x] P4 watchdog：按 `src/search/classic/search.cc:351-389,981-1017` 在 stop 后仍输出最终 info；没有
  `bestmove` 响应资格的停止搜索输出 px0 的无进展 warning。
- [x] P4 DirectML 打包：静态 ORT + `DirectML.dll`，bundle UCI 冒烟不得回退 CPU。

## Stream：LC3-style 新搜索

参考文档：

- <https://lczero.org/dev/lc0/search/lc3/overview/>
- <https://lczero.org/dev/lc0/search/lc3/policy/>
- <https://lczero.org/dev/lc0/search/lc3/glossary/>

本地 lc0 未提供 LC3 源码；以下为架构对齐，不标称逐行翻译。`stream` 与 classic 完全隔离，首版只做 tree，
不做 DAG/TT。

- [x] S0 repository/event：sharded tree repository、edge-local started/completed、不可复制 reservation、
  owned history/variation/generation event；覆盖 collision claim 与 cancel/complete 不变量。
- [x] S1 policy（X7↔classic，非 LC3 公式）：selection 对照 classic FPU/`ComputeCpuct`；不得标称正式
  LC3 Policy 文档公式（该文档无公开具体式）。
- [x] S1 pipeline：单线程 Gather -> Eval -> Backprop 已接正式 `Backend`；terminal 和 failed Eval 均完成或撤销
  reservation。two-fold/rule60 通过 `PositionHistory::compute_game_result()` 进入 terminal 路径。
- [x] S2a–S2b：queues、常驻 worker、ONNX minibatch、SearchLimits、error cancel、长 movetime 冒烟（见历史条目）。
- [x] S1/S2 库内路径：主线 `stream::Search`（`search/stream/search.rs`）、ONNX minibatch、stop/drain、
  generation、settled；对拍 `stream_compare` / `stream_behavior_compare`（bestmove/legal；
  **不保证整条 PV 与 classic 一致**）。
- [x] S2c X7 output（库内，非 UCI）：项目批准对照 classic 的 selection + bestmove/PV；详见
  `temp_stream_x7_policy.md`。LC3 final-move 仍为 TBD，故不得标称正式 LC3 policy。
- [ ] S2d tree reuse：**未做**。每次搜索空 repository；无剪枝/GC。接跨手 reuse 前必须设计 prune。
- [ ] S2e 未做：stream Watchdog/`info`、MultiPV、NN `m` backup 聚合、multivisit 分配。
- [x] stream **不做** contempt / 非零 `draw_score`（收益低；`draw_score=0` 固定）。
- [ ] 后续：stream **精简 SearchParams**——多数 classic UCI 旋钮用不到；现为 `compute_cpuct` 等
  从 `classic::uct` / `classic::params` 整模块导入，应抽成 stream 自用的小参数面（或共享纯函数 +
  最小字段集），不再挂整份 classic 参数表。
- [ ] S3 UCI blocked：正式引擎仍走 classic。接 stream 需会话 lifecycle +（可选）reuse，并验证
  `position -> go -> stop -> position -> go` 无旧 generation、无 reservation 泄漏、恰好一次 bestmove。

## 停止路线：classic TaskWorkers T1-T3

classic 的 shared mutable tree 模型是 px0 特有的 C++ alias 约定，无法在无 raw pointer/unsafe/whole-tree lock
的 Rust 中逐字翻译。下列未完成项不再实施；classic 只作为当前 UCI 行为基线。

## NN：x7 v2 trunk

- [x] x7 v2 CNN trunk：基准为 `stem 124->256`、12 个 `256->112->112->256` 四卷积 PreAct bottleneck，
  两次 mean/max Global Broadcast 分布在三个 trunk stage 之间；policy 保留空间输出，WDL/moves-left 保持 global readout。
- [x] 训练与导出只接受 checkpoint `trunk_kind=x7_v2_bottleneck_gbroadcast`；YAML 可实验 width/blocks/bottleneck_channels，
  但续训与 `init_from` 必须精确匹配 checkpoint 元数据，避免误加载不同尺寸的 state dict。
- [x] 覆盖结构、参数量、前向形状、YAML、ONNX 导出回归：`5,690,808` 参数，`2062 + WDL[3] + moves_left[1]`。

- [x] P4 语义门槛：不改变 px0 backend 派生的 `SearchWorkers` 与 `TaskWorkers=-1` 自动推导；已逐项对拍
  UCI lifecycle、tree reuse、two-fold/rule60、terminal、selection、NN result sign、backup、best child/PV。
  参考 `src/engine.cc:187-235`、`src/chess/uciloop.cc:45-337`、
  `src/search/classic/search.cc:705-808,1423-1508,1510-1974,2109-2235`、
  `src/search/classic/node.cc:245-390,465-520`、`params.cc:478-481,622`。固定 UniformBackend 与真实
  `x7.onnx` lifecycle 冒烟已覆盖；`ponder`、`go depth/mate` 仍是明确未翻译接口。
- [x] 按 px0 `search.cc:1510-1550` 补 two-fold `initial_visits_` 统计基线修正，并以 tree reuse
  回归验证 nodes/playouts/display counters 不重复计数。
- [x] 按 px0 `src/engine.cc:187-219` 收口 UCI clock：`position` 启动时钟；只有无 `wtime/btime`
  的 `go` 重置时钟；带 clock budget 保留 position 起点。
- [x] 按 `src/search/classic/search.cc:705-808,1913-1919,2175-2257` 修正 terminal child rank 与无合法着
  leaf 的父边局部 WDL：已证明终局胜着必须压过败着，mate leaf 由 backup 统一翻转。
- [x] 按 `src/search/classic/stoppers/stoppers.cc:120-129` 修正 `go movetime`：复用 root 到时可直接停止，
  不强制新增 playout。
- [x] 按 `src/search/classic/search.cc:612-621`、`src/chess/uciloop.cc:279-287` 收口 root 无合法着：保留
  null move 的 `bestmove a0a0`，不能因没有 PV/info 而静默结束 UCI 请求。
- [x] 按 `src/search/classic/stoppers/common.cc:133-145` 补默认 `VisitsStopper(4_000_000_000)`；未指定
  `go nodes` 也必须限制树访问总量。
- [x] 按 `src/chess/uciloop.cc:230-237`、`src/search/classic/stoppers/common.cc:123,133-151` 收口边界
  UCI stopper：不额外拒绝 `go nodes 0`，bare `go` 依赖默认 hard cap，`go infinite movetime N` 忽略
  `movetime`。
- [x] 按 `src/search/classic/search.cc:2145-2153` 收口 NN policy 回填：每个 NN result 都写 prior 并排序，
  不保留 `node.N==0` 的 Rust-only 跳过条件。
- [x] 按 `src/search/classic/node.cc:319-341` 修正 two-fold 重开：`MakeNotTerminal` 保留 base WDL visit，
  再合并 child 统计，避免 tree reuse 改变节点平均值。
- [x] 按 `src/search/classic/node.h:375-377`、`search.cc:705-808` 修正 best-child 的未访问 child
  Q 代理：`N == 0` 必须使用默认 Q=0，不能把 terminal/tree-reuse 的 placeholder WDL 带入
  `DrawScore` 下的 PV/bestmove 排序。
- [x] 按 `src/search/classic/search.h:205-244,348-445` 与
  `search.cc:1069-1140,1322-1362,1423-1462` 定义 processing 的安全所有权边界：task 仅消费 owned
  extension input/history 并产出 result；owner 在 `WaitForTasks` 后独占写 tree、提交 backend input。
## 约束

- `UniformBackend` 只用于单元测试与对拍，正式 UCI 不得使用。
- stream 接入 UCI 前，不吸收 KataGo graph/DAG 或未经 LC3 文档定义的性能启发式。
