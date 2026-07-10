# NextStep

## 当前阶段：P2

P1 已完成，当前进入 **P2：在 `px0 classic` 主干不变的前提下，吸收 `KataGo` 的并发与流水线细节**。

这阶段的目标不是再换搜索路线，也不是上 graph / DAG，而是把现有 `shared-tree MCTS` 做得更像真正可持续喂卡、可稳定并发、可继续扩展的搜索器。

当前已经落地的 P2 基线包括：

- `shared-tree` 多 worker 主框架
- `collision / in_flight / multivisit` 基础语义
- watchdog 独立 progress 输出
- root 层 in-flight 分数化折算
- 连续空转重试 + `yield`
- eval cache 基础统计可观测
- eval cache 统计口径已统一：`hits/misses` 为 lookup 口径，`miss_keys` 为去重后真实 miss key
- 单/并行空转重试语义已对齐；`retry_without_playout` 可经 benchmark / UCI 观测
- `go infinite` 无预算上限时空转达阈值后会 `yield` / `sleep` 退让

P2.1~P2.5 实现已落地，量化验收项留待 P2.6 回归与 px0 并排对照。

---

## P2 的总边界

### 要坚持的

- 框架继续对齐 `px0 classic`
- 并发和流水线细节参考 `KataGo`
- 搜索主线仍然是 `tree-based MCTS`
- 继续保留 `shared-tree`
- 继续保留真实 history 输入
- 继续保留当前 `124x10x9 -> 2062 + WDL` 模型契约

### 明确不做的

- 不引入 `KataGo graph / DAG / graphHash` 主线
- 不引入传统 alpha-beta `TT` 主线
- 不改成 heap+BFS 新搜索
- 不先做 `MultiPV`
- 不先做新的 time manager
- 不先改网络 I/O 和训练主语义

---

## 现在最重要的理解

当前搜索已经不是“是否能跑通”的问题，而是“并发细节是否正确”：

1. worker 是否会被同一路径卡住
2. gather 是否会形成有意义的 batch
3. backend 是否能持续被喂饱
4. root 附近是否能做到“宽而不乱”
5. stop / watchdog / progress 是否与并发搜索解耦

P2 就是围绕这五件事做。

---

## P2 实施顺序

### 第一步：先把线程行为调成更像 KataGo

目标：

- worker 允许高频失败后快速重试
- 不把 collision / in-flight 当异常，而当正常搜索状态
- 尽量减少“一个 worker 围着一个局部路径打转”的情况

验收关注：

- 相同 `go nodes` 下，PV 长度更稳定增长
- `seldepth` 不再虚高但 `pv` 仍然短
- 多 worker 时不出现明显卡死或集体空转

### 第二步：把 gather / backend / backup 的节奏继续拉开

目标：

- 搜索线程更像“持续供给器”
- backend 更像“持续消费器”
- 不再过度围绕“一轮 gather 成功了多少”设计控制流

验收关注：

- GPU 利用率更稳定，而不是偶发尖峰
- 小节点和中节点预算下都能形成稳定 batch
- `nps` 波动减小

### 第三步：只做轻量 eval cache，不做传统 TT

目标：

- 先减少重复 NN 推理
- 不合并整棵搜索统计
- 不引入 transposition graph 复杂度

验收关注：

- 相同局面重复搜索时，重复推理数量下降
- 不影响现有 tree reuse / history 语义
- 不改变 UCI / benchmark 统计口径

---

## tree reuse / eval cache / TT 的边界

这三件事必须分清：

### 1. tree reuse

这是当前已经有的主线：

- `advance_root`
- `reset_to_position`
- `position ... moves ...` 保留历史
- 尽量复用已有 subtree

这是必须继续强化的。

### 2. eval cache

这是 P2 可以做的：

- key：局面 / history 编码
- value：NN 输出
- 目标：减少重复推理

这不是传统 TT，也不应该先变成统计融合系统。

### 3. 传统 TT

当前不进入主线：

- 不做 `hash -> bound/value/bestmove`
- 不做跨分支统计合并
- 不做 graph transposition 主结构

等 `shared-tree + worker + eval cache` 稳定后，再决定要不要碰。

---

## 你实现时的工作原则

1. 先抄清楚 `px0 / KataGo` 的行为，再改
2. 不要一边改一边发明新启发式
3. 每一步都要能用固定 FEN + `go nodes` 做回归
4. 如果一个改动不能解释“为何更接近 px0/KataGo”，就先别做
5. 优先让搜索行为正确，再看吞吐

---

## P2 参考代码位置

下面这些位置是后续实现 P2 时最值得直接对照的代码。

### 我们当前代码

- 搜索主循环：
  - `crates/engin/src/mcts/search.rs`
  - 重点看：
    - `SearchSession`
    - `execute_one_iteration`
    - `run_parallel_with_progress`
- gather / selection / backup：
  - `crates/engin/src/mcts/worker.rs`
  - 重点看：
    - `gather_minibatch`
    - `select_pending`
    - `apply_minibatch`
    - `progress_from_tree`
- 边统计 / in-flight：
  - `crates/engin/src/mcts/node.rs`
- tree reuse：
  - `crates/engin/src/mcts/tree.rs`
- 并发协调：
  - `crates/engin/src/mcts/coordinator.rs`

### px0 classic

- 搜索主干：
  - `C:\Users\Administrator\projects\px0\src\search\classic\search.cc`
  - 重点看：
    - `Search::WatchdogThread`
    - `SearchWorker::ExecuteOneIteration`
    - `SearchWorker::GatherMinibatch`
    - `CalculateCollisionsLeft`
    - `CancelSharedCollisions`
- 节点与树：
  - `C:\Users\Administrator\projects\px0\src\search\classic\node.cc`
  - 重点看：
    - `Node::TryStartScoreUpdate`
    - `Node::FinalizeScoreUpdate`
    - `NodeTree::MakeMove`
    - `NodeTree::TrimTreeAtHead`
    - `NodeTree::ResetToPosition`
- backend 接口：
  - `C:\Users\Administrator\projects\px0\src\neural\backend.h`

### KataGo

- 顶层搜索线程循环：
  - `C:\Users\Administrator\projects\KataGo\cpp\search\search.cpp`
  - 重点看：
    - `runWholeSearch`
    - `runSinglePlayout`
    - `playoutDescend`
    - 失败后 `yield`
- 选子与 virtual loss：
  - `C:\Users\Administrator\projects\KataGo\cpp\search\searchexplorehelpers.cpp`
  - 重点看：
    - `selectBestChildToDescend`
    - child virtual loss 参与选择
- 节点结构：
  - `C:\Users\Administrator\projects\KataGo\cpp\search\searchnode.h`
  - 重点看：
    - `STATE_EVALUATING`
    - `virtualLosses`
- 分片锁：
  - `C:\Users\Administrator\projects\KataGo\cpp\search\mutexpool.h`
  - `C:\Users\Administrator\projects\KataGo\cpp\search\mutexpool.cpp`
- eval cache：
  - `C:\Users\Administrator\projects\KataGo\cpp\search\evalcache.h`
  - `C:\Users\Administrator\projects\KataGo\cpp\search\evalcache.cpp`

### 当前只看不抄的代码

- `KataGo` 的：
  - `graphHash`
  - `useGraphSearch`
  - transposition / graph 相关路径

这些代码可以拿来理解设计取舍，但当前阶段不进入我们主线。

---

## 我后续 review 的重点

你后面逐步实现时，我会主要看：

1. 是否偏离 `px0 classic` 主干
2. 是否偷偷引入 graph / TT / 大抽象
3. collision / in-flight / backup 是否语义闭合
4. gather / backend / stop 是否会出现线程死角
5. 代码是否比当前更简洁，而不是更绕
