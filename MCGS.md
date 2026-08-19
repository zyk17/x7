# MCGS 研究设计

本文件属于 `feat/mcgs` 分支，记录当前正式搜索 **stream Monte Carlo Graph Search（MCGS）**
的设计边界。它不是 px0/LC3 的等价移植：worker 生命周期、统计回传与 visit 分配以本仓实现为准。

本实现使用连续 `Gather → Eval → NN → Eval → Backprop`：PUCT 使用 edge in-flight
reservation 作为 virtual visit（计入 started N，偏转选择）；Gather 每次采集一个叶子，
collision 先挂起 reservation / μ，该叶子自己的 backprop complete 后再 cancel，不加 completed visit。Eval 持续向 NN 提交已编码 tensor；NN 空闲时立即处理当前
队列可得请求，不等待逻辑搜索轮次。Gather 在 claim 前把已交给 Eval 的叶子限为两个 `MiniBatchSize`。没有 prefetch 或 tree-batch gather。
时钟到期后 owner 会 stop 并取消未完成的 Eval/NN claim，不把推荐 batch 的推理时间算进 `go movetime` 之后。剩余预算不足 500ms 时提交窗口收窄，避免第一刀合批单独超过时限。

不用 classic 那种一次评估记 K 次 visit 的 **multivisit**。GPU 合批在 NN 队列，不靠 Gather
一次收一批。实战以 `μ=FPU` virtual mean 做比纯 virtual visit 更明显、但仍温和的分流；K 份
假样本会一次打偏 in-flight Q/N。后续若研究分流强度，改 virtual mean / virtual visit，不加 K。

## 目标

当前 repository 已是 64 分片 key-value store。唯一使其成为 tree 的地方是 key：

```text
child_key = HashConcatenate(parent_key, move)
```

MCGS 的 node identity 改为：

```text
node_key = GraphKey(board)
```

相同棋盘从不同 variation 到达时取到同一个 node，因而复用其图结构和统计。这里不另加 transposition
table：repository 就是唯一的 node store。完整历史不属于 GraphKey，而属于本次 event 的 `Variation`。

历史/语义参考（不是兼容目标）：

- LC3 Overview, “Node Repository / Keys / Values”
  <https://lczero.org/dev/lc0/search/lc3/overview/>。
- LC3 Search Policy, `MakeNodeKey`、`DistributeVisits`、`MakeEdgeDelta` 与
  `UpdateNodeAggregate`
  <https://lczero.org/dev/lc0/search/lc3/policy/>。本仓未移植其 policy 公式或
  `DistributeVisits`；每个 event 只做一次 PUCT selection。
- 本地 Lc0 `src/search/dag_classic/node.h:614,957-959`：共享 low node 与弱引用表仅作
  DAG 语义参考；不移植 classic worker/GC 结构。
- 本地 px0 `src/search/dag_classic/{node.h,search.cc}`：理解数据布局与回传的历史参考，
  不是 stream worker 模型的移植目标。
- 本地 KataGo `C:\Users\Administrator\projects\KataGo`（按需）：如 `docs/GraphSearch.md`、
  NN cache 与部分搜索细节；不是默认必读，也不承诺行为等价。

## 统计语义：KataGo 的 idempotent MCGS

KataGo `docs/GraphSearch.md` 的关键结论是：不能把 child node 的总 visit 当成一个 parent action 的
visit；但也不能把 shared node 退化为“只缓存 NN”。MCTS 的 visit distribution 是每个 state 上的
posterior policy，故 MCGS 必须同时保留 **edge action count** 与 **shared node value**。

一个 shared state node `n` 保存：

- 原始 NN 预测 `U(n)`（WDL/M）；
- node `N(n)` 与 `Q(n)`；它们不是入边 playout 的普通 running average；
- 展开状态、合法走法、policy，以及仅由棋盘唯一决定的 terminal；不保存 history-dependent proof。

每条有向 edge `(n,a)` 保存：

- `P(n,a)`；
- `N(n,a)`：**只有 PUCT 在 n 实际选择 a 时才增加**；
- edge-local in-flight reservation。

语义公式为：

```text
N(n) = 1 + Σa N(n,a)
Q(n) = [U(n) + Σa N(n,a) · ValueForParent(Q(child(n,a)))] / N(n)
```

其中 `1` 与 `U(n)` 是第一次网络评估给局面值的正则化权重。`M` 若继续参与搜索，必须采用同样的
递归定义，而不能沿用 tree 的路径平均。PUCT 使用 `N(n,a)`，而不是 child node 的 `N(child)`：

```text
score(n,a) = ValueForParent(Q(child(n,a)))
             + cPUCT · P(n,a) · sqrt(Σb N(n,b)) / (1 + N(n,a))
```

这正是保持“一个 node 内的 visits 是该 node 自己学习出的 policy”的必要条件。

### A/B/D 回传不变量

设两条 variation 汇合：

```text
A → B → D
A → C → D
```

一次沿 `A → B → D` 的 playout：

1. 只增加本 path 的 `N(A,B)`、`N(B,D)` 与 D 以下实际经过的 edge count；
2. 逆向重新计算 path 上 A、B、D 的 `N/Q`；
3. **不**增加 `N(A,C)` 或 `N(C,D)`，也不直接重算 C。

因此 `C→D` 不会因 B 的访问伪装成“已被 C 验证过”；但 D 的 `Q(D)` 已因共享 continuation 变得更准。
当 C 以后真正被访问时，它根据自己的 `N(C,D)` 和最新 `Q(D)` 重新计算 `Q(C)`。这叫 stale Q：非路径
parent 暂时落后，但不会永久落后，且避免了向所有 ancestor 广播造成的统计污染。实现上，Gather 每次进入
已展开 node 后、PUCT 选边前必须先做这次重算；只在 Backprop 时重算不足以满足这个条件。

根选边默认以 root edge 的 `N(root,a)` 排名。node `Q(root)` 是 root posterior policy 的局面估值；两者
都需要，但不能互相替代。当前 X7 在最终 Decision 额外允许 root LCB：仅让 completed N 达到 N 第一候选
`15%` 的非终局 edge 参与，按 `Q - 5·标准误` 选最保守候选。其二阶矩遵循 node 的幂等重算；当前每次
NN evaluation 恰好完成一份 edge visit，样本量直接使用 completed N。转置 child 的总 N 不会伪装成该 action
的 Evidence。LCB 不进入 PUCT 或回传。

### 更新方式选择

首版研究应采用 KataGo 的 **idempotent** 更新：每次 path 回传后，按当前所有 outgoing edge 的
`N(n,a)` 和 child `Q` 重算该 node 的 `N/Q`。它是正确性最直接的写法，代价为每次更新遍历该 node 的
edges；我们的瓶颈目前是 NN，因此先不为节省这点 CPU 引入增量近似。

Czech–Korus–Kersting 论文采用另一条路线：edge 和 node 都保存 Q，发现 transposition 时以 child node
的较精确 value 对 edge Q 做渐进纠偏，并按阈值提前结束 playout。这是可研究的性能策略，不是 MCGS
正确性的最小要求，第一阶段不引入其阈值与 clipping 参数。

### px0 DAG：干净的具体对照

px0 的 `dag_classic` 把一个局面拆成两层对象：

- 每个入边 `Node` 保存该 parent 下的局部 `N/Q/M` 与 move/prior；这相当于本设计的 edge action
  statistics，故 root 选边与 PUCT 不会误用 shared child 的总访问数；
- 多个 `Node` 指向同一个 `LowNode`；它保存一次 NN expansion、共享 `N/Q/M`、policy、terminal/bounds。

回传时 px0 一面更新实际 path 的 `Node`，一面更新经过的 `LowNode`；若发现 shared `LowNode` 比某个
入边 `Node` 拥有更新的信息，`MaybeAdjustForTerminalOrTransposition` 用 value delta 把该信息同步给当前
path。这是 **增量同步**，与 KataGo 的“访问 path 时完整重算”不同，但两者都遵守同一不变量：
`N(parent, move)` 绝不等于 child 的共享总 `N`。

因此本分支不应把 px0 的 `Node/LowNode` 类层级原样搬入 Rust。当前 `Edge + NodeRepository` 已能表达相同
关系：Edge 承担 px0 `Node` 的局部 action 统计；repository 中的 Node 承担 `LowNode` 的共享状态。需要做的
是选择一种更新语义：首版优先 KataGo 的幂等重算；若其实际 CPU 成本成为问题，再以 px0 的增量同步作为
有明确参考的备选实验，而不是预先混入 delta、阈值和 parent 反向索引。

px0 的 key 也不是简单棋盘 hash：`search.cc:2051` 使用
`history.HashLast(cache_history_length + 1)`。px0 因此选择了更严格、但合并更少的 DAG；它是具体实现的
优秀参考，不规定本项目必须使用相同 key。

## GraphKey 与 Variation：有意识的上下文近似

若把 NN 的 8 步历史或 `rule60_ply` 放进 key，`A→B→D` 与 `A→C→D` 往往无法合并 D；更长路径的
`A→B→C→D→E` 与 `A→F→E` 也会因 rule60 不同失去 E。这样虽更严格，却会使 MCGS 退化接近 tree。

本分支的研究假设改为明确的两层：

```text
GraphKey(board) = 仅当前棋盘状态（含走方）
Variation       = `root_history + moves`；可重建完整 PositionHistory
```

因此，D/E 可以跨走法共享；但 shared node 的 `U/Q/M` 是不同上下文 evidence 的混合，而不是严格 Markov
state value。这是为换位收益选择的 **intentional contextual approximation**，必须在 Phase 4 以稳定性与
Elo 验证，不能写成“精确状态合并”。

这里不新增 `PathContext` 结构。现有 event `Variation` 就是路径上下文的 owned 表达；worker 从它重建并临时
维护可变 `PositionHistory`，供 NN 编码、rule60、重复以及连将/追击裁判使用。

这还留下一个必须显式决定的 NN 规则：NN 输入实际依赖 8 步 history 与 rule60，而 shared node 只能保存
一份 policy/value。首版候选是“第一次完成 expansion 的 Variation 决定该 node 的 `U/P/M`”，之后所有 parent
复用它；这与当前按局面复用的 NN cache 方向一致，但会带来 arrival-order 敏感性。它必须有固定的合成测试
与重复运行测试；若不接受该近似，就不能使用 board-key 图。

## Gather、event 与回传

tree 版可以由 `parent_key.child(move)` 得到 child key。图版必须从 event 的 variation 重建 child
position，再计算普通 `GraphKey(child_position.board())`。当前 event 保持最小的
`node_path + reservations`：两者的同一索引顺序表达本次实际走过的 parent edge；正常 path 比
reservation 多一个 leaf node。首次真实重复则额外记录一个 ContinuationTree entry：该 entry edge
不绑定 child，回传时把树根当前 value 作为它的 edge-local 样本。Tree 内的零化 edge 则正常绑定
普通 Graph child；一条 variation 因此可在零化后再次进入新的 Tree，event 记录每个 entry。无需引入
`PathStep` 结构。

一次回传有两件不同的事，不能混为一谈：

1. 只为实际经过的 directed edge 完成 reservation，并增加它的 `N(n,a)`；
2. 自叶向根，对实际 path 上每个 node 按前述公式幂等重算 `N/Q`。重算读取该 node 的全部
   outgoing edge count 与当前 child `Q`，但不改变非 path edge 的 count。

所以共享 node 的 value 会被更新，未经过的 parent 不会被广播更新。若同一 state node 在一条
variation 中出现两次，不能无限下降，也不能把一次 evidence 重复写入 edge；这属于环处理，而不是
普通 backprop。

## 环、真实重复与 ContinuationTree

普通图允许跨分支汇合。repository 只在**首次绑定普通
shared edge**时做拓扑剪枝：每个 node 记下首次到达深度。回边
`depth(parent) > depth(child)` 永久标记 `TopologyPruned`（PUCT 跳过，reservation 取消）；
浅层父节点仍可接到已被更深路径先展开的 child。这是 O(1) 的 X7 结构近似，
不是精确判环。同路径棋规重复不走这里。

若一个非根 shared node 的所有边都已如此过滤，它作为 graph 边界结束本次 playout：只把该 node 自己
已有的 U/Q 回传给进入它的实际入边；被过滤 edge 仍不获得 N/Q。这避免 fixed-playout 在无候选 node
崩溃，也不把任意被过滤着伪造成已搜索结果。

这不是棋规裁决，而是 X7 的结构近似：绝大多数这类绕路只改变 rule60 并拉长 moves-left，不能安全接入
idempotent shared-Q。它必须保留为可复查决策；若实战证明长将、长捉或 rule60 的差异不能忽略，再改为
更严格模型，不把该边悄悄当成 draw 或 NN leaf。

候选 child 的 board 已在**当前 variation**出现时，优先于上述拓扑检查按真实 repetition 处理：

1. 走出该边后的节点就是 `ContinuationTree` 根，即此 variation 的第一次重复局面；入边不绑定 shared graph。
2. ContinuationTree 的每个 key 包含当前 board 与自最近零化着以来的完整规则 history，因此重复上下文
   内不做 transposition merge；唯一共享资源是按 board key 的 NN cache。零化 edge 的 child 立即回到
   普通 Graph，并可复用已有 Graph node 与其后续子图；Tree 之前的 N/Q 不迁移到该 Graph node。
3. 第一次重复（`repetitions == 1`）继续正常 PUCT / NN evaluation。树内再次出现相同局面后
   `repetitions >= 2`，才调用 `RuleJudge` 得到 path-local terminal。

这样沿深度向上的回边被剪掉，真实循环仍有足够路径历史交给棋规裁判。Graph → Tree entry 不绑定
contextual child；Tree → Graph 只在零化 edge 发生并正常绑定 board child。若当前完整 history 自最近
零化着以来已经出现重复，下一回合的 root 仍使用对应 contextual key；零化后则直接成为本回合已创建或
复用的普通 Graph root。普通 graph 与 Tree 不迁移 N/Q；只有 Tree 的零化后 descendants 使用 Graph
自己的共享统计。

## 跨回合复用与 GC

当前“走一步后删除 sibling subtree”只对 tree 正确；图中的 sibling 可能仍从另一条路径可达。

第一版不维护 parent refcount，也不复刻 Lc0 classic 的复杂释放器。每次 `position` 已 drain 后：

1. 只从**当前**搜索 root 做一次 visited-set traversal；
2. repository 删除未标记（从当前根不可达）的 node；
3. 不可达 node 的 `Arc` 自然释放。

这是 `O(V + E)`，但只发生在跨回合 GC 边界，且 repository 本来就是 map。若测量证明成为瓶颈，再
讨论 refcount 或分代；当前不提前引入。

unrelated `position` 仍直接换新 repository。悔棋由完整 UCI `position ... moves` 重建 root，
不保留旧 root / sibling 作为额外 GC 起点。Graph→Tree 入口不绑定 child，sweep 看不到这些
TreeNode；当前 root 的 sibling prune 在 `position` abort 之后同步做，不与下一手搜索重叠。

## 必须同时改掉的 tree 假设

- 与 history 无关、由当前 position 唯一决定的 terminal 可保存在 shared node。当前 tree 的
  proven-bound / sticky mate 依赖路径语义，不能直接共享给全部入边；首版按 px0 的
  `Node`（局部入边）/`LowNode`（共享局面）分层处理，不把 history-dependent result 或 proof 直接写成
  graph-wide terminal。
- PV 与 `graph_shape` 必须有 path-local visited set，遇环停止并显示 cycle，不能无限递归。
- `subtree_is_settled`、GC 和 root reset 都必须使用 graph traversal 的 visited set；不能通过
  `key.child(move)` 推导 child。
- UCI `bestmove` 只读取 root edge，故语义可保持；其 PV 需要 graph-safe 路径展开。

## 分阶段验收

### Phase 0：结构测量

迁移前先在 tree 快照上按 board key 测量重复 node，得到 `mergeable` 基线；它回答的是：若保留该次
tree 搜索的所有 path，合并相同棋盘后可去掉多少物理 tree node。

迁移后 `graph_measure` 从实际 repository 的已完成 edge 只遍历一次每个共享 node。它统计 reachable
node/edge、path-local terminal edge，以及同一 child 被多个 parent 指向的 `merged_edges`。后者只是一跳
fan-in，不能替代 tree 基线中的递归 path 展开率；不枚举所有 variation 是为避免 DAG 的路径组合指数膨胀。

```powershell
cargo run -p engin --release --bin graph_measure -- --fen "<FEN>" --moves "..." --playouts 10000
```

两种指标都不自动等价为固定时间收益：NN cache、collision、不同选择路径都会影响实际 evaluation
数和 Elo。历史、rule60、重复冲突仍由 Variation 在 path-local 层裁决，不由本工具把它们混入共享 node
的统计。

当前 `data/x7.onnx`、DirectML、fresh graph、25,000 completed playout 的第一组结构基线：

| 局面 | graph node | completed child edge | direct merged edge | NN eval |
| --- | ---: | ---: | ---: | ---: |
| `initial` | 24,984 | 29,723 | 4,740 (15.9%) | 24,984 |
| `middle_01` | 23,211 | 24,414 | 1,204 (4.9%) | 23,051 |
| `middle_30` | 24,650 | 26,180 | 1,531 (5.8%) | 24,649 |
| `evasion_01` | 6,096 | 6,277 | 182 (2.9%) | 5,771 |
| `proof_mate_01` | 24,567 | 28,304 | 3,738 (13.2%) | 24,514 |

随预算从 1,000 到 25,000，`initial` 的 direct fan-in 从 10.1% 升至 15.9%，
`proof_mate_01` 从 6.5% 升至 13.2%；`evasion_01` 始终约 3%。这说明当前 MCGS 的直接汇合主要
来自宽且会重汇合的搜索；它不否定 tree 快照的完整展开合并率更高。

同模型、同 4/4 Gather/Eval worker、10,000 completed playout 的 tree/graph 吞吐抽样没有一致的方向：
`middle_30` 的 MCGS 约快 5%，`evasion_01` 约慢 6%，initial 与 proof 接近。根候选虽多数
一致，但 Q 与个别候选会变化；MCGS 改变的是共享 value 下的搜索统计语义，不能当作纯性能替换。
后续以固定时间 Elo 验收，吞吐只作为诊断。

因此，direct fan-in 只能说明当前 graph 的局部形状。与下方 tree 基线合在一起，才说明 board-key
存在潜在复用；之后必须观察 arrival-order 稳定性与 Elo，不能只看任一结构百分比。

迁移前 tree 搜索在同一批局面按 1k / 5k / 10k / 25k completed playout 扫描的 `mergeable` 趋势：

| 局面 | 1k | 5k | 10k | 25k |
| --- | ---: | ---: | ---: | ---: |
| `middle_01` | 12.7% | 18.6% | 17.9% | 18.8% |
| `middle_05` | 6.9% | 10.9% | 12.4% | 18.4% |
| `middle_15` | 4.0% | 9.3% | 14.4% | 19.5% |
| `middle_20` | 15.5% | 18.0% | 25.9% | 30.9% |
| `middle_30` | 5.0% | 12.9% | 14.6% | 19.0% |
| `middle_35` | 17.8% | 22.8% | 27.3% | 30.2% |
| `middle_38` | 18.5% | 24.4% | 24.3% | 30.9% |
| `evasion_01` | 20.6% | 17.7% | 14.2% | 8.6% |
| `evasion_05` | 24.5% | 44.6% | 52.0% | 61.1% |
| `proof_mate_01` | 25.9% | 38.7% | 43.2% | 39.7% |

多数局面随探索增加而有更高合并率，说明换位在树加深后持续累积；少子力/强制局面增长尤其显著。每次运行的
并发调度会造成小幅波动，表用于看趋势而非精确单调性。`evasion_01` 是窄而强制的例外：高 collision 会让
相同 completed playout 下的已展开树显著变浅，故其 board-key 合并率不单调；这不是换位语义反例。

此前该局面在 25k 曾出现 `f1e0`。它不是 policy 映射缺口，而是规则 bug：攻击表把九宫边点也当作士位，
使非法士着有机会进入搜索。现已把士位与将的九宫范围分开，并在 FEN 与内部一致性校验中拒绝非法士位；表中的
25k 数据来自修复后的运行。

### Phase 1：最小 repository graph（已完成）

将 child identity 改为 board graph key，并改为 edge-local action count + shared node idempotent value；只用
合成 node/edge 测试验证两个 parent 可共享一个 child：`B→D` 的回传不增加 `C→D` 的 `N`，但 C 在下一次
重算时可读取更新后的 `Q(D)`。`M` 参考 px0 的分层语义：shared state 保存其估计，经过某入边到 parent 时
加一；局部入边可保持自己的同步值，不能沿用 tree 的路径平均。rule60/repetition 的 pseudo-terminal M
只属于本次 path。

### Phase 2：流式路径安全（已完成）

使用 `node_path + reservations`、环检测和 reservation 配平；通过并发 completion/cancel、cycle termination
与 release UCI 冒烟验证 `position → go → stop → position → go`。

### Phase 3：图 GC、PV 与 terminal（已完成）

改为当前-root 可达 mark/sweep GC；PV/graph-shape 防环；明确 history-dependent terminal 的局部语义。
流式路径与 UCI 生命周期回归已通过；在此基础上才能开始固定时间 Elo 比较。

### Phase 4：评估

同一网络、worker、MiniBatchSize 和固定时间/固定节点下，对比：唯一 state 数、NN eval 数、cache hit、
collision、有效完成 playout、PV 稳定性和 Elo。合并率或 EPS 上升本身不是采用理由。

图实现的 CPU/NPS 回归不以初始局面单独判断：它的换位负载偏低。固定采用
`middle_20`（宽换位）、`evasion_05`（强制/窄树但高合并）和 `proof_mate_01`（强制反例）三个
局面，在相同 ONNX、4/4 Search/Eval、MiniBatchSize、fresh graph 与固定时间下报告多轮 NPS/EPS；初始局面只保留为
一般流水线冒烟。每次图结构、拓扑插入、GC 或 worker 热路径改变后，都用这组比较，不能由单一局面外推。

#### 2026-08-10：`b7b8` 的 tree/graph 候选分歧

`data/x7.onnx`、完整历史 `test.pgn` 至 `41...e8d7`、25,000 completed playout、
`CPuct=1.5`、`CPuctBase=2000`、`CPuctFactor=1.347017`、`FpuReduction=0.2` 下，
MCGS 根选 `b7b8`（N=3733），tree 基线根选 `b4c6`（N=5983；`b7b8` N=5261）。
本地 Pikafish 5 秒 MultiPV 的前列为 `h8h7`、`b4c6`、`b4a6`、`h8i8`、`a3a4`，不含 `b7b8`。

这说明图不是纯吞吐替换：在相同模型、历史、参数与预算下，shared-state 的路径演化足以改变根
action N 排名。它是需要复现的 MCGS 回归样本，但还不能据此直接认定图统计错误：tree 分支并非与
当前分支逐提交同步。后续诊断应记录 `b7b8`/`b4c6` 子图的 transpose、shared node fan-in、首次读取
shared child Q 的位置及其后的 edge N/Q，而不是先修改 PUCT 或图回传语义。

## 当前结论

MCGS 不是“加一个 TT”。当前实现已切换普通 board-key、edge-local N、幂等回传、真实重复的 ContinuationTree、首次到达深度的 topology prune、可达 GC 与防环 PV。
它仍是 board-key 的上下文近似，不是严格 state 合并。下一步评估 reuse/stop 边界、唯一 state 数、NN eval 数、
稳定性与 Elo；合并率或 EPS 上升本身不是采用理由。

## 后续 CPU Proof 的边界

当前 CPU 在 NN batch 等待期间仍有余量；这不是要求刻意降低 collision 的理由。未来若加入 Proof，应优先
利用 collision/等待期间未占用的 CPU 产生可验证的新 Evidence，并且不阻塞 Gather、Eval、Backprop 或长期持有
graph 锁。它可能降低 NPS，但是否采用只能由固定时间 Elo 判断；“NPS 不下降”不是硬约束。
