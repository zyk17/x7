# MCGS 研究设计

本文件只属于 `feat/mcgs` 分支。它记录 stream **Monte Carlo Graph Search（MCGS）** 的
最小实现与研究边界；不会改动正式主线或 worker 生命周期。

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

参考：

- LC3 Overview, “Node Repository / Keys / Values”
  <https://lczero.org/dev/lc0/search/lc3/overview/>。
- LC3 Search Policy, `MakeNodeKey`、`DistributeVisits`、`MakeEdgeDelta` 与
  `UpdateNodeAggregate`
  <https://lczero.org/dev/lc0/search/lc3/policy/>。
- 本地 Lc0 `src/search/dag_classic/node.h:614,957-959`：共享 low node 与弱引用表仅作
  DAG 语义参考；不移植 classic worker/GC 结构。
- 本地 px0 `src/search/dag_classic/{node.h,search.cc}`：与我们的规则和网络边界更接近的 DAG
  实现参考。它用于理解具体数据布局与回传，不作为 stream worker 模型的移植目标。

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

根选边仍以 root edge 的 `N(root,a)` 排名。node `Q(root)` 是 root posterior policy 的局面估值；两者
都需要，但不能互相替代。

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
position，再计算 `GraphKey(child_position.board())`。当前 event 保持最小的
`node_path + reservations`：两者的同一索引顺序表达本次实际走过的 parent edge；正常 path 比
reservation 多一个 leaf node，环边不加入重复 child node，因而两者长度相同。Edge 自己绑定
`child_key`，所以不需要再引入 `PathStep` 结构。

一次回传有两件不同的事，不能混为一谈：

1. 只为实际经过的 directed edge 完成 reservation，并增加它的 `N(n,a)`；
2. 自叶向根，对实际 path 上每个 node 按前述公式幂等重算 `N/Q`。重算读取该 node 的全部
   outgoing edge count 与当前 child `Q`，但不改变非 path edge 的 count。

所以共享 node 的 value 会被更新，未经过的 parent 不会被广播更新。若同一 state node 在一条
variation 中出现两次，不能无限下降，也不能把一次 evidence 重复写入 edge；这属于环处理，而不是
普通 backprop。

## 环与重复：第一版的硬边界

图允许跨分支汇合，但 shared-Q 的幂等重算不能读成循环依赖。Gather 必须维护本 event 的
visited board keys；此外 repository 在**首次绑定一条 edge** 时做一次 DFS：若候选 child 已经
沿既有图边可达 parent，则不绑定这条会闭环的 edge。该 DFS 不在正常 PUCT 热路径运行，且用一把
很小的链接锁包住“检查 + 绑定”，避免两个 worker 同时穿透检查而新建二元环。

这不是重复裁决。截断 edge 仍代表一个合法、但无法安全接入 shared-Q 图的走法：它以 candidate
child 首次网络预测 `U(child)` 作为这一次 edge-local leaf sample 完成，然后不建立 child link。
因此既不会把非重复局面伪装成和棋，也不会让 `Q(B)` 与 `Q(D)` 在 `B ↔ D` 中递归读取。设计取
KataGo `cpp/search/search.cpp:1426-1445` 对 graph-path cycle 的截断作为参照；KataGo 的累计统计
语义与本项目不同，故 X7 选择固定 U leaf 而不是其“加一次 parent edge visit 后结束”的方式。

实现只在 candidate child 已存在于 repository 时 DFS：新 child 不可能沿既有图边回到 parent，直接绑定。
被截断的 edge 缓存固定 `U(child)`，以后再次被选中不再遍历。两项均不改变规则裁决或 edge-local 统计；若
基准显示这类 edge 仍是热点，才考虑更细的 cut 统计，不提前引入索引或 parent 表。

path-local 闭环仍按以下规则处理：

候选 child 的 board key 已在本 path 出现时，一律视为 **repetition**，而非 collision 或普通 graph edge。
worker 在私有 `PositionHistory` 上先调用既有 px0 风格 extension/rule judge；若首次闭环尚未达到
two-fold 的正式裁决门槛，首版按“已重复但未能归责”记为本地和棋，保证图不会无界下降。repetition、
two-fold 与 rule60 都绝不创建第二个 shared node，也绝不写入已存在 node 的 global terminal/proof；结果作为
**这一次** `parent → move` 的 edge-local terminal 样本完成。相同 board edge 可以由不同 history 到达，
所以不能 first-writer-wins：edge action Q 是其所有本地终局样本与其余访问读取的 shared child Q 的加权组合。
这是有意识的上下文近似，仍须用 px0 的 two-fold / `RuleJudge` 用例验证方向。

## 跨回合复用与 GC

当前“走一步后删除 sibling subtree”只对 tree 正确；图中的 sibling 可能仍从另一条路径可达。

第一版不维护 parent refcount，也不复刻 Lc0 classic 的复杂释放器。每次 `position` 已 drain 后：

1. 从全部 retained roots 做一次 visited-set traversal；
2. repository 删除未标记 node；
3. 不可达 node 的 `Arc` 自然释放。

这是 `O(V + E)`，但只发生在跨回合 GC 边界，且 repository 本来就是 map。若测量证明成为瓶颈，再
讨论 refcount 或分代；当前不提前引入。

unrelated `position` 仍直接换新 repository。悔棋保留的历史 root 仍是 reachability 的起点。

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

同模型、同 4/4/1 worker、10,000 completed playout 的 tree/graph 吞吐抽样没有一致的方向：
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

使用 `node_path + reservations`、环检测和 reservation 配平；通过并发 completion/cancel、cycle termination 与
真实 ONNX `position → go → stop → position → go` 回归。

### Phase 3：图 GC、PV 与 terminal（已完成）

改为 retained-root reachability GC；PV/graph-shape 防环；明确 history-dependent terminal 的局部语义。
本地真实 ONNX 已通过 `position → go → stop → position → go` 回归；在此基础上才能开始固定时间 Elo
比较。

### Phase 4：评估

同一网络、worker、MiniBatchSize 和固定时间/固定节点下，对比：唯一 state 数、NN eval 数、cache hit、
collision、有效完成 playout、PV 稳定性和 Elo。合并率或 EPS 上升本身不是采用理由。

图实现的 CPU/NPS 回归不以初始局面单独判断：它的换位负载偏低。固定采用
`middle_20`（宽换位）、`evasion_05`（强制/窄树但高合并）和 `proof_mate_01`（强制反例）三个
局面，在相同 ONNX、4/4/1、MiniBatchSize、fresh graph 与固定时间下报告多轮 NPS/EPS；初始局面只保留为
一般流水线冒烟。每次图结构、DFS、GC 或 worker 热路径改变后，都用这组比较，不能由单一局面外推。

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

MCGS 不是“加一个 TT”。当前实现已切换 board-key、edge-local N、幂等回传、edge-local 路径终局样本、首次绑定 edge 的 DFS 闭环截断、可达 GC 与防环 PV。
它仍是 board-key 的上下文近似，不是严格 state 合并。下一步评估 reuse/stop 边界、唯一 state 数、NN eval 数、
稳定性与 Elo；合并率或 EPS 上升本身不是采用理由。

## 后续 CPU Proof 的边界

当前 CPU 在 NN batch 等待期间仍有余量；这不是要求刻意降低 collision 的理由。未来若加入 Proof，应优先
利用 collision/等待期间未占用的 CPU 产生可验证的新 Evidence，并且不阻塞 Gather、Eval、Backprop 或长期持有
graph 锁。它可能降低 NPS，但是否采用只能由固定时间 Elo 判断；“NPS 不下降”不是硬约束。
