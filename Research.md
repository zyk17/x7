# Research Notes

这里记录已经完成、但未进入正式主线的研究尝试。每条记录只回答：问题是什么、为什么值得试、
实验如何做、结果是什么、还不能说明什么。它不是待办列表，也不把失败结果包装成结论。

当前正式搜索是独立的 X7 stream 路径树，不是 px0 等价实现。早期的按棋盘合并 MCGS 图实验已经
结束；其实现细节不再保留。下列实验说明：cache-only prefetch，以及“碰撞时额外保留 reservation”
的最小 virtual visit 变体未被保留；正式搜索仍保留真实异步 playout 的 in-flight reservation
作为 virtual visit（计入 edge started N）。batch-budget multivisit 也已否决，见 2026-08-17。

## 2026-08-03：cache-only prefetch 与最小 virtual visit

### 背景

stream 的 Gather 产生叶子的速度可以高于 NN 消费速度。深而窄、强制性高的局面尤其容易出现：
许多 playout 碰到同一批正在等待 NN 的叶子，正常请求不足以填满 GPU batch。较大的 GPU batch
可能提高推理吞吐，但也会延长一次 NN 调用的等待时间并增加碰撞。

这提出了一个合理但尚未证实的问题：能否在不改变最终固定时间决策质量的前提下，为这类局面
补充更多有用的 NN 工作？

### 目标

验证两项与 px0 思路相邻的工程尝试是否值得保留：

- **cache-only prefetch**：正常 NN 请求不足时，从当前 tree 按 PUCT 递归挑选未展开叶子，额外
  评估并只写入 NN cache；不创建 node、不更新 N/Q，也不直接改变 bestmove。
- **最小 virtual visit**：碰撞到同一 evaluating leaf 时，临时保留该路径的 edge reservation，
  让其 started visit 影响后续 PUCT；叶子完成、失败或 stop 后再归还。

两者都不是 px0 的完整 batch-budget multivisit：没有一次 Gather batch 的访问预算分配，也没有
一次评估按多次 visit 加权回传。

### 实验方法

- fresh tree：每次运行新建 backend、cache 与 tree，不混入实战 tree reuse。
- 本机 `data/x7.onnx`、DirectML、Gather/Eval/Backprop = `4/4/1`、目标 batch 64、`movetime=1000ms`。
- cache 打开；分别比较 `MaxPrefetch=0` 和 `32`。
- 使用不同形态局面：`initial`、`middle_01`、`middle_05`、`middle_30`、`evasion_01`、`evasion_04`。
- 观察正常完成 playout（`done`）、batch、collision 与根候选；`prefetch` 产生的 cache hit 或 EPS
  不作为收益证据，因为它们可被预取本身机械地提高。

### 结果

`MaxPrefetch=32` 的 batch 确实从约 18–26 提高到约 33–39，但正常完成 playout 没有稳定收益：

| 局面 | done：0 → 32 | 变化 |
| --- | ---: | ---: |
| initial | 6745 → 6768 | +0.3% |
| middle_01 | 2563 → 2464 | -3.9% |
| middle_05 | 2951 → 1762 | -40.3% |
| middle_30 | 2536 → 1440 | -43.2% |
| evasion_01 | 10932 → 8849 | -19.1% |
| evasion_04 | 4749 → 4434 | -6.6% |

collision 没有出现一致改善；`middle_30` 与 `evasion_04` 的根部首选还发生变化。该实验没有外部
棋力标签，不能据此断言候选变化一定更差；但没有任何直接证据表明 prefetch 提高了固定时间决策，
而它在多数代表局面明显挤占了真实 playout 的预算。

### 决定

删除 cache-only prefetch、`MaxPrefetch` UCI option、benchmark 参数和碰撞时额外保留
reservation 的最小 virtual visit 变体。正式搜索仍保留真实异步 playout 的 in-flight
reservation：它计入 edge started N（PUCT virtual visit），并保证每条在途路径的完成/取消配平；
不把一次 NN 评估视作多次访问。

### 尚未排除的解释

- 这不是“深窄局面不需要更大 batch”的结论；它只说明当前两种补 batch 方法没有证明价值。
- 单次 1 秒 fresh-tree 对比不能代替固定时间 Elo、自对弈或带外部锚点的候选质量测试。
- 如果以后有 wave gather，或 collision 时可执行的 CPU Proof 工作，应作为新假设重新设计和独立
  验证，而不是恢复本次 prefetch 实现。batch-budget multivisit 已否决，见下节。

## 2026-08-17：不采用 batch-budget multivisit

classic lc0 的 collision/terminal multivisit 是一次评估按 K 次 visit 加权回传，用来填 batch、
补偿纯 virtual visit 分流弱。X7 不采用：

1. Gather 每次采集一个叶子，不是一批；GPU 合批在 NN 队列完成。
2. 实战 `μ=FPU` virtual mean 已比纯 virtual visit 更明显且更温和。一次打入 K 份 FPU 会破坏
   这份分流，让 select 更不可靠。后续若研究分流，改 virtual mean / virtual visit，不加 K。

## 2026-08-24：时间型 verification credit（快速原型）

### 已完成的方差方向

本轮依次试过三类思路：

- **父节点方差 cPUCT**：按节点整体的 value dispersion 改 cPUCT，语义是局面变宽/变窄，不能把预算定向给
  产生冲突证据的具体 edge；与原始 cPUCT 曲线叠加后也难单独归因。
- **raw sigma edge bonus**：1) 包括仍受 prior 约束的 `P * U * (1 + k*sigma)` 2)不受 prior 约束的 rescue
  bonus。它们确实可让低 P 路线获得额外访问，但 sigma 在大量完成访问后仍可长期存在，因而会持续争抢预算；
  rescue 还需要额外上限，行为更难控。
- **时间型 credit**：不把 sigma 当作长期不确定性，而只把 Q/sigma 相对近期参考值的变化当作“新证据需要验证”。

因此未保留前两类 Select 实现，也没有保留把 LCB 写入 Select 的方案：三者都把已观察到的长期离散度误作需要永久补偿的搜索信号。根最终 decision 的 LCB/UCB 是独立实验，不参与此处讨论的树内选择。

### 原型与结果

原始 `sigma = sqrt(E[wl²] - Q²)` 是已完成样本的离散度，不是均值不确定性，也不会必然随 N 消失；
因此不再把它作为永久 PUCT/LCB bonus。当前原型在 edge 完成时比较 Q 与 sigma 的缓慢参考值：新证据
冲击会累积有限 credit，之后同 edge 的完成回传会衰减它。选择只放大该 edge 原有的 `P * U`，默认
scale 为 0。

固定 `cPuct=e`、`cpuct_factor=0` 的 fresh-tree 扫描 `scale=0/0.5/1/2/4`：5 步杀例中，非零
scale 都比 0 更早切入主杀线；`middle_01` 中低 prior 的 `f3f7 (P=0.051)` 在 0/0.5/1 下于 15k
成为第一，但 2/4 被高 prior `e0e1` 压回。稳定路线即使保留较高 sigma，其 credit 也下降到约
0.001–0.01。结果支持“credit 有时效”这一机制，不支持当前常数、衰减率或 scale 已可进入正式默认。

## 2026-09-01～03：recent-Q 失败与 SE verification bonus

### recent-Q 失败结论

下述 `Q_fast/Q_select` 是已删除的历史实现，不是当前搜索公式。强化到 `a=8,T=80` 后，它确实能改变早期树形；但低 prior、深收益晚显现的分支会因早期暂时偏低的 Q 更快失去访问。它会同时加速近期好证据和近期坏证据，无法区分普通反证与尚未看深的正确分支。

因此删除 `Q_fast`、`Q_select`、`ValueUpdateRate` 与 `FreshQVisits`。结论不是未来不能加权 Q，而是权重必须来自当时可解释的证据质量（例如 terminal/proof 覆盖），不能来自样本序号。未来若引入质量加权，`Q_mean` 仍保持原始 completed evidence 的算术均值真相；新估计必须有独立质量定义与不破坏渐近收敛的证明/实验。

### 已删除 recent-Q 的目标与公式

当时的设想是：小 completed-N 时，算术均值会让一条已被选中的高 Q 边对新证据反应过慢；但把近期样本长期替换
算术均值会破坏大样本收敛。因此曾保留原始算术 `Q_mean`，另维护近期加权的 `Q_fast`，仅在 Select
使用：

`Q_select = Q_mean + g(N; T) * (Q_fast - Q_mean)`，其中 `g(N; T) = max(0, 1 - N/T)`。

`ValueUpdateRate=a` 控制 `Q_fast` 的近期权重，`FreshQVisits=T` 控制它的有限生命周期；`N >= T`
后严格回到 `Q_mean`。这不是改写回传的原始统计，`Q_mean` 仍是全部 completed evidence 的算术均值。
每次完成样本 `w` 时，令旧 completed-N 为 `N`，则
`eta = a/(N+a)`，`Q_fast <- (1-eta) Q_fast + eta w`；因此 `a=1` 恰为算术均值，较大的 `a` 更重视近期样本。

方差方向的目标则不是救回低 prior 招法，而是暂时给已出现分歧证据的 edge 更多复核，以更快降低
该 edge 的均值标准误。当前保留独立项：

`B_var = lambda * SE`（仅 `N >= 2`）。
历史公式为 `score = Q_select + U + B_var`；当前公式为 `score = Q_mean + U + B_var`。

`SE` 是该 edge 的原始 completed `wl` 样本均值的经验标准误：
`Q_mean = sum(w_i)/N`，`std = sqrt(max(0, sum(w_i^2)/N - Q_mean^2))`，
`SE = std/sqrt(N)`。它不读 prior、父 N、reservation 或 depth。
所以未访问/只有一个样本的 edge 仍只由通常的 FPU/PUCT 获得机会；停止回访时 `B_var` 不会像 U 一样
随 sibling 访问增长。即使局面的原始 `std` 持续存在，`SE=std/sqrt(N)` 仍会随自身 evidence 自然趋近 0，
不再人为 cap 高 SE 或设停止阈值。

### 局部验证与边界

用 `Gather=1`、`Eval=1`、batch=1、`nn_window=1` 的 fresh tree，检查根部 trace 和 tree funnel。
在 `proof_draw_01`、`proof_mate_01`、`proof_variance` 与 `middle_37` 中，`lambda * SE` 可在早期给
高 SE、已有 evidence 的边额外访问。同配置 repeat 的根部 trace 可复现。原始的 `U * (1 + k * SE)`
路径因混淆“PUCT 探索”和“复核证据”已移除。

2026-09-03 的正常并发 trace 中，强化 `lambda=1.0` 已能显著改变早期树形：`proof_mate_01` 的低 prior `i3e3` 在 12k 时成为访问第一，而基线仍为第二；但在 `e2e1` 中只是增加早期访问，尚未更早触发深收益。故 Bvar 保留为实验项，尚未声称固定时间候选质量或 Elo 改善。

## 2026-09-03：基础树形参数候选（x7 / DirectML）

为先隔离基础树形，固定 virtual mean FPU scale=1、关闭 Bvar，使用 fresh tree、`Gather=3`、`Eval=5`、
batch=64。初始局面与低 prior `e2e1` 样本共同表明：

- `nn_window=1.25`（claim 上限 80）平均 batch 约 34，初始局面约 6.2k–6.5k NPS；`nn_window=2`
  可将 batch 填至约 61，却降到约 3.5k NPS、并把 collision 从约 45% 推至约 54%。本机候选取 1.25，
  不能把“满 batch”单独当作收益。
- 常数 `cPUCT=0.75` 在初始局面过窄（4k 时 top-1 约 66%、top-3 约 86%）；常数 1.25 虽约为
  52%/77%，却让 `e2e1` 在 10k 停在约 140 visits。常数 `cPUCT=3.0` 在初始局面为约 40%/65%，
  仍有清晰的两层主干；且 `e2e1` 在 4k 已反超、约 1,170 visits，10k 为约 6,820、Q 约 0.44。
  因而下一阶段先用 `CPuct=3.0, CPuctFactor=0`，不让增长曲线掩盖初始探索强度。
- `FpuReduction=.33` 仍能让 `e2e1` 在 4k 反超，但约 950 visits，低于 `.25` 的约 1,170；候选为 `.25`。
- 在该基础树形中，`lambda=.1` 通常只有约 `.003` 的单 edge bonus，难以影响选择。`.5` 在
  `proof_mate_01` 的两个 8k repeat 中均让 `i3e3` 获得更多访问（约 4.2k 对约 3.6k）且更高 Q
  （约 .56 对约 .52）；但未改善 `proof_variance_01` 的 `h8g8`。因此 `.5` 是可继续复验的 Bvar
  候选，而非已确定默认；需要固定时间候选质量与 Elo 验证。

该候选组合为：`CPuct=3.0`、`CPuctFactor=0`、`FpuReduction=.25`、`NnWindow=1.25`、
`VirtualMeanFpuScale=1`，Bvar 暂列 `.5` 候选。它专门解决早期低 prior 证据不足；不声称能解决
AND/OR terminal proof 或残局知识缺失。

## 2026-09-03：常数 cPUCT / 并发窗口与 additive Bvar 复验（x7 / DirectML）

后续固定时间实战提出 `CPuct=2.25`（无增长曲线）、`FpuReduction=.225`、`NnWindow=2.25`、
`VirtualMeanFpuScale=1` 的候选。以 batch=64、Gather=3、Eval=5，在固定 visits 与 2 秒固定时间
各三次 fresh-tree repeat 复验后：

- `NnWindow=2.25` 已能保持约 40--50 的平均 batch；`2.5` 略大但也稳定增加 claim/collision。collision
  不是单独的优劣指标：它同时反映主线集中度、virtual mean 的排斥强度、窗口上限及 NN 延迟；必须与
  固定时间的根部行为一起解释。
- 低 prior `e2e1` 样本中，`FpuReduction=.15/.225` 在 10k 均可使 `e2e1` 反超（约 5k--7k visits），
  `.30/.35` 则稳定只给约 0.3k--0.5k visits，仍由高 prior `f6f3` 主导。因此当前基础候选保留 `.225`，
  不再把 `.30+` 作为同等候选。
- 对 `B_var=lambda*SE` 扫描 `0/.5/1/1.5/2.25`：在上述基础参数下，2 秒低 prior `e2e1` 样本的
  `lambda=0` 三次均反超（约 5.6k--6.5k visits）；从 `.5` 起三次均未反超，`.5/1/1.5/2.25` 的
  `e2e1` 访问约为 0.4k--3.1k。强制杀 `proof_mate_01` 各档仍收敛 `i3e3`，但 lambda 增大没有使其
  更快聚焦；高波动 `proof_variance_01` 的 total SE 有时下降，但主线 `d8c8` 的 Q 也从约 `.984`
  降至 `.971--.980`。

