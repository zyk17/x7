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

因此未保留前两类实现，也没有保留 LCB：三者都把已观察到的长期离散度误作需要永久补偿的搜索信号。

### 原型与结果

原始 `sigma = sqrt(E[wl²] - Q²)` 是已完成样本的离散度，不是均值不确定性，也不会必然随 N 消失；
因此不再把它作为永久 PUCT/LCB bonus。当前原型在 edge 完成时比较 Q 与 sigma 的缓慢参考值：新证据
冲击会累积有限 credit，之后同 edge 的完成回传会衰减它。选择只放大该 edge 原有的 `P * U`，默认
scale 为 0。

固定 `cPuct=e`、`cpuct_factor=0` 的 fresh-tree 扫描 `scale=0/0.5/1/2/4`：5 步杀例中，非零
scale 都比 0 更早切入主杀线；`middle_01` 中低 prior 的 `f3f7 (P=0.051)` 在 0/0.5/1 下于 15k
成为第一，但 2/4 被高 prior `e0e1` 压回。稳定路线即使保留较高 sigma，其 credit 也下降到约
0.001–0.01。结果支持“credit 有时效”这一机制，不支持当前常数、衰减率或 scale 已可进入正式默认。

