# ARCHITECTURE

## 项目目标

> 棋力不是来自网络，也不是来自搜索。棋力来自 Prediction 与 Evidence 的协同；Knowledge 负责产生 Prediction，Proof 负责产生新的 Evidence，Decision 在固定时间预算下融合二者，形成最终行动。

项目目标是在有限计算资源、固定思考时间下，研究 Knowledge（Recognition）与 Proof（Calculation）如何协同，获得最高的固定时间 Elo。ResNet、Transformer、MCTS、Alpha-Beta、CPU/GPU 分工和辅助 head 都是实现选择，不是独立目标。

项目不复制人、Lc0 或 Stockfish 中的任一种实现；它研究象棋在现代硬件约束下的成本模型：GPU 主要生产 Prediction，CPU 主要生产 Evidence，两种资源不应被简单视为零和。

## 研究坐标

- **Knowledge**：从数据学习的预测能力，包括 pattern、局面、战略、value 与 policy。
- **Proof**：计算得到新增证据的过程，不限于 MCTS、Alpha-Beta 或 proof search。
- **Evidence**：Proof 的产物。终局、将杀和强制线是强证据；有限搜索的 Q、访问统计和 PV 是较弱证据，不等同于逻辑证明。
- **Decision**：在固定时间内结合预测与证据选择着法；Proof 不把网络知识“改写”为另一套知识。

这一坐标系同样适用于人、Lc0 与 Stockfish：三者都有 Knowledge 与 Proof，差别只在于二者的来源、实现与成本。人以长期学习和心算协同；Lc0 以 NN 和 MCTS 协同；Stockfish 以 NNUE/启发式和 Alpha-Beta 协同。

## 参考边界

第一版实现参考过 px0 / Lc0 / LC3；**当前仓库是独立的 X7 引擎，不是 px0 的 Rust 重写，也不承诺搜索行为等价**。

外部材料只作历史或语义参考，不能替代项目目标：

- 规则、classical 编码、UCI 外围与训练 record 的历史工程参考：`C:\Users\Administrator\projects\px0`、`C:\Users\Administrator\projects\pxzero-training`。
- stream / MCGS 可参考 [LC3 Overview](https://lczero.org/dev/lc0/search/lc3/overview/)、[Policy](https://lczero.org/dev/lc0/search/lc3/policy/)、[Glossary](https://lczero.org/dev/lc0/search/lc3/glossary/)；本地没有 LC3 源码，不得标称为源码翻译或行为等价。
- KataGo 按需参考：本地源码 `C:\Users\Administrator\projects\KataGo`（如 `docs/GraphSearch.md`、NN cache、部分搜索细节）；不是每次改搜索都必读，也不承诺行为等价。
- 相对早期 px0/Lc0 风格 stream 基线，本实现有 multivisit 与固定两 batch gather 窗口：PUCT 使用 edge in-flight reservation 作为 pending visit（计入 started N）；当前实战还以 `μ=FPU` 暂入 action Q，完成或取消时精确归还。它不进入 completed Evidence 或 shared node Q。每个已展开节点重新分配进入它的 pending visit，未展开叶子才合并同一路径的份额。当前 batch 进入 NN 后最多准备一个后继 batch；同轮 collision 在该轮结束时取消 reservation。没有无界 prefetch 或多轮提前 gather。

## 当前认识

### 当前事实

GPU evaluation 昂贵；象棋的 NN cache 重用有限，因此单次 evaluation 珍贵；开中局的有限预算通常只能获得有限新增信息；网络已学习大量棋型、局面和局部战术，CPU 不应重复实现人类 HCE。最终目标只能是固定时间 Elo，EPS、NPS、参数量和 collision 都只是诊断指标。

### 待验证假设

更强 Knowledge 是否通常比更高吞吐更能提高象棋固定时间 Elo；困难局面是否主要是 Knowledge 不足且短时间内 Proof 也不足的局面；Knowledge/Proof 的占比是否由局面的可证明性而非开中残局阶段决定，均仍是待验证假设，而不是设计前提。

NN 只负责 Knowledge Representation：学习 policy、最终 WDL 与 moves-left Prediction，不模拟搜索、
不承担 Proof。网络应跟进 Lc0、KataGo 已验证有效的实践，而非把发明网络结构作为项目研究；训练期
辅助 target 可以帮助共享表示，但不以增加正式推理 head 为目标。项目的研究重点是 Knowledge 与 Proof
如何协同。网络结构、loss、吞吐和参数量均是诊断或实现选择；网络是否保留只以它对固定时间 Elo 与
搜索可用 Evidence 的实际贡献判断。

### 开放问题与准入

后续研究只围绕四个开放问题：Knowledge 与 Proof 的最佳比例；何种局面应由谁主导；Proof 相对网络到底能提供多少新增信息；二者如何共同决定固定时间 Elo。每个新想法都必须回答：它增强 Knowledge 还是 Proof、是否制造了新的 Evidence、是否能提高固定时间 Elo。回答不了的工作只能作为工程优化，不进入长期主线。

## 模块

| 模块 | 职责 | 当前状态 |
| --- | --- | --- |
| `crates/xiangqi_core` | 唯一规则真相：棋盘、合法着、FEN、Position、history、RuleJudge | 已完成 |
| `crates/engin/src/search` | stream MCGS；`graph.rs` 直接包含 node、edge、repository 与跨回合 graph reuse | `feat/mcgs` 研究实现 |
| `crates/engin/src/search/time.rs` | 单一 stream 的固定中性时钟分配 | 已接入 UCI |
| `crates/engin/src/neural` | 124-plane 编码、policy 映射、ONNX、缓存 | stream 使用的 backend 契约 |
| `nn/` | 训练数据格式沿用 px0 record；训练、checkpoint、ONNX 导出 | 独立 Python 子项目 |

## 单一搜索

仓库只维护 stream 搜索。`Engine` 直接拥有 graph、worker pool 与每次 job，不保留 `SearchBase`、`SearchFactory`、`SearchSession` 或 classic 对照实现。`UniformBackend` 仅用于 stream 测试；正式 UCI 必须加载 ONNX。

`search/time.rs` 独立于 graph/worker：只在 Engine 启动 job 时计算 deadline、在 drain 后归还未用时间；
它不是第二套搜索实现，也不提供策略化调参。

## Stream

- repository 是一个 64 分片的 key-value map。普通 MCGS 只以棋盘（含行棋方）作为共享 node key；每条 edge 仍保存自己的 action N/in-flight。没有单独 TT。真实 variation 第一次重复时，以该重复局面为根进入 `ContinuationTree`：其 key 纳入自最近零化着以来的规则 history，重复上下文内不换位合并，只复用按棋盘索引的 NN cache；Graph → Tree 的入口 edge 不绑定 contextual child。Tree 内吃子等零化 edge 的 child 立即回到普通 Graph，因此本回合已经展开的零化后子图可跨回合复用，但 Tree 先前的 N/Q 不迁移。第三次出现同一局面才由 `RuleJudge` 终局。首次绑定普通 shared edge 若 DFS 发现会形成 shared-Q 图环，则永久标为 topology-pruned 并从 PUCT 排除，不写 N/Q，也不伪装成棋规和棋。这是为保持 shared-Q 无环的 X7 结构近似，须由残局回归与 Elo 验证。
- NN cache 使用同一 board key，不纳入完整 history 或 repetition；容量是 KataGo 风格的 `2^NNCacheSizePowerOfTwo` 直映表，槽冲突由后写结果覆盖。它只缓存 Prediction，不参与路径规则裁决。
- 事件拥有完整 root history、variation、generation 和 edge reservation。
- Engine 直接常驻 Search×4、Eval×4、NN×1；UCI `Threads` 只在 Search/Eval 间近似平分，下一次 `go` 必要时重建 pool。Search worker 优先处理回传，空闲时才 gather；没有独立 Backprop worker。每次 `go` 只下发独占 job（新的 queues、generation、root/graph view），drain 后 worker 回到等待。Eval 处理终局、缓存、稀疏编码、合法 policy；NN 做 ORT 前 expand、稀疏合批推理，并以整批 `EncodedBatch` 交回。
- `SearchLimits`、generation gate、stop/drain 与 edge reservation 回收已实现；UCI 时钟在
  Engine 启动 job 时按固定中性的时间预算（历史上参考 px0 legacy stopper）转换为不可变 deadline，job drain 后才归还剩余时间。
- graph reuse 只保留当前 root 可达图：确认走子后，旧 root 的 sibling 图由后台 mark/sweep 回收，UCI 可立即启动新
  `go`。完整 `PositionHistory` 仍由 UCI `position ... moves` 提供；悔棋回到旧局面会重新建立搜索 root，不承诺复用
  旧 sibling 子树。后台 GC 以 topology 写锁与 node 创建/edge 绑定同步，避免删掉刚由 transposition 接回的 node。
  当前不做 edge 入度引用计数：多父图下仍须维护解绑，且“从当前 root 可达”才是保留语义。无关 `position` 换图时，
  整个旧 repository 同样在后台逐 shard 释放。search owner 已输出最小 info 与一次 bestmove。
- `MultiPV` 只在 search owner 的 root snapshot 中按既有 bestmove 排名输出多条 PV，不改变 graph、PUCT、worker 或 visit 分配。每个 Search batch 分配至多 `MiniBatchSize` 份 logical visit；在每个已展开 node 连续执行 PUCT，并把该 node 收到的预算分给多个子边，未展开叶子才合并同一路径份额。batch-local PUCT 的 U 根号项使用 child 的 started N，故 pending reservation 同时表示父节点已分配预算；当前实战的 virtual mean 仅在 reservation 未完成时进入 action Q，完成后 Q 仍只来自 completed Evidence。同 batch collision 会在该 batch 结束时取消 reservation，不额外改变 completed `N/Q`；未来是否把这段 CPU 时间用于 Proof 是研究问题，而不是当前 MCGS 的既定策略。一个物理 leaf 可以展开为多个 logical visit：它们共同决定 action N/Q，但根 LCB 单独记录物理 leaf observation，不能把同一次 NN 结果误当成多份独立证据。`MiniBatchSize` 限制一个 batch 的逻辑预算上限，`0` 使用 backend 建议值；实际 NN batch 可能因合并叶子、终局、缓存或 collision 小于该值。它可能改变 collision 和固定时间棋力，须以对拍验证。NN `m` 已进入 backup 与已证明终局距离。`draw_score` 固定为零，不做 contempt。

stream 的 selection 使用本仓 PUCT / N-Q-P 形状（历史上参考过 px0 公式，不是 LC3 Policy 的正式公式，也不是 px0 搜索等价实现）。当前 UCI 暴露 `CPuct`、`CPuctBase`、`CPuctFactor` 与 `FpuReduction`；默认 `1.75 / 40000 / 4.0`。所有 node 共用一条对数 cPUCT 曲线，`C(0)=1.75`、`C(50k)≈5`，不维护根专用初值或参数。`CPuctBase` 决定增长何时显著，`CPuctFactor` 决定增长幅度；固定时间 Elo 仍待配对对局确认。`FpuReduction=0.200`：小网络可能有系统性偏差，未知候选应较早获得首次 Evidence；LC0 对照值 `0.330` 与原 X7 `1.0/0.220` 均保留为实验基线。

根最终 Decision 额外使用最小 LCB：仅在非终局根候选的 completed N 达到 N 第一候选的 `15%` 后，按其 shared-Q 一、二阶矩计算 `LCB = Q - 5·标准误`；`LcbStdevs=0` 退回 N→Q→P。它不改变 PUCT、FPU、edge N、MCGS 回传或 worker。二阶矩按 shared child 的当前 Q² 与 path-local leaf 的 WDL² 递归重算；LCB 的样本量用 root edge 物理 leaf 的加权有效样本数 `N² / Σ(weight²)`，而不是 logical N，因此转置 evidence 或同一次 NN evaluation 展开的 K 个 logical visit 都不会伪装成独立 action evidence。该形状借鉴 KataGo `cpp/search/searchhelpers.cpp` 的 LCB 保守半径和最小访问筛选，但未移植其围棋 utility、weight 或 play-selection 机制。参数调整必须以固定节点质量锚点与固定时间 Elo 验证。

## 模型

正式契约固定为 `124x10x9 -> 2062 + WDL + moves-left`。Eval 编码产出与 px0 同系的稀疏
`InputPlane{mask,value}`。DirectML：host expand 后 `Session::run`；TensorRT：GPU expand + IoBinding。
CNN 对照基线为
`width=384`、`blocks=15`、`bottleneck_channels=192`，带两次 Global Broadcast。v3 开始试验
PX0/Lc0 AttentionBody：90-token MHA、Smolgen attention bias、DeepNorm residual scale、LayerNorm、FFN 与 from-to policy，
默认 `width=512`、`blocks=12`、`heads=16`、`ffn=768`。v3 使用 PX0/Lc0 AttentionBody
（MHA + Smolgen + LayerNorm + FFN）。训练期另有 Auxiliary Soft Policy 与 root-WDL
辅助头；二者不进入 ONNX。CUDA 训练和导出均为 FP16 trunk、FP32 heads/outputs；训练、续训和导出
校验 checkpoint 的关键架构元数据，避免模型尺寸漂移。

## 纪律

- 借鉴外部语义时保留来源标注（px0 路径、LC3 URL、KataGo 本地路径/文档等），并写清是历史参考还是本仓已偏离。
- 允许本仓自研搜索决策；不要把尚未验证的启发式伪装成“外部参考要求”。
- `position ... moves ...` 必须保留完整历史。
- stream UCI 的运行边界是 `position -> go -> stop -> position -> go`：旧 generation 必须不再输出，所有 reservation 必须在 drain 后归还，正常 `go` 恰好输出一次 `bestmove`。
