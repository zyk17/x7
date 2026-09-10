# Architecture

## 目标与边界

X7 研究网络 Prediction 与搜索 Evidence 在固定时间下如何协同，并只以固定时间 Elo 评价成果。
GPU 主要生产 Prediction，CPU 主要生产 Evidence；二者的具体比例、模型和 Proof 形式均可演进。

X7 由 Rust 规则核心、Rust 引擎与 Python 网络训练组成。`xiangqi_core` 是 PX0 规则实现的 Rust 重写；
`nn` 是 PX0 网络与训练格式的 Python 重写。正式模型契约固定为
`124x10x9 -> 2062 + WDL + moves-left`。

## 模块边界

- `xiangqi_core` 是规则唯一真相。棋盘、合法着、FEN、PositionHistory、重复和亚洲规则裁判均在此处。
- `engin` 拥有 UCI、ONNX backend、时钟与 stream 搜索。正式 UCI 必须有 ONNX backend；
  `UniformBackend` 仅限测试。
- `engin/src/search` 是单一的 stream 路径树。edge 首次下探时绑定 arena `NodeId`，换位不合并；
  event 自带 variation 和根历史，规则不依赖树拓扑。
- `nn` 是 Python 训练子项目；其输出遵守正式 ONNX 契约，但不进入规则或搜索热路径。

## 搜索不变量

- 流程固定为 `Select -> Expand -> Eval -> NN -> Reply -> Backprop`。Select 一次只 claim 一个叶子；
  Expand 处理终局和合法着，Eval 先查 cache、仅 miss 编码并发 NN，Reply 发布结果，NN 只合批 tensor。
- `Threads` 是固定数目的通用 worker，按就绪队列处理 Select、Expand、Eval、NN 回包和 Backprop；NN
  inference 独占一条设备线程。未来 Proof 复用同一 CPU 调度边界，不预留专用线程。
- edge 的 reservation 是 pending visit；实战的 virtual mean 为 FPU。完成或取消必须精确归还，
  completed Evidence 不包含 pending 值。
- 换根只支持向前复用已展开 child；悔棋或未展开路径直接换新 arena。stop/预算到期先取消并 drain
  所有 event，再异步回收 sibling，slot 才可复用。
- 重复、rule60 与亚洲规则是 variation/history 语义。根终局由 root gate 判断，非根叶子由 Expand
  首次分类；后续 Eval 不重复裁决。
- 已标记 `Terminal` 的 child 仍可由 Select 重选，并继续 exact Backprop；这使其结果持续影响祖先
  的 PUCT。根自身不标记为 `Terminal`，当前搜索范围内的 root child 都已终局时停止；最终决策优先
  已证明必胜并选择最短 mate。
- 只维护这一套 stream 搜索，不保留 classic 对照或多轨训练格式。

## 生命周期

- UCI 的 `Engine` 长期拥有 backend、`SearchTree`、固定 `WorkerPool`、时钟和 `GraphReaper`；
  `SearchTree` 的 root history 是当前 position 的唯一真相。它只在旧 job 停止并 drain 后换 position、
  backend 或树根。
- `Search` 是一次 job：`start -> run* -> finish`。`run` 可分阶段调用；`finish` 才请求停止并归还
  本 job 的常驻 worker。`Drop` 只是遗漏 `finish` 时的兜底。
- `WorkerPool` 不持有 job 状态。UCI 路径由 Engine 复用固定 pool；独立 `Search::start` 可自持一个
  pool，供 benchmark 与测试直接使用。每个 job 将全部队列端点交给 pool，固定 CPU worker 和独立 NN
  worker 处理至 job 完成。
- 树推进产生的 sibling 回收与整张旧 arena 释放由 Engine 在 drain 后交给 `GraphReaper` 后台执行；GC
  不参与正在运行的搜索。

## 搜索树形控制面

Select 的长期分数由算术均值利用、常规探索与证据复核组成：
`score = Q_mean + U + B_var`。这些控制面都可能改变树宽，但职责不同：

| 控制面 | 选择中的作用 | 典型树形影响 |
| --- | --- | --- |
| cPUCT | 放大所有节点的常规 `U(P, N)` | 全局、长期地更早从 Q 利用转向 policy/相对-N 探索。 |
| FPU reduction | 定义未访问 child 的初始 action-Q | 每个节点局部降低首次门槛；常在主线经过的各层先扩出兄弟。 |
| `nn_window` | 限制最多同时持有的 Eval claim | terminal/cache 与等待 claim 的 job 不占；首要是 batch/吞吐上限，reservation 带来的分流只是受该上限约束的副作用。 |
| virtual mean FPU scale | reservation 暂时写入 `scale * FPU`，并混入 in-flight edge 的 action-Q | 碰撞时可能暂时转向兄弟；只在 reservation 存在期间生效，具体方向取决于 FPU 符号。 |
| `B_var` | `lambda * SE` 的已观察证据复核项 | 与 U 同级竞争，但只作用于 `N>=2` 的高 SE edge；不是未访问 child 的首次探索，也不保证单调扩树。 |

调参的单位是固定并发语义下的树形组合：先固定 `nn_window` 与 virtual mean，再联合扫描
cPUCT/FPU（普通探索）和 `lambda`（复核）。固定 visits 或固定时间比较根候选/PV、
访问集中度、completed evidence 的整体 SE 与 `sum(B_var)/sum(U)`；NPS、EPS 和单条 edge 的最终 N 都不是
充分结论。

## 工程约定

搜索设计必须说明其目标和不变量。运行、实验和打包操作见 [commands.md](commands.md)；已否决的方案见
[Research.md](Research.md)。
