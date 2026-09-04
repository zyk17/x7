# Architecture

## 目标与边界

X7 研究网络 Prediction 与搜索 Evidence 在固定时间下如何协同，并只以固定时间 Elo 评价成果。
GPU 主要生产 Prediction，CPU 主要生产 Evidence；二者的具体比例、模型和 Proof 形式均可演进。

规则、训练格式和 UCI 外围曾参考 px0/Lc0；LC3、KataGo 仅作按需的公开/历史参考。X7 是独立实现，
不声称源码翻译或行为等价。

## 模块边界

- `xiangqi_core` 是规则唯一真相。棋盘、合法着、FEN、PositionHistory、重复和亚洲规则裁判均在此处。
- `engin` 拥有 UCI、ONNX backend、时钟与 stream 搜索。正式 UCI 必须有 ONNX backend；
  `UniformBackend` 仅限测试。
- `engin/src/search` 是单一的 stream 路径树。edge 首次下探时绑定 arena `NodeId`，换位不合并；
  event 自带 variation 和根历史，规则不依赖树拓扑。
- `nn` 是独立 Python 训练子项目；其输出遵守正式 ONNX 契约，但不进入规则或搜索热路径。

## 搜索不变量

- 流程固定为 `Gather -> Eval -> NN -> Eval -> Backprop`。Gather 一次只 claim 一个叶子；
  Eval 首次到达叶子时处理终局、缓存与编码，NN 只做已编码 tensor 的合批推理。
- edge 的 reservation 是 pending visit；实战的 virtual mean 为 FPU。完成或取消必须精确归还，
  completed Evidence 不包含 pending 值。
- 换根只支持向前复用已展开 child；悔棋或未展开路径直接换新 arena。stop/预算到期先取消并 drain
  所有 event，再异步回收 sibling，slot 才可复用。
- 重复、rule60 与亚洲规则是 variation/history 语义。根终局由 root gate 判断，非根叶子由 Eval
  首次分类；Gather 不重复裁决。
- 已标记 `Terminal` 的 child 不再由 Gather 选择；其首次发现仍照常 Backprop。根自身不标记为
  `Terminal`，当前搜索范围内的 root child 都已终局时停止；最终决策优先已证明必胜并选择最短 mate。
- 只维护这一套 stream 搜索，不保留 classic 对照或多轨训练格式。

## 搜索树形控制面

Select 的长期分数由算术均值利用、常规探索与证据复核组成：
`score = Q_mean + U + B_var`。这些控制面都可能改变树宽，但职责不同：

| 控制面 | 选择中的作用 | 典型树形影响 |
| --- | --- | --- |
| cPUCT | 放大所有节点的常规 `U(P, N)` | 全局、长期地更早从 Q 利用转向 policy/相对-N 探索。 |
| FPU reduction | 定义未访问 child 的初始 action-Q | 每个节点局部降低首次门槛；常在主线经过的各层先扩出兄弟。 |
| `nn_window` | 限制最多同时在途的 claim | 首要是 batch/吞吐上限；reservation 带来的分流只是受该上限约束的副作用。 |
| virtual mean FPU scale | reservation 暂时写入 `scale * FPU`，并混入 in-flight edge 的 action-Q | 碰撞时可能暂时转向兄弟；只在 reservation 存在期间生效，具体方向取决于 FPU 符号。 |
| `B_var` | `lambda * SE` 的已观察证据复核项 | 与 U 同级竞争，但只作用于 `N>=2` 的高 SE edge；不是未访问 child 的首次探索，也不保证单调扩树。 |

调参的单位是固定并发语义下的树形组合：先固定 `nn_window` 与 virtual mean，再联合扫描
cPUCT/FPU（普通探索）和 `lambda`（复核）。固定 visits 或固定时间比较根候选/PV、
访问集中度、completed evidence 的整体 SE 与 `sum(B_var)/sum(U)`；NPS、EPS 和单条 edge 的最终 N 都不是
充分结论。

## 工程约定

外部语义参考应在代码注释中标明来源和“历史参考”性质。新的搜索设计可以自研，但必须说明其目标
和不变量，不能伪装为外部引擎等价实现。运行、实验和打包操作见 [commands.md](commands.md)；已否决
的方案见 [Research.md](Research.md)。
