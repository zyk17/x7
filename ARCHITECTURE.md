# ARCHITECTURE

## 目标与参考

这是 px0 的 Rust 重写，不是兼容层。规则、classic 搜索和网络外围按 px0 逐函数对照；stream 只按 LC3 的公开架构文档实现等价设计，不能标称为源码翻译。

- px0：`C:\Users\Administrator\projects\px0`
- 训练参考：`C:\Users\Administrator\projects\pxzero-training`
- stream 参考：[LC3 Overview](https://lczero.org/dev/lc0/search/lc3/overview/)、[Policy](https://lczero.org/dev/lc0/search/lc3/policy/)、[Glossary](https://lczero.org/dev/lc0/search/lc3/glossary/)

## 模块

| 模块 | 职责 | 当前状态 |
| --- | --- | --- |
| `crates/xiangqi_core` | 唯一规则真相：棋盘、合法着、FEN、Position、history、RuleJudge | 已完成 |
| `crates/engin/src/search/classic` | 独立 MCTS 对照实现 | 稳定维护，不再推进共享树 TaskWorkers |
| `crates/engin/src/search/stream` | LC3-style streaming tree MCTS 与正式 UCI 主线 | 已接入 UCI，持续验证 |
| `crates/engin/src/neural` | 124-plane 编码、policy 映射、ONNX、缓存 | classic 与 stream 共用 backend 契约 |
| `nn/` | px0 record、训练、checkpoint、ONNX 导出 | 独立 Python 子项目 |

## Classic

classic 是独立对照实现。它保留 `CachingBackend`、ONNX batch、collision、prefetch、tree reuse、WDL/info、watchdog 和 `Abort + Wait` 会话边界；正式 UCI 默认使用 stream，`UniformBackend` 仅用于测试。

classic 的 `TaskWorkers` 共享可变 `NodeTree` 模型不继续迁移：它依赖 px0 C++ 的别名约定，不能在不引入 `unsafe`、raw pointer 或整树锁的条件下安全等价实现。classic 不得再引入此类并发补丁。

## Stream

stream 与 classic 隔离，不复用 classic tree、worker 或 replay/snapshot delta。

- repository 使用分片 map 和 `parent-key + move` 的 tree key；首版不做 DAG/TT。
- 事件拥有完整 root history、variation、generation 和 edge reservation。
- worker 拓扑为 Gather×4、Eval×2、NN×1、Backprop×1。Eval 处理终局、缓存、编码、合法 policy；NN 只执行 `infer_encoded` 与队列 batch。
- `SearchLimits`、generation gate、stop/drain 与 edge reservation 回收已实现。
- tree reuse 会保留已走主线及旧根，并遍历 repository 删除不可达兄弟子树；UCI/Watchdog 已输出最小 info 与一次 bestmove。
- 当前没有 MultiPV 或 multivisit；NN `m` 已进入 backup 与已证明终局距离。`draw_score` 固定为零，不做 contempt。

stream 的 selection 与 bestmove 暂以项目批准的 X7↔classic 对照为准，不是 LC3 Policy 的正式公式。

## 模型

正式契约固定为 `124x10x9 -> 2062 + WDL + moves-left`。x7 v2 基准为 `width=256`、`blocks=12`、`bottleneck_channels=112`，带两次 Global Broadcast；训练、续训和导出严格校验 checkpoint 元数据，避免模型尺寸漂移。

## 纪律

- px0 路径的新 Rust 函数必须记录连续 px0 参考区间；stream 新函数记录 LC3 URL 与对应标题。
- 找不到参考语义时记录缺口，不自行添加搜索启发式。
- `position ... moves ...` 必须保留完整历史。
- stream UCI 持续验证 `position -> go -> stop -> position -> go` 无旧 generation、无 reservation 泄漏且恰好一次 `bestmove`；时钟分配与未实现 stopper 必须保持明确边界。
