# AGENTS

面向在本仓库内工作的自动化助手。

## 1. 开始前先读

1. `README.MD`
2. `ARCHITECTURE.md`
3. `AGENTS.md`
4. `NextStep.md`
5. `TODO.md`
6. `temp.md`

## 2. 当前项目共识

- 这是新项目，不背历史方案包袱
- 搜索主路线只有 **MCTS**；旧 MCTS 已删除，当前处于按 lc0 classic 重建阶段
- 网络只保留 **policy + value**
- 主训练数据路线改为 **px0 / lc0 风格外部数据**
- 当前正式模型契约是 **124x10x9 -> 2062 + WDL**
- 当前 value 主监督是 **WDL + qMix**
- 当前 `q_ratio` 采用 **px0 风格固定标量 / 分阶段切换**
- 当前默认 baseline 容量为 `10x160`
- `engin` 线上输入主线是 **真实 history + fen_only fallback**
- 仓库内已移除 `XRSH` 正式训练/标注链路
- 不再把本地慢速搜索标注当主数据生产方式
- 不再把 `15 planes + move_vocab + scalar value` 当长期正式 I/O
- 当前引擎与搜索基建的**唯一主参考**是：
  - `C:\Users\Administrator\projects\lc0`
  - `C:\Users\Administrator\projects\lczero-training`

## 3. 当前代码边界

- `crates/xiangqi_core`：规则核心
- `crates/engin`：history、ONNX、policy 映射、最小 UCI，以及待重建的 MCTS
- `nn/`：训练与导出

不要再往仓库里放：

- Alpha-Beta 主线代码
- 旧 MCTS 的兼容层、旧搜索统计、旧 benchmark
- 多套正式训练格式长期并存
- 旧 XRSH 训练/标注链路
- 为“未来社区平台”提前做的大抽象

## 4. 代码规范

- 优先复用，不重复造轮子
- 优先简洁、直接、可读
- 这是高频系统，但前期先保可读性
- 性能优化应在清晰实现上定点推进
- 避免重复分配、重复推理、重复序列化
- Python 只做训练与离线数据，不搬规则热路径
- 搜索恢复后，搜索 / benchmark / UCI 的统计口径必须一致
- `position ... moves ...` 不允许再退化成“只保留最终局面”
- MCTS 在重建完成前，`go` 必须明确返回不可搜索状态；不得返回 heuristic 或旧树结果

## 5. 参考纪律

- 引擎基建、搜索语义、UCI 行为、训练数据主链路，默认必须先对照：
  - `C:\Users\Administrator\projects\lc0`
  - `C:\Users\Administrator\projects\lczero-training`
- 目标是**一比一还原主线语义**，禁止先凭主观理解自创实现，再用补丁修语义。
- 旧 `crates/engin/src/mcts/` 不得作为实现参考或兼容目标；新的 Rust 函数必须注明对应 lc0 函数/连续行区间。
- 禁止“名字像 lc0，行为不是 lc0”的接口。
- 禁止在未确认参考实现前，对搜索主循环、预算、统计、batch、stop、tree reuse 做主观改造。
- 如果中国象棋与国际象棋存在规则差异，必须先确认差异点，再做最小必要偏离；不能借“规则不同”提前扩大发挥空间。
- 任何后续修改，必须在变更说明、`NextStep.md`、`TODO.md` 或 review 记录中，附上对应参考代码的：
  - 文件路径
  - 行号
  - 本仓库对应文件
- 现代 `lc0` 的主参考落点不是旧 `lczero` 的 `UCI.cpp / UCIOption.cpp` 扁平文件，而是优先对照：
  - `C:\Users\Administrator\projects\lc0\src\engine_loop.cc`
  - `C:\Users\Administrator\projects\lc0\src\engine.cc`
  - `C:\Users\Administrator\projects\lc0\src\chess\uciloop.cc`
  - `C:\Users\Administrator\projects\lc0\src\search\classic\params.cc`
  - `C:\Users\Administrator\projects\lc0\src\search\classic\search.cc`
- 如果暂时找不到参考代码位置，不要直接实现；先把缺口记录到计划文档。

## 6. 文档规则

稳定文档只保留：

- `README.MD`
- `ARCHITECTURE.md`
- `AGENTS.md`

临时文档只保留：

- `NextStep.md`
- `TODO.md`
- `temp.md`

`docs/` 不承担正式文档角色。
