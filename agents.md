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
- 搜索主路线只有 **MCTS**
- 网络只保留 **policy + value**
- 主训练数据路线改为 **px0 / lc0 风格外部数据**
- 当前正式模型契约是 **124x10x9 -> 2062 + WDL**
- 当前 value 主监督是 **WDL + qMix**
- 当前默认 `q_ratio=0.0`，即最终结果 WDL 优先
- `engin` 线上输入主线是 **真实 history + fen_only fallback**
- 仓库内已移除 `XRSH` 正式训练/标注链路
- 不再把本地慢速搜索标注当主数据生产方式
- 不再把 `15 planes + move_vocab + scalar value` 当长期正式 I/O

## 3. 当前代码边界

- `crates/xiangqi_core`：规则核心
- `crates/engin`：MCTS、ONNX、UCI
- `crates/xiangqi_dataset`：最小 PGN / 自对弈预处理地基
- `nn/`：训练与导出

不要再往仓库里放：

- Alpha-Beta 主线代码
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
- 搜索 / benchmark / UCI 的统计口径必须一致
- `position ... moves ...` 不允许再退化成“只保留最终局面”

## 5. 文档规则

稳定文档只保留：

- `README.MD`
- `ARCHITECTURE.md`
- `AGENTS.md`

临时文档只保留：

- `NextStep.md`
- `TODO.md`
- `temp.md`

`docs/` 不承担正式文档角色。
