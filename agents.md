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
- `value` 主监督优先是 **search_q**
- 自对弈只做小批量、分轮次、受控混合
- 正式训练格式只保留 **XRSH v5**

## 3. 当前代码边界

- `crates/xiangqi_core`：规则核心
- `crates/engin`：MCTS、ONNX、UCI
- `crates/xiangqi_dataset`：词表、PGN、XRSH、搜索标注
- `nn/`：policy/value 训练与导出

不要再往仓库里放：

- Alpha-Beta 主线代码
- 复盘语义头代码
- 多套正式数据格式
- 为“未来社区平台”提前做的大抽象

## 4. 代码规范

- 优先复用，不重复造轮子
- 优先简洁、直接、可读
- 这是高频系统，但前期先保可读性
- 性能优化应在清晰实现上定点推进
- 避免重复分配、重复推理、重复序列化
- Python 只做训练，不搬规则热路径

## 5. 文档规则

稳定文档只保留：

- `README.MD`
- `ARCHITECTURE.md`
- `AGENTS.md`

临时文档只保留：

- `NextStep.md`
- `TODO.md`
- `temp.md`

`docs/` 先不承担正式文档角色。
