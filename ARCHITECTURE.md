# ARCHITECTURE

## 1. 项目边界

这是一个围绕 **MCTS + 小型 policy/value 网络** 的象棋项目。

当前只保留四块基础设施：

1. 规则内核
2. MCTS 搜索引擎
3. 数据集生产与搜索标注
4. policy/value 训练与导出

当前明确不做：

- Alpha-Beta 主路线
- 复盘语义头
- 多格式长期并存
- 分布式训练平台

## 2. 模块职责

### `crates/xiangqi_core`

职责：

- FEN / 局面表示
- 合法着生成
- do / undo
- 终局判断

约束：

- 只处理规则真相
- 不混入搜索策略
- 不混入训练逻辑

### `crates/engin`

职责：

- MCTS 节点与树结构
- selection / expansion / backup
- policy/value ONNX 推理接入
- UCI 入口
- benchmark / 最小调试统计

约束：

- 搜索核心尽量同时服务线上走棋与离线搜索标注
- 不承担数据集打包职责
- 搜索预算口径统一为 `playouts / nodes / deadline`

### `crates/xiangqi_dataset`

职责：

- PGN 解析
- canonical 词表生成
- PGN -> XRSH
- 对人类局面跑 MCTS，并把搜索标签写回 XRSH

约束：

- 维护者工具，不塞进用户引擎进程
- 只保留一个正式训练格式

### `nn/`

职责：

- `policy` 训练
- `value` 训练
- 搜索 visit 分布蒸馏
- checkpoint / ONNX 导出

约束：

- 不重新实现规则
- 不重新实现搜索
- 只消费物化后的样本

## 3. 当前网络设计

当前网络只做：

- `policy logits`
- 可选 `value logit`

结构上保持简单：

- shared ResNet trunk
- 全局池化
- `fc` 输出 policy
- 可选 `fc_value` 输出 value

当前不保留：

- `danger`
- `attack`
- `tactical`
- 额外复杂 head 组合

## 4. 数据契约

当前正式格式：**XRSH v5**

设计目标：

- 同时容纳人类 policy 样本与搜索标注样本
- 不引入 proto / 多平台交换层
- 本地单机训练优先

单条样本字段：

- `fen`
- `root_fen`
- `uci_prefix`
- `target_idx`
- `legal_idx`
- `ply`
- `game_result_red`
- `ply_total`
- `search_q`
- `search_visits`
- `search_counts`

约定：

- 没有搜索标注时，`search_visits == 0`
- `search_counts` 与 `legal_idx` 对齐
- `search_q` 为当前行棋方视角 value 标签
- `search_visits` 当前记录搜索 playout 数

## 5. 当前 UCI / Benchmark 语义

- `setoption name Playouts` 是默认搜索预算入口
- `setoption name Visits` 仅作兼容别名
- `go nodes` 对应树总节点预算
- `go movetime` 对应时间预算
- `go infinite` 保留支持
- `go depth` 目前明确不支持，不做伪兼容
- `info` 与 benchmark 统一输出：
  - `playouts`
  - `root_visits`
  - `nodes`
  - `nps`

## 6. 训练主线

第一阶段只做 `policy + value + MCTS` 闭环。

推荐顺序：

1. 用大师棋谱打稳 `policy`
2. 对人类局面跑 MCTS，得到 `search_q + visit distribution`
3. 用 `search_q` 训练 `value`
4. 视情况用 `search_counts` 辅助蒸馏 policy
5. 小规模、分轮次、自对弈补冷门局面
6. 受控混合人类数据与搜索数据继续训练

## 7. 自对弈原则

自对弈不是海量主数据源，而是受控增量源。

原则：

- 按轮次推进
- 每轮数据量不大
- 优先补冷门局面
- 优先补 value 信号
- 始终让人类数据保持锚点

## 8. 稳定文档

长期稳定文档只保留：

- `README.MD`
- `ARCHITECTURE.md`
- `AGENTS.md`

临时文档只保留：

- `NextStep.md`
- `TODO.md`
- `temp.md`
