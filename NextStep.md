# NextStep

当前阶段开始收敛，优先按 `lc0 / px0` 风格把规则层和搜索地基做稳。

当前正式共识：

- 模型契约保持：`124x10x9 -> 2062 + WDL`
- 训练主线改为：纯 CNN `KataGo` 风格 trunk，不再保留 `attention policy`
- 引擎输入保持：真实 `history` 主线，`fen_only` fallback
- 搜索主线保持：`MCTS`
- 当前已支持：单线程主线 + 最小 shared-tree 多线程
- 当前明确不做：
  - `MultiPV`
  - 复杂 ONNX backend / evaluator 池
  - 本地大规模搜索标注主线

下一阶段只做三类高价值事项：

## 1. 继续对齐 `xiangqi_core` 规则真相

目标：

- 保持规则层和 `pyffish / px0` 可对拍
- 优先消灭会直接污染搜索的规则误差

优先顺序：

1. 持续补 `rules_regression`
2. 对可疑局面做合法着对拍
3. 只修规则真相，不往规则层掺搜索策略

完成标准：

- 关键局面合法着与参考实现一致
- 回归测试能固定住已知坑点

## 2. 继续收紧 `engin` 搜索语义

目标：

- 尽量和 `lc0 / px0` 的主线语义保持一致
- 在不引入大后端系统的前提下，把搜索行为做对

优先顺序：

1. 继续完善 root tree reuse / subtree reuse
2. 继续核对 `playouts / nodes / depth / seldepth / pv` 统计口径
3. 继续检查 repetition / terminal 处理
4. 用固定命令量化 `MctsWorkers` 和 `MctsBatchCap`

完成标准：

- `bestmove` 稳定合法
- `pv / seldepth / nps` 变化可解释
- GUI 联调下不再出现明显 UCI 行为异常

## 3. 优先把新网络 baseline 重新训起来

目标：

- 把新的纯 CNN trunk 跑通
- 把 `px0` 里已有的 `q / visits / policy_kld / plies_left` 用起来
- 在不扩正式引擎 I/O 的前提下，把 value 学习做稳

优先顺序：

1. 跑新 trunk 的短训与导出
2. 看 `policy / value_ce / value_q_mse / moves_left` 是否同时下降
3. 再决定 `q_ratio` 的阶段式训练策略

完成标准：

- 新网络能稳定训练和导出
- 评估结果不再受旧 `attention policy` 包袱干扰
- 下一轮长训配置清楚

## 当前明确先不做

1. 不做 `MultiPV`
2. 不做复杂 ONNX 推理后端优化
3. 不把训练辅助头扩成正式引擎输出契约
4. 不为了未来复盘系统提前堆抽象
