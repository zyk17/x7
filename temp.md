当前临时记录只保留与现实现一致的内容。

# 当前临时共识

- 搜索主线仍然是 `MCTS`
- 正式模型契约仍然是 `124x10x9 -> 2062 + WDL`
- 当前网络主线已经改成：
  - `Pre-Activation ResNet`
  - `Global Pooling Residual`
  - `pure CNN policy head`
  - `WDL value head`
  - train-only `moves-left` aux head
- 当前训练主数据源是 `px0 v6`
- 当前训练已经接入 `best_q / visits / policy_kld / plies_left`
- 当前引擎正式消费的仍然只有：
  - `policy logits`
  - `WDL -> q = W - L`

# 当前网络与搜索的真实分工

- 网络负责给出：
  - policy 先验
  - WDL value
- 搜索负责：
  - 基于 policy 做扩展排序
  - 基于 value 做 backup
  - 通过 `in_flight / collision / multivisit` 稳定批量搜索

# 当前不再采用的旧想法

下面这些只视为历史草案，不再代表仓库主线：

- 非 `MCTS` 的 `Priority Frontier Search`
- `WDL + scalar` 双正式 value 输出
- `opponent policy` 正式辅助头
- 用 GPU-first 新搜索替代当前 MCTS

# 现在真正要验证的事情

1. 新 pure-CNN trunk 的短训结果是否明显优于旧 baseline
2. `policy_kld / visits / plies_left` 是否确实帮助 value 学习
3. 搜索在当前 `WDL + MCTS` 语义下是否稳定、合法、可解释
4. 再决定后续是否需要新的辅助头或新的训练阶段

# 当前判断

- trunk 改动方向是对的
- 引擎与 ONNX 契约目前仍匹配
- 现在最需要避免的是：
  - 继续保留旧产物误导测试
  - 继续保留旧草案误导实现
