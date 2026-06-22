# NextStep

当前已经完成的阶段性收口：

1. 项目主线已经固定为 `MCTS + policy/value + XRSH v5`
2. `UCI -> MCTS -> XRSH -> nn` 主链路已经打通
3. 纯人类数据 `policy` baseline 已经可以实际训练
4. Windows + CUDA 训练吞吐瓶颈已经定位并完成第一轮修正

这意味着当前不再是“继续搭地基”，而是进入第一阶段 baseline 收口：

1. 固定一版可复现的 `policy` baseline 结果，作为后续对照组
2. 产出第一批可用的搜索标注 XRSH 数据
3. 跑通 `value(search_q)` baseline
4. 做第一轮受控混合训练并评估是否值得继续下一轮

## 当前建议执行顺序

### 1. 固定 `policy` 对照组

目标：

- 保留一版稳定 checkpoint
- 固化训练命令、日志和评估结果
- 作为后续 `value` / 混合训练的对照

完成标准：

- 有明确 checkpoint
- 有 val 指标记录
- 有一版后续可重复运行的训练命令

### 2. 产出 round_1 搜索标注数据

目标：

- 不是海量自对弈
- 只做小批量、分轮次、受控生成
- 优先补冷门、偏离主流但仍有人类合理性的局面

原则：

- 搜索主监督是 `search_q`
- 数据格式只写回 `XRSH v5`
- 第一轮先追求数据可用，不追求数据量大

完成标准：

- 产出一版 `xrsh_search_r1` 或等价目录
- 能统计其中 `search_visits / search_q / search_counts` 覆盖情况

### 3. 跑通 `value(search_q)` baseline

目标：

- 单独验证当前 value 设计是否终于能学起来
- 不和 policy 蒸馏、混合训练同时耦合

原则：

- 先只训 `value(search_q)`
- 默认只让 `search_visits >= value_min_visits` 的样本参与 value loss
- 先看 value 是否稳定下降，再谈更复杂的权重策略

完成标准：

- 跑通训练
- 有基础验证指标
- 能看出是否比“只看最终胜负”的目标更稳定

### 4. 做第一轮受控混合训练

目标：

- 在不明显破坏人类特征的前提下，让搜索和 value 变好

原则：

- 人类数据始终作为锚点
- 搜索标注数据只做增量补充
- 每轮先评估，再决定是否继续下一轮

完成标准：

- 有一版 round_1 mix 配置
- 有一轮训练结果
- 有和纯人类 policy baseline 的对比

## 当前明确不做

1. 不回头加 Alpha-Beta 主线
2. 不回头加辅助头、多任务头
3. 不再新增正式训练格式
4. 不为未来社区平台提前加大抽象

## 你现在最应该做的事

1. 固定纯人类 `policy` baseline 结果
2. 开始做 round_1 搜索标注数据
3. 然后单独跑 `value(search_q)` baseline

你做完其中任何一个子阶段，我再按当前主线给你做专项 review。
