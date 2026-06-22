# TODO

## P0 已完成地基

- [x] 收敛项目边界，只保留 `MCTS + policy/value + XRSH v5`
- [x] 清掉 Alpha-Beta / 辅助头 / 多正式格式残留
- [x] 打通 `UCI -> MCTS -> XRSH -> nn` 主链路
- [x] 固化 baseline 训练命令、checkpoint 命名和结果记录方式
- [x] 明确训练用搜索与线上对弈搜索的参数分离策略
- [x] 设计第一轮小批量、分轮次、受控混合方案
- [x] 明确停止条件：什么时候不再继续下一轮自我升级

## P1 当前正在收口

- [x] 跑通纯人类数据 `policy` baseline
- [x] 定位并完成第一轮训练吞吐修正（Windows + CUDA 数据加载默认值）
- [ ] 固定一版纯人类 `policy` baseline 对照组结果
- [ ] 整理本轮 baseline 的 checkpoint / 日志 / val 指标

## P2 搜索标注数据

- [x] 明确冷门局面补充来源与筛选规则
- [x] 记录各数据源占比，避免后续混合失控
- [ ] 整理一批更适合搜索标注的人类局面数据
- [ ] 产出第一版搜索标注 XRSH 数据包
- [ ] 补一份搜索标注数据统计摘要：`search_visits / search_q / search_counts`

## P3 Value Baseline

- [ ] 跑通纯搜索标注 `value(search_q)` baseline
- [ ] 确认 `value_min_visits` 的初始默认值
- [ ] 评估搜索标注参数：`visits / cpuct / 是否带噪声`
- [ ] 判断当前 `search_q` 目标是否稳定优于“只看最终胜负”

## P4 第一轮闭环

- [ ] 产出 round_1 混合配置
- [ ] 完成第一轮混合训练实验
- [ ] 比较第一轮前后 `policy`、`value`、搜索表现是否同步提升
- [ ] 判断是否继续 round_2，还是先停下来整理设计

## P5 暂缓

- [ ] `search_counts` 蒸馏 policy baseline

说明：

- 这项不是当前最优先
- 先把 `search_q value` 跑顺，再决定是否引入

## P6 Review 入口

- [ ] 每完成一个子阶段后做一次专项 review
- [ ] review 重点只看：是否更简洁、是否复用充分、是否偏离主线、是否影响可训练性
