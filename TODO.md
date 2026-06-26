# TODO

## P0 已完成

- [x] 收敛项目边界，只保留 `MCTS + policy/value`
- [x] 清掉 Alpha-Beta 主线
- [x] 打通最小 `UCI -> MCTS -> ONNX` 链路
- [x] 证明本地搜索标注可行但过慢
- [x] 决定主训练数据路线切到 `px0 / lc0`
- [x] 打通 `px0 v6` 最小读取路径
- [x] 跑通 `px0` 子集冒烟训练
- [x] 固定当前主模型 I/O：`124x10x9 -> 2062 + WDL`
- [x] 把小模型头部改成 `px0 classical` 风格
- [x] 把 value 语义切到 `WDL + qMix`

## P1 当前进行中

- [x] 跑第一版 `px0` 长跑 baseline
- [x] 记录 best checkpoint 与验证结果
- [x] 导出当前 baseline ONNX
- [x] 验证引擎消费新的 `WDL` ONNX 契约
- [x] 把 `engin` 输入切到真实 history 主线
- [x] 保留孤立 `FEN -> fen_only` fallback
- [x] 把搜索主线重构为单线程 `iteration + minibatch + in_flight`
- [x] 去掉旧 root-only reservation 语义

## P2 下一步实验

- [x] 跑当前 best checkpoint 的 ONNX 导出
- [x] 完成一次 `onnx-smoke`
- [x] 完成一次 `bench`
- [x] 完成一次最小 `UCI` 联调
- [ ] 用自家 GUI 做一次“真实 history”联调
- [ ] 记录“最终 FEN vs 完整历史”对照样例
- [ ] 记录重构后 `pv / seldepth / nps` 与旧版本的对照结论
- [ ] 做 `value_loss_weight` 对照实验
- [ ] 做非 `1.0 q_ratio` 对照实验

## P3 仓库收口

- [ ] 删除仓库内旧 `pt / onnx` 数据产物
- [x] 从仓库主线移除 `XRSH` 训练/标注链路
- [x] 从 `engin` 主线移除 `move_vocab`
- [ ] 清理不再需要的旧命令与临时记录

## P4 后续再决定

- [ ] 评估是否需要人类数据 second-stage fine-tune
- [ ] 评估是否需要为复盘分析增加额外输出
- [ ] 设计自对弈数据预处理与混入策略
