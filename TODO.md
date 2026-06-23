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
- [ ] 导出当前 baseline ONNX
- [ ] 验证引擎消费新的 `WDL` ONNX 契约

## P2 下一步实验

- [ ] 跑当前 best checkpoint 的 ONNX 导出
- [ ] 完成一次 `onnx-smoke`
- [ ] 完成一次 `bench`
- [ ] 完成一次最小 `UCI` 联调
- [ ] 做 `value_loss_weight` 对照实验
- [ ] 做非 `1.0 q_ratio` 对照实验

## P3 仓库收口

- [ ] 删除仓库内旧 `pt / onnx / xrsh` 数据产物
- [ ] 决定是否连 `move_vocab.json` 一并移除
- [ ] 清理不再需要的旧命令与临时记录

## P4 后续再决定

- [ ] 评估是否需要人类数据 second-stage fine-tune
- [ ] 评估是否需要为复盘分析增加额外输出
- [ ] 评估是否要保留最小 XRSH 支线代码
