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
- [x] 去掉 `attention policy`，切到纯 CNN `KataGo` 风格 trunk
- [x] 把 `visits / policy_kld / plies_left` 接回 `px0` 主训练链路

## P1 当前进行中

- [x] 跑第一版 `px0` 长跑 baseline
- [x] 记录 best checkpoint 与验证结果
- [x] 导出当前 baseline ONNX
- [x] 验证引擎消费新的 `WDL` ONNX 契约
- [x] 把 `engin` 输入切到真实 history 主线
- [x] 保留孤立 `FEN -> fen_only` fallback
- [x] 把搜索主线重构为单线程 `iteration + minibatch + in_flight`
- [x] 去掉旧 root-only reservation 语义

## P2 当前高优先级

- [x] 跑当前 best checkpoint 的 ONNX 导出
- [x] 完成一次 `onnx-smoke`
- [x] 完成一次 `bench`
- [x] 完成一次最小 `UCI` 联调
- [x] 对齐 `xiangqi_core` perft / 合法着到 `pyffish`
- [x] 补 `rules_regression` 基础回归
- [x] root tree reuse 从“单步”扩到“连续追加历史”
- [x] 接上最小 `Threads` shared-tree 搜索
- [ ] 用自家 GUI 做一次“真实 history”联调
- [ ] 记录 `Threads=1/2/4` 与 `SearchBatchSize=16/32/64` 的固定对照结果
- [ ] 继续检查 `pv / seldepth / nps` 是否和行为一致
- [ ] 检查 repetition / terminal 处理是否还有边界问题

## P3 模型相关高优先级

- [ ] 跑新纯 CNN baseline 的短训
- [ ] 导出新 baseline 的 ONNX 并做冒烟
- [ ] 做 `q_ratio` 分阶段训练对照
- [ ] 继续观察 opening policy / value 偏差

## P4 仓库收口

- [x] 删除仓库内旧 `pt / onnx` 数据产物
- [x] 从仓库主线移除 `XRSH` 训练/标注链路
- [x] 从 `engin` 主线移除 `move_vocab`
- [ ] 清理不再需要的旧命令与临时记录

## P5 明确先不做

- [ ] `MultiPV`
- [ ] 复杂 ONNX backend / evaluator 池
- [ ] 扩正式引擎输出 head
- [ ] 为未来复盘系统提前堆抽象
