# NextStep

当前 `px0` 主线 baseline 已经跑完第一轮长训。

当前确认结果：

- 模型：`124x10x9 -> 2062 + WDL`
- 引擎输入：`px0 classical` 真实 history 主线，`fen_only` fallback
- 搜索核心：单线程 `iteration + minibatch + in_flight`
- 下一轮正式 baseline：`width=128 blocks=8 batch=256 q_ratio=0.0`
- 已完成：`200k steps`
- 当前 best：`baseline_px0_wdl_v1.best.pt`
- 当前 best 指标大致为：
  - `val_policy ~= 2.4583`
  - `val_value_ce ~= 0.7199`
  - `val_value_q_mse ~= 0.0654`

下一阶段只做三件事：

## 1. 做真实历史链路联调

目标：

- 把当前 ONNX 真正接入你们自己的 GUI 历史输入
- 对比“只有最终 FEN”和“完整历史”两种输入差异
- 重点观察 `pv / seldepth / bestmove` 是否比旧搜索稳定

完成标准：

- GUI 侧能按“重置局面 + 逐步追加 move”驱动引擎
- 至少记录 2~3 个非起始局面的差异样例
- 明确当前 baseline 是否仍然存在明显 opening 偏差与离谱应将问题

## 2. 做小规模对照实验

目标：

- 不再重复跑同一条 200k 长训
- 只做少量高价值对照，验证下一轮该改哪一个维度

优先顺序：

1. `value_loss_weight` 上调一组
2. `q_ratio` 做一组非 `1.0` 对照
3. 视显存和速度决定是否把模型稍微放大一档
4. 只在 baseline 行为仍明显异常时，再考虑继续调搜索常数

完成标准：

- 每组只跑 `10k~20k`
- 记录与 `baseline_px0_wdl_v1` 的对比
- 明确下一轮长期配置

## 3. 收掉旧支线与旧产物

目标：

- 代码可留最小兼容
- 仓库数据与实验产物尽量干净

完成标准：

- 删除旧 `pt / onnx`
- 只保留 `px0` 分片清单和必要脚本
- `xiangqi_dataset` 只保留 PGN / 后续自对弈预处理地基

## 当前不该做的事

1. 不再扩大本地搜索标注数据
2. 不再围绕 `15-plane human bootstrap` 继续投入
3. 不引入新的 head
4. 不为了未来复盘系统提前堆抽象
