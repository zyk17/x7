# NextStep

当前 `px0` 主线 baseline 已经跑完第一轮长训。

当前确认结果：

- 模型：`124x10x9 -> 2062 + WDL`
- 训练配置：`width=96 blocks=6 batch=256 q_ratio=1.0`
- 已完成：`200k steps`
- 当前 best：`baseline_px0_wdl_v1.best.pt`
- 当前 best 指标大致为：
  - `val_policy ~= 2.4583`
  - `val_value_ce ~= 0.7199`
  - `val_value_q_mse ~= 0.0654`

下一阶段只做三件事：

## 1. 导出并接回引擎

目标：

- 把当前 `124x10x9 -> 2062 + WDL` 模型导出成 ONNX
- 确认 `engin` 能稳定消费

完成标准：

- ONNX 导出成功
- 引擎按 `q = W - L` 消费 WDL
- UCI / benchmark / 搜索链路能跑通

## 2. 做小规模对照实验

目标：

- 不再重复跑同一条 200k 长训
- 只做少量高价值对照，验证下一轮该改哪一个维度

优先顺序：

1. `value_loss_weight` 上调一组
2. `q_ratio` 做一组非 `1.0` 对照
3. 视显存和速度决定是否把模型稍微放大一档

完成标准：

- 每组只跑 `10k~20k`
- 记录与 `baseline_px0_wdl_v1` 的对比
- 明确下一轮长期配置

## 3. 决定是否彻底移除 XRSH 数据支线

目标：

- 代码可留最小兼容
- 仓库数据与实验产物尽量干净

完成标准：

- 删除旧 `pt / onnx / xrsh`
- 明确只保留 `px0` 分片清单和必要脚本

## 当前不该做的事

1. 不再扩大本地搜索标注数据
2. 不再围绕 `15-plane human bootstrap` 继续投入
3. 不引入新的 head
4. 不为了未来复盘系统提前堆抽象
