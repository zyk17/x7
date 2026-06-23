# ARCHITECTURE

## 1. 目标

这是一个围绕 `MCTS + 小型 policy/value` 的象棋项目。

当前阶段只有两个核心目标：

1. 稳定维护一条可用的搜索与推理主链路
2. 用 `px0` 风格数据持续训练一个小而正确的 baseline

## 2. 当前主路线

主路线已经固定为：

- 搜索：`MCTS`
- 训练数据：`px0 / lc0` 风格外部 chunk
- 模型：小型 shared trunk + `policy/value`
- value 监督：`WDL + qMix`

当前不再把下面这些当主线：

- `XRSH` 扩样本
- 本地大规模搜索标注
- `15 planes + move_vocab` 的长期正式 I/O
- Alpha-Beta 搜索

## 3. 模块职责

### `crates/xiangqi_core`

职责：

- 局面表示
- FEN
- 合法着生成
- do / undo
- 终局判断

约束：

- 只处理规则真相
- 不混入搜索策略
- 不混入训练逻辑

### `crates/engin`

职责：

- MCTS 树与搜索循环
- ONNX policy/value 推理接入
- UCI
- benchmark / 调试统计

约束：

- 搜索预算口径统一成 `playouts / nodes / deadline`
- 线上与离线尽量共用一套搜索核心
- 不再扩 Alpha-Beta 主线

### `crates/xiangqi_dataset`

职责：

- 最小数据辅助工具
- 保留 PGN / XRSH / 小批量搜索标注能力

约束：

- 不承担主规模训练数据生产
- 不再向“本地大规模数据平台”演化

### `nn/`

职责：

- 读取 `px0 v6` chunk
- 训练小型 `policy + value`
- 导出 ONNX

约束：

- 不重写规则
- 不重写搜索
- 优先对齐 `px0 classical` 的数据与网络契约

## 4. 当前模型契约

### 输入

- 形状：`124 x 10 x 9`
- 数据来源：`px0 v6`
- 当前只支持：`classical input_format=1`
- 不含历史

### 输出

- policy：`2062` 维 logits
- value：`3-way WDL`

### 网络

- shared trunk：`stem + residual blocks`
- policy head：`1x1 conv -> flatten -> linear(2062)`
- value head：`1x1 conv -> flatten -> mlp -> wdl(3)`

### value 语义

- 主训练 target：`q_ratio * search_wdl + (1 - q_ratio) * winner_wdl`
- 当前默认：`q_ratio=1.0`
- ONNX 导出：`value` 为 WDL 概率
- 引擎消费：派生 `q = W - L`

这里的关键取舍是：

- 保持网络小
- 保持实现清楚
- 但不再用 `global average pooling -> single linear` 这种会过早抹掉空间信息的 policy 头

## 5. 数据策略

当前默认策略：

1. 先纯 `px0`
2. 不预设人类数据混入
3. 后续只有在复盘解释或 second-stage fine-tune 明确有效时，再引入人类数据

因此本地 `XRSH` 当前只剩下两种角色：

- 调试工具
- 可选的小支线实验

如果你决定彻底不走这条支线，仓库内的 `XRSH` 数据、旧 checkpoint、旧 ONNX 都可以删除。

## 6. 现在真正要维护的地基

真正需要稳定维护的只有这些：

1. `xiangqi_core` 规则正确
2. `engin` 的 MCTS / UCI / ONNX 语义一致
3. `px0_record.py` 和 `dataset_px0.py` 能稳定读数据
4. `train_px0.py` 能稳定长跑、保存、恢复、导出
5. 导出的 ONNX 契约与引擎消费侧一致

## 7. 删除原则

仓库内可以不保留的内容：

- 旧 `*.pt`
- 旧 `*.onnx`
- `data/xrsh_*`
- 旧人类 baseline 数据
- 旧实验 CSV / 临时统计

仓库内建议保留的最小数据文件：

- `data/rounds/px0_train_v1.json`
- `data/rounds/px0_val_v1.json`

它们只是分片清单，体积相对小，而且能直接复用你的本地 px0 数据目录。
