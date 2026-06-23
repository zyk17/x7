# 77xiangqi_engine

一个围绕 `MCTS + 小型 policy/value 网络` 的象棋工程。

当前项目已经收口到一条明确主线：

- 搜索主线只有 `MCTS`
- 训练主线只有 `px0 / lc0` 风格外部数据
- 网络只保留 `policy + value`
- 当前 value 主语义改为 `WDL + qMix`

## 当前真实状态

仓库当前主训练契约已经不是旧的 `15-plane human bootstrap`，而是：

- 输入：`124 x 10 x 9`
- 输入格式：`px0 classical input_format=1`
- 不含历史
- policy 输出：`2062` 维
- value 输出：`3-way WDL`

网络结构保持小，但不再过度压缩：

- shared trunk：`stem + residual blocks`
- policy head：`1x1 conv -> flatten -> linear(2062)`
- value head：`1x1 conv -> flatten -> mlp -> wdl(3)`

当前导出 / 引擎消费语义：

- ONNX `value` 输出为 `WDL` 概率
- 引擎侧统一按 `q = W - L` 消费 value
- 当前默认 `q_ratio=1.0`，即主训练默认优先学搜索 `WDL`

## 数据路线

现在的默认决策是：

1. 先用纯 `px0data` 跑通和迭代 baseline
2. 不预设人类数据混入
3. 只有当后续复盘分析或 second-stage fine-tune 明确受益时，才再引入人类数据

这意味着：

- 本地 `XRSH`
- 本地 `search-label-manifest`
- 旧 `move_vocab` 人类 policy 路线

都不再是主线。

## 当前建议的数据保留策略

仓库内建议只保留：

- `data/rounds/px0_train_v1.json`
- `data/rounds/px0_val_v1.json`

其余都视为可再生临时产物：

- `*.pt`
- `*.onnx`
- `data/xrsh_*`
- 旧的人类 baseline 数据包

如果你已经决定不再保留 XRSH 支线数据，那么这些都可以直接删除。

## 目录

```text
crates/
  xiangqi_core
  engin
  xiangqi_dataset
nn/
  src/nn
  scripts/data
  scripts/train
  scripts/export
```

架构边界见 [ARCHITECTURE.md](/C:/projects/77xiangqi_engine/ARCHITECTURE.md)。
