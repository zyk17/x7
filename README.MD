# 77xiangqi_engine

一个围绕 `MCTS + 小型 policy/value 网络` 的象棋工程。

当前项目已经收口到一条明确主线：

- 搜索主线只有 `MCTS`
- 训练主线只有 `px0 / lc0` 风格外部数据
- 网络只保留 `policy + value`
- 当前 value 主语义改为 `WDL + qMix`
- 线上推理输入主线改为 `px0 classical` 的真实 history 编码
- 仓库内已移除 `XRSH` 正式训练/标注链路

## 当前真实状态

仓库当前正式模型契约是：

- 输入：`124 x 10 x 9`
- 输入格式：`px0 classical input_format=1`
- 历史语义：`8 x 15` history blocks + `4` aux planes
- 主线输入：真实 `PositionHistory`
- fallback：孤立 `FEN` 走 `fen_only`
- policy 输出：`2062` 维
- value 输出：`3-way WDL`

网络结构保持小，但当前已经改成纯 CNN 主线：

- shared trunk：`stem + pre-activation residual + global-pooling residual`
- policy head：`pure CNN conv policy head -> px0 52x10x9 conv move map -> 2062`
- value head：`global pooled CNN value head -> wdl(3)`
- aux train-only head：`moves-left`

当前导出 / 引擎消费语义：

- ONNX `value` 输出为 `WDL` 概率
- 引擎侧统一按 `q = W - L` 消费 value
- 当前 value 目标为固定标量 `q_ratio` 下的 `winner_wdl / search_wdl` 混合
- `q_ratio=0.0` 表示只学最终结果，`q_ratio=1.0` 表示只学搜索目标
- 分阶段训练时允许通过新输出文件或 `--init-from` 切到新的固定 `q_ratio`
- 当前默认 baseline 容量为 `10x160`
- 当前训练会额外使用 `best_q / visits / policy_kld / plies_left`
- 搜索核心已切成单线程 `iteration -> minibatch gather -> batched NN -> backup`
- 当前搜索语义包含 `in_flight / collision / multivisit`
- `position startpos moves ...` / `position fen ... moves ...` 会在引擎内保留真实历史
- 只给最终 `FEN` 的第三方 GUI 仍可用，但属于 fallback 路径，强度可能下降

## 当前 UCI 公开选项

当前主 UCI 只公开这些日常需要的选项：

- `PolicyFile`：ONNX 路径
- `MctsPlayouts`：默认 playout 预算
- `MctsCpuct`：主探索强度
- `MctsFpuReduction`：未访问边的 first-play urgency 扣减
- `MctsBatchCap`：单轮 gather / evaluator 的 batch 上限，不是固定目标值
- `MctsWorkers`：shared-tree worker 数

说明：

- 不再兼容 `Playouts / Visits / Cpuct / FpuReduction / SearchBatchSize / Threads`
- 这样可以避免和 `Pikafish / Stockfish` 风格 GUI 选项名冲突
- 当前默认 `MctsWorkers=8`
- 当前默认 `MctsBatchCap=2048`
- 当前 `MctsBatchCap` 公开上限为 `8192`
- 当前 `MctsWorkers` 公开上限为 `1024`
- 当前搜索会按 root 访问数逐步放大 gather 大小，并在 evaluator 侧按动态目标 batch + 短等待 flush 提交

## 数据路线

现在的默认决策是：

1. 先用纯 `px0data` 跑通和迭代 baseline
2. 不预设人类数据混入
3. 只有当后续复盘分析或 second-stage fine-tune 明确受益时，才再引入人类数据

这意味着当前仓库只保留 `px0` 主训练路径；本地 `XRSH` / `search-label-manifest` 已经不再保留为正式入口。

## 当前建议的数据保留策略

当前主数据源是本地版本目录：

- `C:\work\px0data\{version}\`

仓库内的数据文件都应视为可再生产物，包括：

- `data/rounds/*.json`
- `*.pt`
- `*.onnx`
- 旧的人类 baseline 数据包

如果你想固定一次 train/val 文件切分，可以临时保留 `data/rounds/*.json`；
如果不需要固定切分，直接用 `--px0-version` 训练即可，不要求仓库长期保留这些 manifest。

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
