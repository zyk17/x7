# 77xiangqi_engine

一个围绕 `MCTS + policy/value 网络` 的象棋工程。

当前项目已经收口到一条明确主线：

- 搜索主线只有 `MCTS`
- 训练主线只有 `px0 / lc0` 风格外部数据
- 网络只保留 `policy + value`
- 当前 value 主语义改为 `WDL + qMix`
- 引擎基建当前主对齐对象改为 `lc0 classic`
- 线上推理输入主线保持真实 history 编码
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
- 当前已支持最小 `shared-tree` 多线程搜索
- `position startpos moves ...` / `position fen ... moves ...` 会在引擎内保留真实历史
- 只给最终 `FEN` 的第三方 GUI 仍可用，但属于 fallback 路径，强度可能下降

## 当前 UCI 公开选项

当前主 UCI 只公开这些日常需要的选项：

- `PolicyFile`：ONNX 路径
- `MctsPlayouts`：默认 playout 预算（`go` 无 `nodes/movetime` 时）
- `CPuct`：PUCT 探索强度（lc0 默认 `1.745`）
- `FpuValue`：未访问边 first-play urgency（lc0 默认 `0.330`）
- `MinibatchSize`：gather batch；`0` = backend `recommended_batch_size`（ONNX CPU/CUDA 默认 `256`，DML 默认 `64`；上限 `1024`）
- `NNCacheSize`：NN 评估 LRU 缓存容量（默认 `200000`）
- `Threads`：搜索线程；`0` = `suggested_num_search_threads + !runs_on_cpu`
- `MultiPV`：UCI 主变线条数（default `1`，max `500`）

说明：

- **`MultiPV` 当前只覆盖 lc0 主选项与 `SendUciInfo` 默认输出形态**；不含 `PerPVCounters`（每条线独立 `nodes`）
- `MultiPV>1` 时，每条 `info` 共用总 `nodes`（与 lc0 默认、`PerPVCounters=false` 一致）
- 其余 lc0 搜索参数已内化，不再以 `Mcts*` 前缀公开
- 内化能力含：OOO eval、NN prefetch、7 步搜索流水线、独立 watchdog
- `Threads=0` 时 GPU backend 通常解析为 `2`，CPU 为物理核数
- `MinibatchSize=0` 时使用 backend `recommended_batch_size`（上限 `1024`）

## 当前阶段重点

当前短期重点不是继续折腾模型。

当前判断是：

- policy 已经可用
- 中残局 value 问题短期内不会靠小修小补立刻解决
- 现阶段最值得投入的是继续把搜索做对、做稳、做深

因此当前主线优先级是：

1. 继续收紧搜索语义
2. 继续提高搜索稳定性与并行细节
3. 继续补搜索侧回归护栏
4. 模型训练保持可用即可，不作为当前主攻方向

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

当前后续计划见 [NextStep.md](/C:/projects/77xiangqi_engine/NextStep.md) 和 [TODO.md](/C:/projects/77xiangqi_engine/TODO.md)。

架构边界见 [ARCHITECTURE.md](/C:/projects/77xiangqi_engine/ARCHITECTURE.md)。
