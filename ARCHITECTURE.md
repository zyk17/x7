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
- 模型：小型纯 CNN trunk + `policy/value`
- value 监督：`WDL + qMix`

当前不再把下面这些当主线：

- 本地大规模搜索标注建包
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
- 当前搜索主线固定为：
  `iteration -> gather minibatch -> batched eval -> backup`
- 当前允许的并发形态是：
  最小 `shared-tree` 多线程
- 当前树统计正式包含：
  `visits + in_flight`、`collision`、`multivisit`
- 不再扩 Alpha-Beta 主线
- 当前明确先不做：
  `MultiPV`、复杂 ONNX backend / evaluator 池
- 对外 UCI 选项保持收敛：
  只公开 `PolicyFile / MctsPlayouts / MctsCpuct / MctsFpuReduction / MctsBatchCap / MctsWorkers`
  不再兼容旧的 `Playouts / Visits / Cpuct / FpuReduction / SearchBatchSize / Threads`

### `crates/xiangqi_dataset`

职责：

- 最小数据辅助工具
- 保留 PGN / ICCS 清理能力
- 作为后续自对弈数据预处理地基

约束：

- 不承担主规模训练数据生产
- 不再保留 `XRSH` 正式数据格式
- 不再向“本地大规模数据平台”演化

### `nn/`

职责：

- 读取 `px0 v6` chunk
- 训练小型 `policy + value`
- 使用 train-only 辅助头改善 trunk / value 学习
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
- 编码语义：`8 x 15` history blocks + `4` aux planes
- 线上主线：真实 `PositionHistory`
- fallback：孤立 `FEN` 使用 `fen_only`

### 输出

- policy：`2062` 维 logits
- value：`3-way WDL`

### 网络

- shared trunk：`stem + pre-activation residual + global-pooling residual`
- policy head：`pure CNN conv head -> px0 52x10x9 conv move map -> 2062`
- value head：`global pooled CNN head -> wdl(3)`
- train-only aux head：`moves-left`

### value 语义

- 主训练 target：固定标量 `q_ratio` 下混合 `winner_wdl` 与 `search_wdl`
- `q_ratio=0.0` 时等价于只学 `winner_wdl`
- `q_ratio=1.0` 时等价于只学 `search_wdl`
- 训练阶段允许按阶段切换固定 `q_ratio`，但单次训练运行内保持常量
- 当前默认 baseline 容量为 `10x160`
- ONNX 导出：`value` 为 WDL 概率
- 引擎消费：派生 `q = W - L`

这里的关键取舍是：

- 保持网络小
- 保持实现清楚
- trunk 向 `KataGo` 风格靠拢，用全局池化强化 value 学习
- policy 正式主线改成纯 CNN，不再保留 `attention policy`
- 引擎正式推理仍只消费 `policy + WDL`
- 训练时允许 `moves-left` 辅助头，但不扩引擎正式输出契约

## 5. 数据策略

当前默认策略：

1. 先纯 `px0`
2. 不预设人类数据混入
3. 后续只有在复盘解释或 second-stage fine-tune 明确有效时，再引入人类数据

因此当前仓库内已经不再保留 `XRSH` 正式支线；后续如需补数据，只在 `px0` 主线或未来自对弈预处理链路上继续演化。

## 6. 现在真正要维护的地基

真正需要稳定维护的只有这些：

1. `xiangqi_core` 规则正确
2. `engin` 的 MCTS / UCI / ONNX / history 编码语义一致
3. `px0_record.py` 和 `dataset_px0.py` 能稳定读数据
4. `train_px0.py` 能稳定长跑、保存、恢复、导出
5. 导出的 ONNX 契约与引擎消费侧一致

### `nn/` 当前训练额外使用的 px0 字段

- `best_q / best_d`
- `result_q / result_d`
- `visits`
- `policy_kld`
- `plies_left`

当前最值得继续投入的是前两项：

1. `xiangqi_core` 与参考实现对拍
2. `engin` 搜索语义继续向 `lc0 / px0` 靠拢

### `engin` 当前输入约束

- UCI `position ... moves ...` 必须保留真实历史
- 搜索 root 输入必须从 history 编码，不再伪造 history
- 搜索过程只在 root 构造一次 history；simulation 仍基于 `Position do/undo`
- 搜索 iteration 内允许收集多个待评估叶子，再统一做 batched ONNX 推理
- 当前多线程只做到最小 `shared-tree` worker 语义
- 第三方 GUI 若只给最终 `FEN`，允许走 fallback，但不视为最佳接入方式

### `engin` 当前复用边界

- 已有主线复用：`tree reuse`
  - `advance_root`
  - `reset_to_position`
  - `position ... moves ...` 增量复用
- P2 可做的复用：`eval cache`
  - 只缓存 `position/history -> NN output`
  - 用于减少重复推理
- 当前不做的复用：传统 `TT`
  - 不做 `hash -> bound/value/bestmove`
  - 不做跨分支搜索统计合并
  - 不做 `graph / DAG` 主结构

### `engin` 当前 P2 方向

- 框架继续对齐 `px0 classic`
- worker 并发行为参考 `KataGo`
- 重点吸收：
  - 高失败重试容忍（单/并行语义一致；`retry_without_playout` 可观测）
  - 无预算上限时空转 `yield` + 周期性 `sleep` 退让
  - virtual loss / in-flight 分流
  - gather / backend / backup 的持续流水线
  - root 附近“宽而不乱”的访问分配
- 当前明确不吸收：
  - `KataGo graphHash`
  - `useGraphSearch`
  - ownership / score utility / pattern bonus 等扩展系统

## 7. 删除原则

仓库内可以不保留的内容：

- 旧 `*.pt`
- 旧 `*.onnx`
- 旧人类 baseline 数据
- 旧实验 CSV / 临时统计

当前真正的主数据源不是仓库内文件，而是本地版本目录：

- `C:\work\px0data\{version}\`

`data/rounds/*.json` 只是在你想固定一次 train/val 文件切分时才需要保留；
如果不需要固定切分，直接走 `--px0-version` 即可，这些 manifest 也不属于必须长期保留的仓库资产。
