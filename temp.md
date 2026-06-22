# temp

当前临时讨论点（已做阶段性默认决策，后续实验可调整）：

## 1. `search_q` 口径

- **默认**：根节点 `root_value` 直接作为 `search_q` 训练目标
- **默认**：`value loss` 仅作用于 `search_visits >= 1` 的样本（人类无标注行不参与）
- 暂不对 visits 做额外置信度加权；若后续噪声大，可在 `value_target_weight_alpha` 上试验

## 2. 搜索蒸馏

- **第一轮默认**：`search_policy_weight = 0`（见 `data/rounds/round_0.json`）
- 专用 baseline 试验从 `0.2` 起（见 `nn/BASELINES.md`）
- 混入比例通过 `train_mix.sources[].weight` 与独立 `--search-policy-weight` 控制

## 3. 自对弈节奏（无 PGN 路径）

- **round_0**：仅现有大师 XRSH 训 policy
- **round_1 计划**：小批量 MCTS 自对弈 → `search_q` XRSH，再与人类数据混合训 value / 可选 policy 蒸馏
- 停止条件见 `data/rounds/round_0.json`

## 4. 搜索参数分离

- 训练标注：`configs/search_train.json`（512 playouts, cpuct 1.25, 无噪声）
- 线上对弈：`configs/search_play.json`（800 playouts, cpuct 1.25, 无噪声）
- UCI 主预算选项：`Playouts`
- UCI `Visits`：仅兼容别名
- `go depth`：当前明确不支持
- UCI `info` 与 benchmark 统一输出 `playouts / root_visits / nodes / nps`
