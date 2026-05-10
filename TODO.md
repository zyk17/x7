# 执行清单（阶段索引）

本文件只保留**阶段入口**与**近期主线**。

- 阶段定义：`MILESTONES.md`
- 目标原则：`工程目标.md`
- 近期执行：`NEXT_STEPS.md`
- 架构与契约：`ARCHITECTURE.md`

---

## 产品与方法论（全员对齐）

- **定位**：**人类认知驱动的搜索**——模型主要学习 **人类棋谱与局面语义**（policy、危险、先手权等），用于 **候选空间与剪枝先验**；**不是**「堆一个大网络替代 Pikafish 式静态评估」。
- **双主线**：**短期主产品**是复盘系统，模型输出需要**可解释**；**长期路线**是搜索引擎，搜索负责验证这些语义是否真的有用。
- **分工**：**机器话 / 战术穷尽 / 物质与深度验证** → **`engin` 搜索 + `xiangqi_core`**；**人类风格与大局观** → **`nn/` 小网络 + ONNX**。
- **文档**：原则性表述以 **`ARCHITECTURE.md`**、根目录 **`README.MD`**、**`.cursorrules`**、**`agents.md`** 为准。

---

## 近期主线

当前只做三件事：

1. 做实 `P3 engin`
2. 建 benchmark / ablation
3. 对现有 heads 做“复盘解释价值 + 搜索收益”双重归因

详细拆解见 `NEXT_STEPS.md`。

执行方式：

- 写代码时：优先做 `engin`、benchmark、ablation、最小消费链路
- 不写代码时：优先跑训练对比矩阵，判断 `attack / danger / tactical / value` 的保留价值

---

## P0 — `xiangqi_core`：规则 + 合法 UCI（对齐 pikafish-rust / Pikafish）

**当前阶段：实现已落地，「可对拍验证」是门禁**

- [x] 对照 `pikafish-rust` 划定模块边界：`types` / `misc` / `board` / `movegen`
- [x] 局面表示：`from_fen` / `set_fen`；perft + do/undo 回归
- [x] 各子力走法与阻挡、将/帅照面、过河等（移植自 pikafish-rust）
- [x] 将军 / 应将 / 合法性过滤（`GenType::Legal` + `Position::legal`）
- [x] **合法着 UCI**：`legal_moves_uci`、`parse_move_uci`（纵坐标 **0～9**，与 Pikafish 等引擎 UCI 一致）
- [x] **一致性测试（门禁）**：`nn/scripts/parity/pyffish_xiangqi_core_parity.py` + `pytest tests/test_pyffish_xiangqi_core_parity.py`（须 `pyffish` + `cargo`）；`xiangqi_core` 二进制 `legal_moves_dump`；种子含 **根 FEN + 非空 `uci_prefix`**；可选扩展：与 **Pikafish** 边界用例对照（外部仓库 `c:\projects\Pikafish`，本仓仅文档引用）
- [x] 文档：`crates/xiangqi_core/README.md`（来源、API、测试）

---

## P1 — `xiangqi_dataset`：PGN → 二进制 shards（按局并行）

**当前阶段：MVP 已可用；下一步是「规则单一来源」增强**

- [x] 输入：**PGN**（ICCS / UCI，Rust `encode`）；**`vocab-enum`** 生成固定 canonical 词表
- [x] 输出：**XRSH**（`shard_NNNNN.xrsh`，魔数 `XRSH`）+ `pack_meta.json`；**当前默认 `xrsh_v3`**
- [x] `vocab_sha256` 与 Python 词表指纹一致；`pack_meta.format` 为 **`xrsh_v3`**
- [x] CLI：`vocab-enum` / `pgn-shards`，`--jobs`、`--games-per-shard`
- [x] Python **`PolicyXrshDataset`**（`nn.dataset_xrsh`，`--train-xrsh-dir`）
- [x] Rust：单元测试（`iccs` / `pgn` / `vocab`）、集成冒烟 `tests/pgn_xrsh_smoke.rs`（PGN → XRSH + `read_shard_header` + `pack_meta`）
- [x] **XRSH v3**：三辅助 float + **`game_result_red` / `ply_total`**（`pack_meta.format: xrsh_v3`，文件头版本 3）；Python **`xrsh_io` / `PolicyXrshDataset`** 读入；**value 头**用终局标签（非 `2*attack-1`）

---

## P2 — `nn/`：二进制训练包 + 多头网络

**当前阶段：XRSH + 训练 + 多头 + ONNX 已打通；与 P1「Rust 预计算标签」衔接前注意双源风险**

- [x] `Dataset` / DataLoader 读取 P1 **XRSH**（`.xrsh`）
- [x] 模型：shared trunk + **policy** + **attack / danger / tactical**（定义见 ARCHITECTURE / `aux_pseudo_labels.py`）
- [x] 损失与权重；验证指标（`--aux-loss-weight`、`val_aux_mse`）
- [x] **ONNX 导出**（`export_onnx.py`：logits + 可选三头，图中 sigmoid）
- [x] 训练路径 **XRSH only**
- [x] **惯例**：**`xrsh_v3`** + 多头训练 → 辅助标签来自 Rust；**`train_policy` 默认开 value 头**（棋谱须含 `[Result]`；**`--no-value-head`** 关闭）；旧 **v1/v2** 或想排除辅助头噪声时用 **`--no-aux-heads`**

---

## P3 — `engin`：搜索 + UCI

- [x] UCI 最小闭环：`uci`（含 **id**、**option**、**uciok**）/ `isready` / `ucinewgame` / **`setoption`**（含 **Clear Hash** 按钮项）/ `position startpos|fen …` + `moves` / `go`（`depth` `movetime` `infinite` `ponder` `nodes`）/ `stop` / `ponderhit` / `quit`；**无参数启动 `engin` 即 UCI（stdin/stdout）**
- [x] **Alpha-Beta** + **静止搜索**（吃子延伸；被将军时全应将）+ **置换表 TT**
- [x] **迭代加深**（`go` / `go depth`）；**movetime** / **nodes** 在搜索内检查（不再先睡眠再搜）；**infinite** 配合 **stop** 与节点内轮询
- [x] **Move ordering**：TT + MVV-LVA + 杀手 + 根 policy logit；静止阶段仅 MVV-LVA（不吃 ONNX）
- [x] ONNX Runtime 加载 P2 导出模型 + **单次局面推理**：`engin::PolicyOnnx`（输入名 `board`，输出 `logits` + 可选 `attack`/`danger`/`tactical`）；`cargo run -p engin -- --onnx-smoke [PATH]`；`cargo test -p engin` 在存在 `data/policy.onnx` 时起推理冒烟
- [x] 与 `xiangqi_core` 走子、合法性、终局判定联调（`crates/engin/tests/p3_integration.rs` + `parse_position_uci`）
- [x] 建立固定 FEN benchmark 集与统一输出格式（`engin::benchmark`、`engin --bench [--depth N]`，NDJSON）
- [x] 建立搜索侧消融：`setoption name UsePolicyOrdering` / `UseNNLeaf`（**attack/danger/tactical** 当前不进入搜索树，无独立开关；见 `NEXT_STEPS`）
- [x] **ONNX 契约回归**：`nn/tests/test_policy_onnx_contract.py` 校验 `data/policy.onnx` 的 I/O 名与形状（与 `export_onnx.py` 一致；**`data/` 被 gitignore**，本地放入导出文件后跑 `pytest` 即执行）

---

## P4 — Value Head

- [ ] 仅保留“最小 value head”路线
- [ ] 明确 value 契约、标签来源、训练指标，优先服务“人类局面感”解释
- [ ] 只先接入 `engin` 的最小消费点
- [ ] 以 benchmark 收益决定是否继续扩大作用范围

---

## P5 — Search-aware Heads

- [ ] 先比较现有 `attack / danger / tactical / value` 的真实收益
- [ ] 只保留有复盘解释价值或 benchmark 收益的 head
- [ ] 在完成收益归因前，不新增 `forcing / volatility / mobility_tension`
- [ ] 暂不做 `style / sacrifice / initiative / psychological`

---

## P6 / P7 — 后续阶段

- [ ] `P6`：搜索注意力蒸馏
- [ ] `P7`：动态搜索

说明：

- 这两项在完成 `P3` 做实、benchmark、head 收益归因之前不进入近期主线
- 详细任务以后续版本 `NEXT_STEPS.md` 为准

---

## 已完成（归档区）

- **2026-05**：P0 主体 — 自 pikafish-rust 并入 `types` / `misc` / `board` / `movegen`；`Position::from_fen`、`global_zobrist`、`legal_moves_uci`；`tests/perft.rs`（depth 1–3 = 44 / 1926 / 80069）。
- **2026-05**：P1 MVP — `xiangqi_dataset`：`vocab-enum` / `pgn-shards`，XRSH（`.xrsh`），`pack_meta.json`；`uci_format` 与 Pikafish UCI（纵坐标 0～9）对齐。
- **2026-05**：P2 — `PolicyResNet` 可选多头；`aux_pseudo_labels` + `root_fen`/`uci_prefix`/合法 UCI 表；ONNX 多输出；训练 `unpack_*_batch`。
