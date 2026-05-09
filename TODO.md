# 执行清单（随进度勾选）

权威优先级见 **`MILESTONES.md`**；架构上下文见 **`ARCHITECTURE.md`**；中期研发目标见 **`工程目标.md`**。

---

## 推进顺序（先读这段，减轻混乱）

仓库里 **P0→P1→P2** 仍成立，但近期容易「乱」的根因是：**policy 监督来自 Rust（XRSH），辅助头伪标签曾走 pyffish**，存在双规则源风险。下面按**依赖**排序，不必样样并行。

| 顺序 | 做什么 | 为什么 |
|------|--------|--------|
| **A** | 完成 **P0 未完成项**：`xiangqi_core` 与 **pyffish** 合法着（及必要时的吃子/局面）**抽样对拍**（脚本或 CI） | 不绿则不敢声称「数据与训练规则正确」；行为有疑问时对照本地 **`c:\projects\Pikafish`**（权威棋规边界）、结构对齐 **`c:\projects\pikafish-rust`** |
| **B** | 在 A 未完全绿之前：**仍可**用当前管线 **生成 XRSH + 训练**，但建议 **仅 policy**：`train_policy.py` 加 **`--no-aux-heads`**，避免辅助头再引入 pyffish；先把 policy 与词表、划分、管线跑通 | 降低噪声与心理负担 |
| **C** | **已完成**：**`xrsh_v2`** 包内 Rust 预计算辅助标签；训练与**水平镜像**分支均优先用分片标量（镜像下启发量与原局面同构，不再为辅助头调 pyffish）；仅 **v1** 或缺字段时回退 `aux_pseudo_labels` | 新数据与 policy 同一规则源 |
| **D** | **P3 `engin`**：搜索 + UCI + ONNX，依赖 `xiangqi_core` + P2 导出 | 产品闭环 |

**结论**：不是「先别生成数据」——而是 **先别在高风险配置（默认多头 + 未对拍）上投入大规模算力**；**完善 `xiangqi_core` + 对拍** 是当前最值得做的「止血」工作。

---

## P0 — `xiangqi_core`：规则 + 合法 UCI（对齐 pikafish-rust / Pikafish）

**当前阶段：实现已落地，「可对拍验证」是门禁**

- [x] 对照 `pikafish-rust` 划定模块边界：`types` / `misc` / `board` / `movegen`
- [x] 局面表示：`from_fen` / `set_fen`；perft + do/undo 回归
- [x] 各子力走法与阻挡、将/帅照面、过河等（移植自 pikafish-rust）
- [x] 将军 / 应将 / 合法性过滤（`GenType::Legal` + `Position::legal`）
- [x] **合法着 UCI**：`legal_moves_uci`、`parse_pyffish_uci`（纵坐标 **1～10**，与 pyffish 字符串一致）
- [x] **一致性测试（门禁）**：`nn/scripts/parity/pyffish_xiangqi_core_parity.py` + `pytest tests/test_pyffish_xiangqi_core_parity.py`（须 `pyffish` + `cargo`）；`xiangqi_core` 二进制 `legal_moves_dump`；种子含 **根 FEN + 非空 `uci_prefix`**；可选扩展：与 **Pikafish** 边界用例对照（外部仓库 `c:\projects\Pikafish`，本仓仅文档引用）
- [x] 文档：`crates/xiangqi_core/README.md`（来源、API、测试）

---

## P1 — `xiangqi_dataset`：PGN → 二进制 shards（按局并行）

**当前阶段：MVP 已可用；下一步是「规则单一来源」增强**

- [x] 输入：**PGN**（ICCS / UCI，Rust `encode`）+ **`jsonl-shards`** 读 JSONL
- [x] 输出：**XRSH**（`shard_NNNNN.xrsh`，魔数 `XRSH`）+ `pack_meta.json`；**当前默认 `xrsh_v2`**
- [x] `vocab_sha256` 与 Python 词表指纹一致；`pack_meta.format` 为 **`xrsh_v2`**（旧 v1 包仍兼容）
- [x] CLI：`pgn-shards` / `jsonl-shards`，`--jobs`、`--games-per-shard`
- [x] Python **`PolicyXrshDataset`**（`nn.dataset_xrsh`，`--train-xrsh-dir`）
- [x] Rust：单元测试（`iccs` / `pgn` / `vocab`）、集成冒烟 `tests/jsonl_smoke.rs`（JSONL → XRSH + `read_shard_header` + `pack_meta`）
- [x] **XRSH v2**：编码行走 **`xiangqi_core`** 预计算 **attack/danger/tactical** 写入分片（`pack_meta.format: xrsh_v2`，文件头版本 2）；Python **`xrsh_io` / `PolicyXrshDataset`** 优先读入；缺字段或旧 **v1** 分片回退 pyffish（`aux_pseudo_labels`）

---

## P2 — `nn/`：二进制训练包 + 多头网络

**当前阶段：XRSH + 训练 + 多头 + ONNX 已打通；与 P1「Rust 预计算标签」衔接前注意双源风险**

- [x] `Dataset` / DataLoader 读取 P1 **XRSH**（`.xrsh`）
- [x] 模型：shared trunk + **policy** + **attack / danger / tactical**（定义见 ARCHITECTURE / `aux_pseudo_labels.py`）
- [x] 损失与权重；验证指标（`--aux-loss-weight`、`val_aux_mse`）
- [x] **ONNX 导出**（`export_onnx.py`：logits + 可选三头，图中 sigmoid）
- [x] 训练路径 **XRSH only**
- [x] **惯例**：**`xrsh_v2`** + 多头训练 → 辅助标签来自 Rust，与 policy 同源；仅持有旧 **v1** 分片或想排除辅助头噪声时用 **`--no-aux-heads`**

---

## P3 — `engin`：搜索 + UCI

- [ ] UCI：`uci` / `isready` / `position` / `go` / `stop` / `quit` 等最小闭环
- [ ] **Alpha-Beta** + **置换表 TT**
- [ ] **Move ordering**（静态启发 + 日后 policy 排序接口）
- [ ] ONNX Runtime（或约定后端）加载 P2 模型；节点上推理接口
- [ ] 与 `xiangqi_core` 走子、合法性、终局判定联调

---

## P4 — Value Head

**目标：补齐局面真值估计，为搜索稳定性和资源分配提供基础信号**

- [ ] 在 `PolicyResNet` 上扩展可选 `value head`
- [ ] 明确 value 输出契约：范围、激活方式、ONNX 输出名
- [ ] 设计第一版 value 标签来源与回归目标
- [ ] 在 `xiangqi_dataset` 中落地 value 标签写入 XRSH
- [ ] 在 `train_policy.py` 中加入 value loss、权重和验证指标
- [ ] 为 value 头补充 smoke / shape / loss 测试
- [ ] 在 `engin` 中预留 value 推理消费接口

---

## P5 — Search-aware Heads

**目标：让辅助头直接服务搜索，而不是只做训练增强**

- [ ] 固化 `danger` 的定义、标签生成和搜索消费方式
- [ ] 新增 `volatility` head：定义标签与训练目标
- [ ] 新增 `forcing` head：定义标签与训练目标
- [ ] 评估是否加入 `mobility_tension` head，并给出剪枝用途
- [ ] 将上述头统一迁到 Rust 数据侧生成，避免训练步在线多规则源回退
- [ ] 更新 ONNX 导出：明确哪些头输出给引擎消费
- [ ] 为每个 head 补充文档：语义、来源、用途、风险

---

## P6 — Search Distillation

**目标：学习“搜索资源应该放哪里”**

- [ ] 定义蒸馏数据格式：visit count / principal variation / node stats
- [ ] 选择蒸馏来源：本仓引擎搜索日志或外部参考搜索样本
- [ ] 在数据管线中支持搜索分布样本导出
- [ ] 设计训练目标：best move 监督之外的分布监督
- [ ] 比较 policy-only 与 visit distillation 的收益差异
- [ ] 建立固定测试集，评估对 move ordering 与搜索命中率的提升

---

## P7 — Dynamic Search

**目标：由模型输出控制搜索深度、宽度与剪枝策略**

- [ ] 在 `engin` 中定义 head 信号到搜索参数的映射表
- [ ] 将 `danger` 接入 extension / reduction 判定
- [ ] 将 `volatility` 接入宽度分配 / top-k 搜索
- [ ] 将 `forcing` 接入 singular extension / forcing line 深搜
- [ ] 评估 `value` 对 aspiration / pruning / null move 的帮助
- [ ] 建立固定时间预算基准，比较节点利用率与战术命中率
- [ ] 形成可回归测试的搜索配置集

---

## 已完成（归档区）

- **2026-05**：P0 主体 — 自 pikafish-rust 并入 `types` / `misc` / `board` / `movegen`；`Position::from_fen`、`global_zobrist`、`legal_moves_uci`；`tests/perft.rs`（depth 1–3 = 44 / 1926 / 80069）。
- **2026-05**：P1 MVP — `xiangqi_dataset`：`pgn-shards` / `jsonl-shards`，XRSH（`.xrsh`），`pack_meta.json`；`uci_format` 与 pyffish 纵坐标 1～10 对齐。
- **2026-05**：P2 — `PolicyResNet` 可选多头；`aux_pseudo_labels` + `root_fen`/`uci_prefix`/合法 UCI 表；ONNX 多输出；训练 `unpack_*_batch`。
