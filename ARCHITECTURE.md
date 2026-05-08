# 架构说明（产品、实现与路线）

## 产品与方法论（Human Policy & Cognition）

### 一句话

**用人类顶尖棋感指引方向，用算力验证深度**——轻量网络提供候选着与局面语义，**神经引导的 Alpha-Beta** 负责中局战术深度；残局更多依赖搜索与残局库而非单一分值。

### 系统定位（摘要）

| 维度 | 选择 |
|------|------|
| 身份 | **人类知识引导的深度搜索**：以大师棋谱为先验、**语义引导的选择性搜索**；中局战术深度优先，非纯「大宽树硬算」 |
| 搜索骨架 | **Alpha-Beta**（非 MCTS） |
| 网络 | 小 **ResNet**；输入 **`15×10×9`**；**当前** 单 **policy**；**路线** 上为 shared trunk + **policy / attack / danger / tactical**（及可选 **phase**） |
| 数据 | 人类特级大师棋谱，**不自对弈** 为主；划分须按整局 **`game_id`** |
| 搜索侧 | **Policy 排序** → Top-k 选择性搜索 → 由多头语义驱动 **动态宽度与 extension**；**NN 与 Search 解耦**（ONNX / CPU / CUDA / 未来 NPU） |

**导出**：**ONNX**，静态 batch=1；带宽与体积由 width/blocks 与量化控制。

### 边界（不做或暂不默认交付）

- **不做**：复盘 UI、自然语言解说产品、Rust/JS 宿主的具体集成（本仓库可提供契约与库）。
- **标签**：**不使用 Elo / 等级分** 作为训练条件（噪声大，难以表达认知分布）。
- **引擎**：完整 UCI 对弈引擎、开局库可列为 **后续里程碑**；当前阶段优先 **规则 / 合法着 / 标注管线** 与训练数据栈。

### 数据与标签

- **来源**：特级大师棋谱（ICCS / PGN）；人类 **不自对弈** 生成主标签。
- **样本**：`position → move`；划分必须 **按整局 `game_id`**，禁止局面级随机切分导致泄漏。
- **坐标**：棋谱侧 ICCS → **pyffish UCI**（与皮卡鱼 / 77 象棋一致）；权威解析见 Python 包 `notation_iccs`、`board`（在向 Rust 迁移规则前仍以 pyffish 为准）。

### 路线摘要（实现以代码与本文件里程碑为准）

- **语义**：先让网络学会「局面语义」，再谈盲目扩数据量。
- **多头**：第一版建议 **policy + attack + danger + tactical**（phase 可后加）；标签宜 **规则/浅搜自动生成**，避免手工标。
- **搜索**：语义用于 **move ordering、选择性宽度、extension**；残局侧倾向 **纯搜索 + 残局库**，可参考皮卡鱼在残局少用 NNUE 评估的思路。

### 参考工程（本地可选克隆）

- **`c:\projects\Pikafish`**：规则与搜索语义权威之一。
- **`c:\projects\pikafish-rust`**：Rust 移植；本仓库 **象棋规则与标注加速** 对齐其实现分层（`board`、`movegen` 等）。

---

## 仓库定位（Monorepo）

1. **`xiangqi_core`（Rust 库）**：规则、局面表示、合法着；**所有** 可执行体与 Python 训练栈**共用**。
2. **`nn`（Python）**：PGN → JSONL → 索引 / policy pack；**ResNet** 训练与 **ONNX** 契约。
3. **`engin`（Rust 二进制）**：**分发给终端用户** 的 **UCI 引擎**——搜索 + `xiangqi_core` + 神经网络（ONNX）推理；**不包含** 数据标注/数据集工具。
4. **`xiangqi_dataset`（Rust 二进制）**：**维护者/训练侧** 专用——二进制 dataset 生成、按局并行、规则侧标注等；**不** 与「用户引擎」同一发布物。

## 分层（依赖方向）

```
                    ┌──────────────────────────────────┐
        终端用户     │  UCI：`engin`（搜索+NN+core）     │
        维护者       │  CLI：`xiangqi_dataset`（数据管线）│
                    └─────────────────────────────────┘
                                      │
                                      ▼
                              ┌─────────────┐
                              │ xiangqi_core │
                              │ 规则·MoveGen  │
                              └─────────────┘
                                      ▲
                                      │
                              ┌───────┴───────┐
                              ▼               ▼
                     ┌─────────────┐   ┌─────────────┐
                     │ ONNX（推理）  │   │ nn（训练栈）  │
                     └─────────────┘   └─────────────┘
```

- **向下依赖**：Python 训练栈当前仍用 **pyffish**；Rust 规则成熟后，**数据集与标注** 优先走 **`xiangqi_core` + `xiangqi_dataset`**，减少热路径 Python。
- **横向**：ONNX 契约固定张量形状；搜索与 NN 解耦。

## Python 包布局（`nn/src`）

| 区域 | 路径 | 说明 |
|------|------|------|
| 数据与棋谱 | `constants.py`、`board.py`、`notation_iccs.py`、`pgn.py`、`dataset_pgn.py` | ICCS ↔ UCI、行迭代 |
| 增强 | `augment_mirror.py` | 水平镜像 |
| 神经网络 | `nn/` | `fen_tensor`、`model`、`dataset`、`policy_pack`、`materialize_pack` 等 |

脚本：`nn/scripts/data_pgn/*`、`nn/scripts/train/train_policy.py`、`nn/scripts/export/export_onnx.py`。

## 数据管线（现状）

- **JSONL 行**：`fen`、`root_fen`、`uci_prefix`、`human_move_pyffish`、`game_id`、`pgn_source` 等（详见 `extract_rows`）。
- **大规模训练（Python）**：`build_jsonl_index` → mmap；可选 **`policy_pack_v2`** 离线包（`nn` 内 `materialize_policy_pack`）。
- **二进制分片（Rust，`xiangqi_dataset`）**：**`XQB` v1**（`shard_*.xqb`）  
  - 子命令 **`pgn-shards`**：读 PGN / `.pgns`，按局 **Rayon 并行** 编码；**`jsonl-shards`**：读已有 JSONL。  
  - 每样本含：`fen`、`root_fen`、`uci_prefix`、**词表下标**的合法着列表、`target`、`ply`；`pack_meta.json` 含 **`vocab_sha256`**（与 Python `vocab_fingerprint_ordered_moves` 一致）。  
  - 细节与 CLI 见 **`crates/xiangqi_dataset/README.md`**。P2 在 `nn` 侧增加 XQB 读取器后即可替代或并存 `policy_pack_v2`。

## 模型与 ONNX（Python）

- **输入**：`float32[1,15,10,9]`。
- **输出**：`float32[1,V]` policy logits；推理时对非法位掩码后再 softmax。

路线上的 **多头（attack / danger / tactical）** 需在契约中追加输出名与形状；当前代码以 **单 policy** 为主。

## Rust Crate 划分

| Crate | 路径 | 受众 | 职责 |
|--------|------|------|------|
| **`xiangqi_core`** | `crates/xiangqi_core` | 库 | 类型、规则、合法着（实现自 **pikafish-rust** 迁入，见 crate 内 `README.md`）；搜索与工具共用 |
| **`xiangqi_dataset`** | `crates/xiangqi_dataset` | **维护者** | 数据集生成、标注、语料工具 CLI（**不** 随引擎分发） |
| **`engin`** | `crates/engin` | **终端用户** | **UCI 引擎**：搜索 + `xiangqi_core` + NN（ONNX） |

规则逻辑始终集中在 **`xiangqi_core`**；`engin` 与 `xiangqi_dataset` 均只依赖该库（引擎侧后续可增加 ONNX 等依赖）。

## 里程碑（执行顺序）

与 **`MILESTONES.md`**（阶段定义、依赖图）及 **`TODO.md`**（可勾选任务）同步维护。

| 优先级 | 内容 |
|--------|------|
| **P0** | **`xiangqi_core`**：完整象棋规则 + **合法 UCI**（对齐 **pikafish-rust** / Pikafish） |
| **P1** | **`xiangqi_dataset`**：数据管线（**PGN → 二进制 shards**，**按局并行**） |
| **P2** | **`nn/`**：Python 训练接入二进制数据包 + **多头网络** |
| **P3** | **`engin`**：搜索模块（**AB、TT、move ordering**）+ **UCI 协议** |

## 外部参考路径（本地）

- `c:\projects\Pikafish`
- `c:\projects\pikafish-rust`
