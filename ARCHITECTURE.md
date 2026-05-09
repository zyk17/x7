# 架构说明（产品、实现与路线）

## 产品与方法论（Human Policy & Cognition）

### 核心立场：人类认知驱动的搜索

本仓库的引擎哲学是 **Human-Guided Search（人类引导的深度搜索）**，**不是**「训练一个越来越强、越来越像传统引擎静态评估的巨型网络」：

| 组件 | 职责（应当） |
|------|----------------|
| **神经网络** | 学习 **人类特征**：大师 **policy**、与棋感相关的 **语义头**（危险、物质压力、战术警觉等）。输出用于 **候选排序、剪枝先验、何处值得加深**，侧重 **大局观与风格**，而非独自背负「局面终极真理」。 |
| **搜索（Alpha-Beta + TT + 扩展/静止搜索）** | 承担 **战术穷尽、静态评估（物质等）、深度验证** 等 **机器侧行为**；**终极对错与深层战术** 主要由搜索与规则核心验证，而不是勉强塞进小网络。 |

### 双主线：复盘系统 + 搜索引擎

本仓库存在两条并行主线：

- **短期主产品：复盘系统**
  - 模型优先学习**可解释的人类风格与局面语义**
  - 直接服务候选着解释、危险提示、战术提醒、复盘分析
- **长期路线：搜索引擎**
  - 搜索用于验证模型是否真的学会了有用的棋感
  - 再逐步把这些语义信号转化为搜索收益

因此：

- 模型不只是给搜索打工
- 搜索也不只是独立追求棋力
- 两者关系是**相辅相成**：模型提供可解释人类先验，搜索负责验证这些先验是否真的成立

### 一句话

**用人类顶尖棋感缩小搜索空间并指路，用算力在关键枝上验证深度**——轻量网络提供 **人类风格候选着与局面语义**，**神经引导的 Alpha-Beta** 在缩过的树上做战术深度；残局更多依赖搜索与残局库而非单一分值。

### 系统定位（摘要）

| 维度 | 选择 |
|------|------|
| 身份 | **人类知识引导的深度搜索**：以大师棋谱为先验、**语义引导的选择性搜索**；追求 **高效剪枝 + 定向加深**，非「单靠大模型背评估」 |
| 搜索骨架 | **Alpha-Beta**（非 MCTS） |
| 网络 | 小 **ResNet**；输入 **`15×10×9`**；训练默认可含 **policy + attack / danger / tactical** 等辅助头，后续可扩展 **value**；导出 ONNX 见下文契约（可选仅 **`logits`**）——设计上避免把网络当作 **完整 Pikafish 式静态评估的替代品** |
| 数据 | 人类特级大师棋谱，**不自对弈** 为主；划分须按整局 **`game_id`** |
| 搜索侧 | **Policy 排序** → Top-k 选择性搜索 → 由多头语义驱动 **动态宽度与 extension**；**NN 与 Search 解耦**（ONNX / CPU / CUDA / 未来 NPU） |

**导出**：**ONNX**，静态 batch=1；带宽与体积由 width/blocks 与量化控制。

### 边界（不做或暂不默认交付）

- **不做**：复盘 UI、自然语言解说产品、Rust/JS 宿主的具体集成（本仓库可提供契约与库）。
- **标签**：**不使用 Elo / 等级分** 作为训练条件（噪声大，难以表达认知分布）。
- **默认不把「蒸馏皮卡鱼 NNUE/终评当作唯一 value 真理」作为产品主线**：可作为对照实验或校准手段；**主线 value** 应服务于 **人类局面理解 / 剪枝**，详见 **`TODO.md` P4** 表述。
- **引擎**：完整 UCI 对弈引擎、开局库可列为 **后续里程碑**；当前阶段优先 **规则 / 合法着 / 标注管线** 与训练数据栈。

### 数据与标签

- **来源**：特级大师棋谱（ICCS / PGN）；人类 **不自对弈** 生成主标签。
- **样本**：`position → move`；划分必须 **按整局 `game_id`**，禁止局面级随机切分导致泄漏。
- **坐标字符串**：棋谱侧 ICCS → **pyffish 风格 UCI 字符串**（与皮卡鱼 / 77 象棋纵坐标 1～10 一致）；ICCS 解析在 **`crates/xiangqi_dataset/src/iccs.rs`**。
- **合法着（主路径）**：**XRSH** 由 Rust **`xiangqi_dataset`** 使用 **`xiangqi_core`** 枚举并写入每样本 **合法着词表下标**；`train_policy` / **`PolicyXrshDataset`** 仅消费掩码与下标，**不在训练热路径上再用 pyffish 枚举全谱合法着**（当前默认 **`xrsh_v2`**）。**pyffish（Python `board`）** 用于 **对拍/冒烟**、水平镜像与 **`nn.aux_pseudo_labels`** 在 **旧 v1 分片或缺失辅助字段** 时的 **回退**，而非主数据契约的规则源。

### 路线摘要（实现以代码与本文件里程碑为准）

- **语义**：先让网络学会「人类关心的局面语义」（危险、进攻压力等），再谈盲目扩数据量。
- **多头**：第一版建议 **policy + attack + danger + tactical**（phase 可后加）；标签宜 **规则/浅规则自动生成**，服务于 **搜索调度与人类棋感**，而非替代引擎全文搜索。
- **搜索**：语义用于 **move ordering、选择性宽度、extension**；**战术与终极裁决** 依赖搜索；残局侧倾向 **纯搜索 + 残局库**。

当前推荐优先级：

- **先做解释性强、能进入复盘系统的头**
- 再从中筛出真正能改善搜索的头

暂不作为近期主线的标签：

- `style`
- `sacrifice`
- `initiative`
- `psychological`

### 能力边界与可选数据扩展（讨论性）

- **人类棋谱为主**时，策略先验主要来自 **人类对局分布**；对「人类历史中极少出现、但搜索可证明成立」的着法，监督信号可能稀疏——这是 **分布覆盖** 问题，不是搜索能否算清的问题。
- **大规模自对弈引擎**（如皮卡鱼）的先验可 **自我迭代到人类谱之外**；在相同「引擎 vs 引擎」标尺下，纯模仿人类的路线与该类引擎的 **相对排名** 需用实测衡量；文献或内部讨论中的 **具体 Elo 区间** 仅作量级参考，**本仓库不作验收 KPI**。
- **可选折中**（产品决策，非当前默认）：在 **保留人类风格约束** 的前提下引入 **受限自对弈** 或 **混合训练**（例如在 policy 质量门控或候选邻域内探索），以填补盲点；与无约束自对弈「棋风彻底机器化」是不同取舍。讨论摘要见 **`docs/design-summary.md`**。

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

- **向下依赖**：**数据集与 XRSH 标注主路径** 已落在 **`xiangqi_core` + `xiangqi_dataset`**；Python **`train_policy`** 以 **XRSH** 为唯一训练数据源。**pyffish** 保留为 **镜像、parity、legacy 样本回退** 等辅助能力，不是训练步合法集的主规则源。
- **横向**：ONNX 契约固定张量形状；搜索与 NN 解耦。

## Python 包布局（`nn/src`）

| 区域 | 路径 | 说明 |
|------|------|------|
| 数据与规则面 | `constants.py`、`board.py`（pyffish） | parity / 镜像 / legacy；**语料物化与合法着枚举**在 Rust（`xiangqi_dataset` → **XRSH**，规则源 **`xiangqi_core`**） |
| 增强 | `augment_mirror.py` | 水平镜像 |
| 神经网络 | `nn/` | `fen_tensor`、`model`、`dataset_xrsh`、`policy_pack`（词表指纹）、`xrsh_io` 等 |

脚本：`nn/scripts/vocab/build_vocab.py`；`nn/scripts/train/train_policy.py`（**仅 XRSH**）、`export/export_onnx.py`。

## 数据管线（现状）

- **训练用数据**：**XRSH** 分片（`shard_*.xrsh` + `pack_meta.json`），由 Rust **`xiangqi_dataset`** 从 PGN 或 JSONL 生成。中间 **JSONL** 仍可用于 `build_vocab.py` 与 `jsonl-shards` 输入，但 **`train_policy` 不再直接读 JSONL / mmap 索引 / policy npy 包**。
- **二进制分片（Rust，`xiangqi_dataset`）**：**XRSH**（Xiangqi Review Shard），文件 **`shard_*.xrsh`**（魔数 `XRSH`）。**当前写入 `xrsh_v2`**（文件头版本 2）：在 v1 字段基础上每样本追加 **`aux_attack` / `aux_danger` / `aux_tactical`**（`float32×3`，由 **`xiangqi_core`** 预计算，与 `nn.aux_pseudo_labels` 公式一致）。**v1 分片仍可读**（Python `xrsh_io` 兼容文件版本 1）。  
  - 子命令 **`pgn-shards`**：读 PGN / `.pgns`，按局 **Rayon 并行** 编码；**`jsonl-shards`**：读已有 JSONL。  
  - 每样本至少含：`fen`、`root_fen`、`uci_prefix`、**词表下标**的合法着列表、`target`、`ply`；v2 含上述三辅助标量；`pack_meta.json` 含 **`format`**（`xrsh_v2`）、**`vocab_sha256`**。  
  - 细节与 CLI 见 **`crates/xiangqi_dataset/README.md`**。Python 训练仅 **`nn.dataset_xrsh.PolicyXrshDataset`**（`train_policy.py --train-xrsh-dir`）；若样本含辅助字段则 **训练步不再对辅助头调 pyffish**。

## 模型与 ONNX（Python）

- **输入**：`float32[1,15,10,9]`。
- **输出**：
  - **仅 policy**：`logits`，`float32[1,V]`；推理时对非法位掩码后再 softmax。
  - **多头（`aux_heads`，默认训练开启）**：在上述基础上追加 **`attack`、`danger`、`tactical`**，均为 **`float32[1]`**。**ONNX 图中**三辅助输出已含 **sigmoid**（见 `export_onnx.py` 的 `PolicyOnnxExport`），数值约在 `[0,1]`；训练仍对原始 logit 做 sigmoid 后与伪标签对齐。语义头设计目的是 **人类式局面提示**、**复盘系统解释** 与 **搜索调度**，不是单独复刻引擎评估曲线。
  - **可选 `value` 输出**（若训练导出）：见 **`TODO.md` P4**；语义上为 **人类局面感 / 复盘辅助 / 剪枝辅助**，与 **`engin`** 叶子评估（可与物质 fallback 协作）一致即可。
  - **XRSH v2** 已含 Rust 预计算辅助伪标签；读 **v1** 或缺失字段时回退 `nn/src/nn/aux_pseudo_labels.py`（pyffish）。物质项与机动均应与 **`xiangqi_core`** 一致后再依赖纯 Rust 路径。

## Rust Crate 划分

| Crate | 路径 | 受众 | 职责 |
|--------|------|------|------|
| **`xiangqi_core`** | `crates/xiangqi_core` | 库 | 类型、规则、合法着（实现自 **pikafish-rust** 迁入，见 crate 内 `README.md`）；搜索与工具共用 |
| **`xiangqi_dataset`** | `crates/xiangqi_dataset` | **维护者** | 数据集生成、标注、语料工具 CLI（**不** 随引擎分发） |
| **`engin`** | `crates/engin` | **终端用户** | **UCI（stdin/stdout）** + `xiangqi_core` + **`ort`**；**`setoption`**：`PolicyFile` / `VocabFile` / `Hash` / `Threads` / `MultiPV` / **`Clear Hash`**；**`go`** 支持 `infinite`+`stop`（后台线程）；无搜索树时 `bestmove` 由 policy（若已加载且词表维匹配）或合法着首项给出 |

规则逻辑始终在 **`xiangqi_core`**；`engin` 另依赖 **`ort`/`ndarray`** 做推理；`xiangqi_dataset` 仅依赖核心库。

## 里程碑（执行顺序）

与 **`MILESTONES.md`**（阶段定义、依赖图）及 **`TODO.md`**（可勾选任务）同步维护。

| 优先级 | 内容 |
|--------|------|
| **P0** | **`xiangqi_core`**：完整象棋规则 + **合法 UCI**（对齐 **pikafish-rust** / Pikafish） |
| **P1** | **`xiangqi_dataset`**：数据管线（**PGN → 二进制 shards**，**按局并行**） |
| **P2** | **`nn/`**：Python 训练接入二进制数据包 + **多头网络** |
| **P3** | **`engin`**：搜索模块（**AB、TT、move ordering**）+ **UCI 协议** |
| **P4+** | **语义头 / value / 动态搜索**：见 **`MILESTONES.md`**；主线始终是 **人类特征 → 剪枝与调度**，引擎式精确评估交给 **搜索** |

## 外部参考路径（本地）

- `c:\projects\Pikafish`
- `c:\projects\pikafish-rust`
