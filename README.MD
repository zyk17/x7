# 77xiangqi_engine

## 我们在做什么

本仓库走的是 **人类认知驱动的搜索（Human-Guided Search）**，而不是「造一个超大的单一网络去替代引擎真理」：

- **模型（轻量 ResNet）**：主要学习 **人类特征**——大师棋谱上的 **policy**、以及与「大局观、危险、先手感」等一致的 **语义头**；目的是给出 **人类风格的候选空间与剪枝先验**，而不是把皮卡鱼式 **NNUE/终评** 照搬进小网络当唯一真理。
- **搜索（Alpha-Beta + TT + 选择性扩展）**：承担 **战术深度、静态评估 fallback、穷举验证** 等 **机器侧行为**；算力主要花在「模型圈定值得算的变化」上，而不是盲目炸树。

一句话：**人类顶尖棋感指路 + 算力在关键枝上纵深验证**；目标是在剪枝与方向上超越纯引擎惯性，在深度上超越人类时限。

本仓库为 **Monorepo**：

| 目录 | 职责 |
|------|------|
| **`crates/xiangqi_core`** | Rust **库**：规则、位棋盘、合法着（对齐 Pikafish / pikafish-rust）；被引擎与数据工具共用 |
| **`crates/engin`** | Rust **可执行文件**：**面向用户的 UCI 引擎**（搜索 + 核心 + ONNX 推理）；**不含** 数据集/标注逻辑 |
| **`crates/xiangqi_dataset`** | Rust **可执行文件**：**维护者专用**——二进制 dataset、并行标注、语料工具（不随引擎分发给终端用户） |
| **`nn/`** | Python：数据管线、ResNet policy 训练、ONNX 导出（`torch` 为可选 `[train]`） |

**产品边界、方法论、网络与搜索分工、数据契约与路线图** 见 **`ARCHITECTURE.md`**。中期研发目标见 **`工程目标.md`**。

## 能力上下限与数据路线（讨论性说明，非 KPI）

从**计算学习**角度看，本架构与「大规模自对弈 + NNUE（如皮卡鱼）」的差异主要在 **先验从哪来**：

- **仅人类特级大师棋谱**（当前主线）：网络 + 搜索的「知识闭包」大致落在 **人类对局曾出现过的着法/局面类型** 内；在尚未见过的最优冷门变化上，policy 质量可能稀疏。**深层战术仍可由搜索补救**，但整体策略分布的上限会受 **训练分布覆盖** 约束——文献型讨论里常把这类路线与「引擎对战 Elo 标尺上的天花板低于无限自对弈引擎」联系在一起（具体数值依评测池、版本而异，**不作本仓库指标**）。
- **皮卡鱼类引擎**：先经由海量自对弈把 NNUE 推到 **超越单纯模仿人类** 的区域，再配合强搜索；其标尺上的强度可与「纯人类模仿」路线定性区分。
- **若既要强度又要人味**：可考虑 **在人类候选附近宽松探索**——例如在 policy 或语义约束下的混合数据或受限自对弈，用于补足盲点；与「完全自由自对弈直至棋风机器化」是不同档位的产品权衡。

**重要**：各类 **Elo 数字** 均为 **相对标尺下的量级讨论**，对手池、时限、版本一变即变；**象棋与围棋 Elo 不可横向比较**。仓库代码与里程碑 **不以某一估算 Elo 为验收标准**。已定稿的路线摘要见 **`docs/design-summary.md`**（**非** ONNX/XRSH 契约）。

## 环境

- **Python**：在 `nn/` 下使用虚拟环境（例如 `nn/.venv`），`pip install -e ".[train]"`。
- **Rust**：`cargo build --workspace`；引擎：`cargo build -p engin`；数据工具：`cargo build -p xiangqi_dataset`。

## Python 训练栈（`nn/`）

**词表（`build_vocab`）→ Rust 生成 XRSH → `train_policy` / ONNX** 的完整步骤见 **`nn/README.md`** 中「数据集创建与使用流程」。

```bash
cd nn
pip install -e .
pip install -e ".[train]"
pytest
```

## Rust

```bash
cargo build -p xiangqi_core
cargo run -p engin                   # **UCI**：自 stdin 读协议（GUI 对弈）；`setoption name PolicyFile value …` / `VocabFile value …` 可接 ONNX + `move_vocab.json` 用 policy 选着
cargo run -p engin -- --onnx-smoke   # 加载 data/policy.onnx 对起始局面推理（路径见 --help）
cargo run -p xiangqi_dataset    # 训练侧数据集/标注 CLI（当前占位）
```

（`cargo run` 传给**二进制**的参数必须写在 **`--` 之后**；若写 `cargo run --onnx-smoke` 会被当成 Cargo 自己的选项而报错。）

**当前优先**：见根目录 **`TODO.md`**「推进顺序」。规则实现已在 **`xiangqi_core`**（合法着、perft）；下一步重点是 **与 pyffish / Pikafish 行为对拍**，再放心大规模造数；棋谱规则边界可参考本地 **`c:\projects\Pikafish`**、结构对齐 **`c:\projects\pikafish-rust`**（不纳入本仓库）。

## 文档索引

| 文件 | 内容 |
|------|------|
| `ARCHITECTURE.md` | 人类认知驱动搜索、产品与契约、crate 划分、数据格式、里程碑摘要 |
| `MILESTONES.md` | P0–P7 里程碑定义与依赖顺序 |
| `TODO.md` | 分阶段执行清单（勾选推进） |
| `工程目标.md` | P3 之后的中期工程目标：value、搜索控制头、蒸馏、动态搜索 |
| `crates/xiangqi_dataset/README.md` | `pgn-shards` / `jsonl-shards` 与 **XRSH**（`.xrsh`）格式 |
| `agents.md` | 给自动化助手 / Agent 的仓库约定 |
| `docs/design-summary.md` | 人类认知驱动搜索、能力边界与 Elo 声明的 **归档摘要**（非数据契约） |
