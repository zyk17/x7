# 77xiangqi_engine

中国象棋 **人类知识引导的深度搜索** 方向：轻量 CNN 输出 policy 与（路线上的）多头语义，配合 **Alpha-Beta + 选择性扩展**；训练数据来自 **大师棋谱**，不以自对弈为主。

本仓库为 **Monorepo**：

| 目录 | 职责 |
|------|------|
| **`crates/xiangqi_core`** | Rust **库**：规则、位棋盘、合法着（对齐 Pikafish / pikafish-rust）；被引擎与数据工具共用 |
| **`crates/engin`** | Rust **可执行文件**：**面向用户的 UCI 引擎**（搜索 + 核心 + ONNX 推理）；**不含** 数据集/标注逻辑 |
| **`crates/xiangqi_dataset`** | Rust **可执行文件**：**维护者专用**——二进制 dataset、并行标注、语料工具（不随引擎分发给终端用户） |
| **`nn/`** | Python：数据管线、ResNet policy 训练、ONNX 导出（`torch` 为可选 `[train]`） |

**产品边界、方法论、分层与路线图** 见 **`ARCHITECTURE.md`**。中期研发目标见 **`工程目标.md`**。

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
| `ARCHITECTURE.md` | 产品与方法边界、crate 划分、数据格式、里程碑摘要 |
| `MILESTONES.md` | P0–P7 里程碑定义与依赖顺序 |
| `TODO.md` | 分阶段执行清单（勾选推进） |
| `工程目标.md` | P3 之后的中期工程目标：value、搜索控制头、蒸馏、动态搜索 |
| `crates/xiangqi_dataset/README.md` | `pgn-shards` / `jsonl-shards` 与 **XRSH**（`.xrsh`）格式 |
| `agents.md` | 给自动化助手 / Agent 的仓库约定 |
