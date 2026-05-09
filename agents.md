# 仓库 Agent 指南

面向在本仓库内工作的自动化助手 / CI Agent。

## 必读

1. **`ARCHITECTURE.md`** — **人类认知驱动的搜索**（网络学人类特征与剪枝先验；战术与静态评估真理主要由搜索承担）、Rust/Python 分层、数据契约、路线图。
2. **`.cursorrules`** — 沟通语言（简体中文）、文档同步约定、产品哲学摘要。

## 目录约定

- **Python 包**：在 `nn/` 下开发与安装；虚拟环境放在 `nn/.venv`（或用户自定，勿提交 venv）。
- **Rust**：仓库根 `Cargo.toml` workspace；核心库 **`crates/xiangqi_core`**；**用户 UCI 引擎** **`crates/engin`**；**数据管线/标注 CLI** **`crates/xiangqi_dataset`**（维护者用，与引擎发布物分离）。

## 修改契约时

若变更 **ONNX 输入输出**、**policy pack 格式**、**JSONL 字段** 或 **Rust 二进制 dataset 头格式**，必须同步：

- `ARCHITECTURE.md`
- 根目录 `README.MD` 或 `nn/README.md` 中的命令示例（如有影响）

## 实现优先级（与维护者路线一致）

1. **象棋规则与合法着（Rust）** — 参考 `pikafish-rust` 的 `board.rs`、`movegen.rs`，保持与 Pikafish 语义一致。
2. **数据生成 / 标注加速（Rust）** — 多线程按 `game_id` 分片；输出二进制便于 mmap；标签服务于 **人类棋感与搜索调度**，不把「蒸馏外部引擎终评」默认为唯一主线。
3. **Python** — 保留训练与导出；避免在热路径重复实现规则。

## 测试

- Python：`cd nn && .venv\Scripts\python.exe -m pytest`（或 `pytest`，均在 venv 下；无 `torch` 时跳过 NN smoke）。
- Rust：`cargo test -p xiangqi_core`；`cargo test -p engin`（含 `PolicyOnnx` / FEN 平面单测；若存在 `data/policy.onnx` 则跑 ORT 推理冒烟；首编译可能经 `ort` 拉取 ONNX Runtime）。

## Rust 格式与静态检查

提交或合并前建议在仓库根目录执行：

- **`cargo fmt --all`** — 格式规则见根目录 **`rustfmt.toml`**；仅检查可加 **`cargo fmt --all -- --check`**。
- **`cargo clippy --workspace --all-targets`** — 消除可避免告警（本项目常规修正：`manual_contains`、`needless_range_loop`、`explicit_auto_deref`、`too_many_arguments` 等）。

Python 侧无强制格式化命令；本地可用 Black/isort 等**手动**保持一致。

## 勿做

- 未在任务中要求时，不要大篇幅重写与我方边界无关的模块。
- 不要将训练代码与搜索引擎绑死在同一进程（未来通过 API / ONNX）。
- **数据标注、二进制 dataset 生成** 放入 **`xiangqi_dataset`**，不要塞进 **`engin`**（引擎面向终端用户）。
- 默认假设「越大越像 Pikafish 静态评估越好」——与本仓库 **人类引导搜索** 的定位不符；价值头与蒸馏若以引擎为 Teacher，应在文档与任务中 **显式说明用途（对照/实验）**，避免 silent 扭转产品哲学。
