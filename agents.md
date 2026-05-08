# 仓库 Agent 指南

面向在本仓库内工作的自动化助手 / CI Agent。

## 必读

1. **`ARCHITECTURE.md`** — 产品与方法边界、Rust/Python 分层、数据契约、路线图。
2. **`.cursorrules`** — 沟通语言（简体中文）、文档同步约定。

## 目录约定

- **Python 包**：在 `nn/` 下开发与安装；虚拟环境放在 `nn/.venv`（或用户自定，勿提交 venv）。
- **Rust**：仓库根 `Cargo.toml` workspace；核心库 **`crates/xiangqi_core`**；**用户 UCI 引擎** **`crates/engin`**；**数据管线/标注 CLI** **`crates/xiangqi_dataset`**（维护者用，与引擎发布物分离）。

## 修改契约时

若变更 **ONNX 输入输出**、**policy pack 格式**、**JSONL 字段** 或 **Rust 二进制 dataset 头格式**，必须同步：

- `ARCHITECTURE.md`
- 根目录 `README.MD` 或 `nn/README.md` 中的命令示例（如有影响）

## 实现优先级（与维护者路线一致）

1. **象棋规则与合法着（Rust）** — 参考 `pikafish-rust` 的 `board.rs`、`movegen.rs`，保持与 Pikafish 语义一致。
2. **数据生成 / 标注加速（Rust）** — 多线程按 `game_id` 分片；输出二进制便于 mmap。
3. **Python** — 保留训练与导出；避免在热路径重复实现规则。

## 测试

- Python：`cd nn && pytest`（无 `torch` 时跳过 NN smoke）。
- Rust：`cargo test -p xiangqi_core`。

## 勿做

- 未在任务中要求时，不要大篇幅重写与我方边界无关的模块。
- 不要将训练代码与搜索引擎绑死在同一进程（未来通过 API / ONNX）。
- **数据标注、二进制 dataset 生成** 放入 **`xiangqi_dataset`**，不要塞进 **`engin`**（引擎面向终端用户）。
