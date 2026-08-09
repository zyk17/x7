# AGENTS

## 开始前

依次阅读：`README.MD`、`ARCHITECTURE.md`、`AGENTS.md`、`NextStep.md`、`TODO.md`。

## 工程共识

- 这是全新 Rust 实现；旧 Rust 核心与旧 MCTS 不是兼容目标。
- `xiangqi_core`、网络外围和 NN 的工程参考为 `C:\Users\Administrator\projects\px0` 与 `C:\Users\Administrator\projects\pxzero-training`。
- stream 仅参考 LC3 官方文档；本地没有 LC3 源码，不得宣称 1:1 翻译。
- 正式模型契约固定为 `124x10x9 -> 2062 + WDL + moves-left`。
- 搜索只有 stream；不维护 classic 对照实现或 `TaskWorkers`。

## 模块边界

- `crates/xiangqi_core`：唯一规则真相，翻译 px0 `src/chess`。
- `crates/engin`：UCI、网络外围、stream 搜索与固定中性时钟管理。
- `crates/engin/src/search`：独立的 streaming MCGS；事件必须 owned 并携带完整 root history 与 variation。node 只按棋盘共享，历史规则仍在 variation 内裁决。
- `nn/`：独立训练与 ONNX 导出，不进入规则或搜索热路径。

## 禁止项

- 不引入多套正式训练格式、未经参考支持的搜索参数或启发式。
- stream 不引入共享可变树/task-worker 模型；MCGS 不新增独立 TT，也不为兼容树 MCTS 保留双轨数据结构。
- 正式 UCI 不得使用 `UniformBackend`；未接入的 UCI 命令必须明确拒绝，不能伪装支持。

## 参考与文档

- 每个新 Rust 函数必须在代码注释、变更说明、`NextStep.md`、`TODO.md` 或 review 记录中标明参考位置。
- px0 找不到连续参考、或 LC3 找不到对应语义时，先记录缺口，再决定是否实现。
- 稳定文档：`README.MD`、`ARCHITECTURE.md`、`AGENTS.md`；活动路线：`NextStep.md`、`TODO.md`。不要新增长期未合并的 `temp*.md` 记录。
