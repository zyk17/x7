# AGENTS

## 开始前

依次阅读：`README.MD`、`ARCHITECTURE.md`、`agents.md`；按需阅读 `MCGS.md`、`Research.md`、`commands.md`。延后低优先级项见 `NextStep.md`。

## 工程共识

- 这是独立的 X7 Rust 引擎。第一版规则、网络外围、UCI 与训练格式参考过 px0 / Lc0；**当前不是 px0 重写，也不以兼容 px0/LC3 为目标**。
- 旧 Rust 核心与旧树 MCTS 不是兼容目标。`mcts` 分支可保留树 MCTS 对照，本仓正式搜索只有 stream MCGS。
- 规则与训练数据格式的历史工程参考：`C:\Users\Administrator\projects\px0`、`C:\Users\Administrator\projects\pxzero-training`。
- stream / MCGS 可参考 [LC3 公开文档](https://lczero.org/dev/lc0/search/lc3/overview/)；本地没有 LC3 源码，不得宣称 1:1 翻译或行为等价。
- KataGo 按需参考：本地源码 `C:\Users\Administrator\projects\KataGo`（如 GraphSearch、NN cache、部分搜索细节）；不是默认必读，也不承诺行为等价。
- 已明确偏离 px0/Lc0 stream 基线：无 multivisit、无 prefetch、无 tree-batch gather。PUCT 使用 edge in-flight reservation 作为 virtual visit（计入 started N，偏转选择）；碰撞取消未完成路径的 reservation。
- 正式模型契约固定为 `124x10x9 -> 2062 + WDL + moves-left`。
- 搜索只有 stream；不维护 classic 对照实现或 `TaskWorkers`。

## 模块边界

- `crates/xiangqi_core`：唯一规则真相；棋盘/合法着/FEN/历史/裁判语义历史上源于 px0 `src/chess`，现由本仓维护。
- `crates/engin`：UCI、网络外围、stream MCGS 与固定中性时钟管理。
- `crates/engin/src/search`：独立的 streaming MCGS；事件必须 owned 并携带完整 root history 与 variation。node 只按棋盘共享，历史规则仍在 variation 内裁决。
- `nn/`：独立训练与 ONNX 导出，不进入规则或搜索热路径。

## 禁止项

- 不引入多套正式训练格式。
- 不把「找不到外部参考」当成自动否决；自研搜索决策可以做，但必须在注释或专题文档写清是 X7 决策，而不是伪装成 px0/LC3 等价。
- stream 不引入共享可变树/task-worker 模型；MCGS 不新增独立 TT，也不为兼容树 MCTS 保留双轨数据结构。
- 正式 UCI 不得使用 `UniformBackend`；未接入的 UCI 命令必须明确拒绝，不能伪装支持。

## 参考与文档

- 沿用或借鉴外部语义时，在注释里保留来源（px0 路径、LC3 URL、KataGo 本地路径/文档等），并标明是历史/语义参考还是本仓已偏离。
- 稳定文档：`README.MD`、`ARCHITECTURE.md`、`agents.md`；专题：`MCGS.md`、`Research.md`、`commands.md`；延后项：`NextStep.md`。不要新增长期未合并的 `temp*.md` 记录。
