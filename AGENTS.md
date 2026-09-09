# AGENTS

## 开始前

依次阅读 `README.MD`、`ARCHITECTURE.md`、本文件；按需阅读 `Research.md`、`commands.md` 和
`NextStep.md`。

## 工程共识

- X7 包含 Rust 规则核心、Rust 引擎和 Python 网络训练；`xiangqi_core` 是 px0 规则实现的 Rust 重写，
  `nn` 是 px0 网络与训练格式的 Python 重写。
- 正式模型契约固定为 `124x10x9 -> 2062 + WDL + moves-left`。
- 当前正式搜索是 stream 路径树：edge 首次下探时绑定 arena `NodeId`，不合并换位；规则历史随
  variation/event 裁决，不依赖树拓扑。跨回合只向前复用已展开 child；旧 job drain 后才后台回收 sibling。
- 正式 UCI 不得使用 `UniformBackend`，未实现的 UCI 命令必须明确拒绝。

## 模块边界

- `crates/xiangqi_core`：唯一规则真相。
- `crates/engin`：UCI、网络外围、stream 搜索和时钟。
- `crates/engin/src/search`：owned event 的
  `Select -> Expand -> Eval -> NN -> Reply -> Backprop` 流水线；Select 每次一个叶子，Expand
  处理规则终局与合法着，Eval 处理缓存并只为 miss 编码。
- `nn/`：训练与 ONNX 导出，不进入规则或搜索热路径。

## 约束

- 不引入多套正式训练格式，也不保留 classic 搜索双轨。
- 新的搜索决策应说明目标和不变量。
- 只维护短而稳定的文档：README（背景/目标）、ARCHITECTURE（结构）、Research（历史实验）、
  commands（手册）和 NextStep（当前队列）。不要新增临时长期文档。
