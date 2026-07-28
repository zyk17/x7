# 77xiangqi_engine

中国象棋 `policy + WDL` 网络与 MCTS 引擎。项目是 px0 的新 Rust 实现；旧 Rust 核心和旧 MCTS 不是兼容目标。

## 当前状态

- `xiangqi_core` 已完成棋盘、合法着、FEN、`PositionHistory` 与规则裁判。
- 正式 UCI 默认走 **stream MCTS**：常驻 Gather / Eval / NN / Backprop 流水线（默认 `4 / 2 / 1 + 1×NN`）、ONNX、缓存、tree reuse/prune、Watchdog 与完整会话生命周期。
- classic 保留为独立的对照实现；不再推进 classic `TaskWorkers` 的共享可变树并行路线。

正式模型契约固定为：`124 x 10 x 9 -> 2062 policy + WDL[win, draw, loss] + moves-left`，引擎 value 为 `win - loss`。

## 使用 UCI

正式搜索必须先成功加载 ONNX；不会回退到 `UniformBackend`。

```text
setoption name WeightsFile value C:/models/x7.onnx
position startpos
go nodes 1000
```

支持 `go nodes`、`go movetime`、`go infinite` 与 `searchmoves`；`UCI_ShowWDL`、`UCI_ShowEPS` 与 `WeightsFile` 是当前最小 option 集。`go depth`、`go mate`、时钟字段、`go ponder` 与 `ponderhit` 会明确拒绝，直至对应语义完成。

启动时按顺序查找 `engin.exe` 同目录、当前目录和 `data/x7.onnx`。Windows 发行版使用 DirectML，初始化失败时回退 CPU。

## 验证与构建

```powershell
cargo fmt --check
cargo test -p engin --lib
cargo clippy -p engin --all-targets -- -D warnings
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1
```

训练子项目位于 `nn/`，详见 [nn/README.md](nn/README.md)。当前任务和 stream 的能力边界见 [TODO.md](TODO.md) 与 [NextStep.md](NextStep.md)。
