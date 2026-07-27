# 77xiangqi_engine

中国象棋 `policy + WDL` 网络与 MCTS 引擎。项目是 px0 的新 Rust 实现；旧 Rust 核心和旧 MCTS 不是兼容目标。

## 当前状态

- `xiangqi_core` 已完成棋盘、合法着、FEN、`PositionHistory` 与规则裁判。
- 正式 UCI 走 **classic MCTS**，已接通 ONNX、缓存、minibatch、prefetch、tree reuse、时控、watchdog 与完整会话生命周期。
- `search::stream` 是正在迁移的下一代 MCTS：常驻 Gather / Eval / NN / Backprop 流水线，默认 `4 / 2 / 1 + 1×NN`。它可在库内搜索、对拍和基准测试，尚未接 UCI。
- classic 保留为稳定行为基线；不再推进 classic `TaskWorkers` 的共享可变树并行路线。

正式模型契约固定为：`124 x 10 x 9 -> 2062 policy + WDL[win, draw, loss]`，引擎 value 为 `win - loss`。

## 使用 UCI

正式搜索必须先成功加载 ONNX；不会回退到 `UniformBackend`。

```text
setoption name WeightsFile value C:/models/x7.onnx
position startpos
go nodes 1000
```

支持 `go nodes`、`go movetime`、`go infinite` 和 legacy 时钟字段；`go depth`、`go mate`、`go ponder`、`ponderhit` 会明确拒绝。

启动时按顺序查找 `engin.exe` 同目录、当前目录和 `data/x7.onnx`。Windows 发行版使用 DirectML，初始化失败时回退 CPU。

## 验证与构建

```powershell
cargo fmt --check
cargo test -p engin --lib
cargo clippy -p engin --all-targets -- -D warnings
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1
```

训练子项目位于 `nn/`，详见 [nn/README.md](nn/README.md)。当前任务和 stream 的能力边界见 [TODO.md](TODO.md)、[NextStep.md](NextStep.md) 与 [temp.md](temp.md)。
