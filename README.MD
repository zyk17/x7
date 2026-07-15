# 77xiangqi_engine

中国象棋 `policy + WDL` 网络与 MCTS 引擎项目。

## 当前状态

旧 Rust 规则核心与 MCTS 已删除。项目当前唯一目标是将 px0 的规则、history、搜索、UCI 与网络调用主线逐函数翻译为 Rust。

当前工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

P4 尚未关闭。当前已接通真实 `124x10x9` ONNX backend、minibatch、NN cache、prefetch、
collision、shared-tree search worker、`MakeSolid`、watchdog UCI info 和 WDL/contempt display。
此前 Rust 端把一个 `SearchWorker` 的可变状态错误地交给多个 task thread，已删除该不安全路径；
因此 px0 的“每个 task worker 独占其 `SearchWorker`/workspace”的生命周期仍是 P4 未完成项。

主 UCI 使用 px0 名称 `WeightsFile` 配置 ONNX 权重，并会在下一条 `position` 前加载它。正式
`ClassicEngine` 在权重成功加载前没有 `ClassicSearch`，不会以 `UniformBackend` 代替模型搜索。
启动时依次查找 `engin.exe` 同目录的 `x7.onnx`、当前工作目录的 `x7.onnx`、开发目录
`data/x7.onnx`。Windows 上当前优先使用 DirectML；CUDA 仅保留 fallback，不是当前验收后端。

`info` 输出 px0 语义的平均/选择深度、NPS/EPS、WDL、PV、MultiPV、`ScoreType/WDL_mu` 与
可配置的 `ContemptMode`/WDL calibration。`go nodes N` 是 px0 的 completed-iteration 下限，
不承诺 root visits 严格等于 `N`。和 px0 一样，未先发送 `position` 的 `go` 会从 `startpos`
初始化并加载默认权重；若权重不存在，返回明确的 `info string cannot search`。

```text
position startpos
go nodes 1000
```

如需切换模型，仍可在下一条 `position` 前覆盖默认值：

```text
setoption name WeightsFile value C:/models/x7.onnx
position startpos
go nodes 1000
```

## 正式模型契约

- 输入：`124 x 10 x 9`
- policy：`2062` logits
- value：`WDL[win, draw, loss]`
- 引擎 value：`q = win - loss`

px0 翻译顺序与参考位置见 [NextStep.md](/C:/projects/77xiangqi_engine/NextStep.md) 与 [TODO.md](/C:/projects/77xiangqi_engine/TODO.md)。

Python 训练独立位于 `nn/`：固定纯 CNN、`124x10x9 -> 2062 + WDL`，从单一 YAML 配置启动；见
[nn README](C:/projects/77xiangqi_engine/nn/README.md)。
