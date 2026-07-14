# 77xiangqi_engine

中国象棋 `policy + WDL` 网络与 MCTS 引擎项目。

## 当前状态

旧 Rust 规则核心与 MCTS 已删除。项目当前唯一目标是将 px0 的规则、history、搜索、UCI 与网络调用主线逐函数翻译为 Rust。

当前工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

P4 classic 搜索基建已收口：真实 `124x10x9` ONNX backend、minibatch、NN cache、prefetch、
collision、task worker、多 `SearchWorker` tree phase、`MakeSolid`、watchdog 实时 UCI info 和
WDL/contempt display 均已按 px0 对照实现。主 UCI 使用 px0 名称 `WeightsFile` 配置 ONNX 权重，
并会在下一条 `position` 前加载它；`UniformBackend` 只用于测试，不能把其搜索结果当作模型结果。

`info` 输出 px0 语义的平均/选择深度、NPS/EPS、WDL、PV、MultiPV、`ScoreType/WDL_mu` 与
可配置的 `ContemptMode`/WDL calibration。`go nodes N` 是 px0 的 completed-iteration 下限，
不承诺 root visits 严格等于 `N`。

```text
setoption name WeightsFile value data/x7.onnx
position startpos
go nodes 1000
```

## 正式模型契约

- 输入：`124 x 10 x 9`
- policy：`2062` logits
- value：`WDL[win, draw, loss]`
- 引擎 value：`q = win - loss`

px0 翻译顺序与参考位置见 [NextStep.md](/C:/projects/77xiangqi_engine/NextStep.md) 与 [TODO.md](/C:/projects/77xiangqi_engine/TODO.md)。
