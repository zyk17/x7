# 77xiangqi_engine

中国象棋 `policy + WDL` 网络与 MCTS 引擎项目。

## 当前状态

旧 Rust 规则核心与 MCTS 已删除。项目当前唯一目标是将 px0 的规则、history、搜索、UCI 与网络调用主线逐函数翻译为 Rust。

当前工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

P4 的单 worker 主线已接通真实 `124x10x9` ONNX backend、minibatch、prefetch、collision、
shared-tree、`MakeSolid`、watchdog UCI info 和 WDL/contempt display。正式 ONNX 已由
px0 `MemCache` 风格的 `CachingBackend` 包裹：按局面 hash 查找、以合法着数量防碰撞、batch 完成后写入、
`ucinewgame` 清空；默认 `NNCacheSize=2000000`。缓存容器使用 `quick_cache` 分片 S3-FIFO，缓存命中以
`Arc<EvalResult>` 共享 policy 向量；这替代了 px0 的严格 FIFO 容器，但不改变 NN cache 的 key、collision
guard 或 completed-only 回填语义。
px0 的常驻 processing task-worker 已按 `src/search/classic/search.h:205-244,348-445` 与
`search.cc:1069-1140,1322-1362,1423-1462` 接入：task 只拥有 leaf path、root history、workspace 与
extension result；主 worker 在 `WaitForTasks` 后独占写 tree、提交 ONNX 和 backup。旧 scoped raw-pointer
bridge 已删除，后台 task 不会借用 `NodeTree`、minibatch 或 backend computation。gathering task 仍由 owner
tree phase 同步执行，避免把 px0 的 `nodes_mutex_` 别名模型错误搬到 Rust。

主 UCI 使用 px0 名称 `WeightsFile` 配置 ONNX 权重，并会在下一条 `position` 前加载它。正式
`ClassicEngine` 在权重成功加载前没有 `ClassicSearch`，不会以 `UniformBackend` 代替模型搜索。
启动时依次查找 `engin.exe` 同目录的 `x7.onnx`、当前工作目录的 `x7.onnx`、开发目录
`data/x7.onnx`。Windows 发行构建只启用 DirectML；DirectML 初始化失败时明确回退 CPU，不依赖本机 CUDA 或 cuDNN。

`info` 输出 px0 语义的平均/选择深度、NPS/EPS、WDL、PV、MultiPV、`ScoreType/WDL_mu` 与
可配置的 `ContemptMode`/WDL calibration。`go nodes N` 是 px0 的 completed-iteration 下限，
不承诺 root visits 严格等于 `N`。和 px0 一样，未先发送 `position` 的 `go` 会从 `startpos`
初始化并加载默认权重；若权重不存在，返回明确的 `info string cannot search`。

同一 UCI 会话的每次新 `go`、`position` 或 `ucinewgame` 都会先静默 `Abort + Wait` 上一次
搜索，再启动或重置下一状态；因此 GUI 可以发送 `position -> go -> position -> go`，最后只会收到
新搜索的 `bestmove`。这一生命周期对照 px0 `src/engine.cc:148-151,187-224` 与
`src/search/classic/wrapper.cc:100-140`。

当前开放 `go nodes`、`go movetime`、`go infinite` 与 px0 默认 legacy 时钟预算
`wtime/btime/winc/binc/movestogo`；对应 UCI 选项是 `MoveOverheadMs` 和 `Slowmover`。`go depth`、
`go mate`、`go ponder`、`ponderhit` 仍会明确拒绝，不能静默变成普通搜索。只翻译并暴露 px0 factory 的
默认 `legacy` time manager，不暴露尚未翻译的 `simple/smooth/alphazero` 变体。

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

## Windows DirectML 打包

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1
```

脚本清理并重新编译 `engin`，然后生成 `bundle\`。当前 `ort` 静态链接 ONNX Runtime，因此 bundle 包含
`engin.exe`、`x7.onnx` 和 `DirectML.dll`，不应混入其他版本的 `onnxruntime.dll`。

Python 训练独立位于 `nn/`：固定纯 CNN、`124x10x9 -> 2062 + WDL`，从单一 YAML 配置启动；见
[nn README](C:/projects/77xiangqi_engine/nn/README.md)。
