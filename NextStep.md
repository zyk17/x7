# NextStep

## 当前状态

当前 `engin` 主线已经收口到：

- 主参考：`C:\Users\Administrator\projects\lc0`
- 保留外壳：`main.rs / lib.rs / policy_onnx.rs / history.rs / fen_tensor.rs`
- 主实现落点：`uci.rs / mcts/config.rs / mcts/engine.rs / mcts/search.rs / mcts/worker.rs / mcts/tree.rs`
- 已删除或内联：`task_pick.rs / coordinator.rs / prefetch.rs / pick.rs`

当前已经基本对齐的部分：

1. `position / go / stop / ucinewgame`
2. `nodes / nps / depth / seldepth / pv`
3. `Threads=0 / MinibatchSize=0`
4. `shared-tree + batched eval + subtree reuse`
5. `policy + WDL` ONNX 主链路

## 还差什么要抄

接下来不要再发散做新路线，直接补 lc0 还没抄完的部分。

### 1. 完整 time manager

当前 `go wtime / btime / winc / binc / movestogo` 还是 warning + reject/忽略，不是 lc0 完整主线。

参考代码：

- `C:\Users\Administrator\projects\lc0\src\chess\uciloop.cc:221-243`
- `C:\Users\Administrator\projects\lc0\src\search\classic\stoppers\common.cc:118-165`

本仓库落点：

- `C:\projects\77xiangqi_engine\crates\engin\src\uci.rs`
- `C:\projects\77xiangqi_engine\crates\engin\src\mcts\config.rs`

### 2. 完整 UCI 生命周期

当前 `ponder / searchmoves / mate` 还只是最小兼容，不是 lc0 语义闭环。

参考代码：

- `C:\Users\Administrator\projects\lc0\src\chess\uciloop.cc:167-245`
- `C:\Users\Administrator\projects\lc0\src\engine.cc:200-257`

本仓库落点：

- `C:\projects\77xiangqi_engine\crates\engin\src\uci.rs`

### 3. 更完整的并行细节

当前已经有最小 `shared-tree` 多线程，但还没完全抄到 lc0 的 task workers / node-lock 细节。

参考代码：

- `C:\Users\Administrator\projects\lc0\src\search\classic\search.h:216-303`
- `C:\Users\Administrator\projects\lc0\src\search\classic\search.cc:1209-1439`
- `C:\Users\Administrator\projects\lc0\src\search\classic\search.cc:2018-2377`

本仓库落点：

- `C:\projects\77xiangqi_engine\crates\engin\src\mcts\search.rs`
- `C:\projects\77xiangqi_engine\crates\engin\src\mcts\worker.rs`
- `C:\projects\77xiangqi_engine\crates\engin\src\policy_onnx.rs`

### 4. 补回回归护栏

当前单测是通的，但 engine 侧长期回归护栏还不够厚。

参考代码：

- `C:\Users\Administrator\projects\lc0\src\chess\uciloop.cc`
- `C:\Users\Administrator\projects\lc0\src\engine.cc`
- `C:\Users\Administrator\projects\lc0\src\search\classic\search.cc`

本仓库落点：

- `C:\projects\77xiangqi_engine\crates\engin\tests\p3_integration.rs`
- `C:\projects\77xiangqi_engine\crates\engin\src\benchmark.rs`
- `C:\projects\77xiangqi_engine\crates\engin\src\uci.rs`

## 实施顺序

后续严格按这个顺序做：

1. `time manager`
2. `ponder / searchmoves / mate`
3. `task workers / node-lock` 并行细节
4. `integration / bench` 回归测试

## 不做的事

当前这轮先不做：

- 参考 `px0` 搜索实现继续改结构
- 吸收 `KataGo` 的 graph / DAG / cache 设计
- 新搜索路线
- `MultiPV`
- 复杂自创参数体系

## 之后 review 的标准

后续我会按这几个标准 review：

1. 是否继续严格对齐 `lc0`
2. 是否又引入了不必要的新抽象
3. 是否破坏 `history / tree reuse / UCI` 的语义闭合
4. 统计口径是否一致
5. 文档和真实接口是否一致
