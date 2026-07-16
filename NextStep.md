# NextStep

## 当前基线

唯一工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

P0-P3 已完成。P4 的安全单 worker 主线已完成：真实 `124x10x9 -> 2062 + WDL` ONNX、minibatch、
prefetch、collision、OOO、MemCache、tree reuse、watchdog、legacy time manager 与 UCI 生命周期均可用。
Windows 发行构建只启用 DirectML，失败时明确回退 CPU。

`ContemptMode=play` 的无限搜索按 px0 `Search::Search`
(`src/search/classic/search.cc:156-170`) 在启动 worker 前降为 `none`；仅当 WDL rescale diff 非零时输出
对应 `info string` warning。旧 bridge 遗留的 `outputs` / `search_active` 状态已删除，不再作为搜索真相。
watchdog 也会在 stopper 第一次令搜索停止后继续执行 px0 `MaybeOutputInfo`
(`src/search/classic/search.cc:351-389,981-1017`)；不允许响应 `bestmove` 的无限搜索会得到无进展 warning。

Rust 的 `ClassicSearch` 跨多条 UCI `go` 复用，而 px0 每条 `go` 都销毁其 `Search`。因此 Rust 在所有
search worker join 后显式执行 px0 `Search::CancelSharedCollisions`，确保上一搜索未 backup 的 collision
virtual visit 不会带入下一次 `go`。参考 `src/search/classic/search.cc:1027-1064`。

## P4 当前边界

`TaskWorkers` 按 px0 `src/search/classic/search.h:205-224` 解析，并在 `SearchWorker` 构造/析构时创建、
停止和回收常驻 processing task worker。`active_task_workers` 不再是 `0` 门控。

px0 的 `RunTasks` 让后台 task 在主 worker 持有 `nodes_mutex_` 的同一 tree phase 中直接访问普通
`Node`/`NodeTree`，见：

- `src/search/classic/search.h:205-244,348-445`
- `src/search/classic/search.cc:1069-1140,1322-1362,1423-1508,1551-1897`
- `src/search/classic/node.cc:245-373`

Rust 当前的 `NodeTree` 需要独占 `&mut` 借用，且 `Node` 的 WDL、terminal、bounds、children 和 edge
不是可并发字段。processing task 因此只消费 owned leaf path/root history，以私有 `TaskWorkspace` 计算
`ExtendNode` 的 rule/move result；owner 在 `WaitForTasks` 后独占写 node、提交 backend input、OOO 和 backup。
此前 raw-pointer scoped bridge 在真实 ONNX 的长时间搜索中重复扩展节点并停顿，已删除。

禁止：

- 恢复 raw pointer、`unsafe impl Send` 或共享 `&mut SearchWorker`
- 用整树锁把后台 task 串行化后宣称已经并行
- 仅凭 `n/n_in_flight` 原子化或 child slot 唯一创建解除门控

## P4 剩余翻译点：gathering task

1. processing task 已完成安全所有权边界：其连续参考为 px0 `search.h:205-244,348-445` 与
   `search.cc:1069-1140,1322-1362,1423-1462`。后台 task 只能拥有 `NodeToProcess + moves/history`、workspace
   和 extension result；owner 在 `WaitForTasks` 后写 tree/backend。
2. gathering 仍由 owner tree phase 同步运行。若继续翻译 px0 `PickNodesToExtendTask`
   (`search.cc:1551-1897`)，必须先给出不共享 `NodeTree` 的 owned selection delta 方案；不能用 raw pointer、
   `unsafe impl Send`、整树锁或共享 `&mut SearchWorker`。
3. `WaitForTasks` 返回后才允许 collision、NN、fetch、backup 进入下一阶段。
4. 以真实 x7 ONNX/DirectML 验证固定 nodes、长 `movetime`、`stop -> wait`、`position ... moves ...`，并验证所有节点的 `NInFlight == 0`。

## 验收命令

```powershell
cargo fmt --check
cargo test -p engin --lib
cargo clippy -p engin --all-targets -- -D warnings
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1
```
