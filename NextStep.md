# NextStep

## 当前状态

当前唯一工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

P0-P3 的规则、历史、UCI、classic tree/worker 基础已建立。P4 的正式 ONNX 路径使用
`124x10x9 -> 2062 + WDL`，Windows 实测后端为 DirectML。

`nn/` 的训练入口已完成独立收口：参考 pxzero-training 的 YAML 布局，但不移植其旧 TensorFlow
兼容层；当前唯一启动方式是 `train_px0.py --config <yaml>`。这不是 P4 搜索任务的一部分。

`ClassicEngine::new()` 现在是正式 UCI 构造：它在 `WeightsFile` 被下一次 `position` 成功加载前
不创建搜索对象，不再用 `UniformBackend` 伪装可搜索状态。参考 px0
`src/engine.cc:137-197,206-219` 与 `src/neural/shared_params.cc:43-80`。

## P4 未完成：按 px0 重建 task-worker 所有权

此前的 Rust 实现把同一个可变 `SearchWorker` 借给 scoped task thread，导致 node index 越界和
poisoned tree lock。该路径已移除，正式搜索当前禁用 task split，不能再称为 px0
task-worker 完成。

下一步必须逐函数翻译，不补写并发捷径：

1. `src/search/classic/search.h:205-249,357-448`
   - 定义一个 task worker 独占的 worker/workspace/context；不得共享 `&mut SearchWorker`。
2. `src/search/classic/search.cc:1069-1140,1268-1508`
   - 翻译 `RunTasks`、任务领取、gathering/processing 分派、完成和等待。
3. `src/search/classic/search.cc:1828-1897`
   - 翻译 task split 和 `ResetTasks`；`task_count=-1` 只表示 idle，`exiting` 才关闭线程。
4. `src/search/classic/search.cc:1142-1231,1977-2008,2109-2334`
   - 对照 `MaxConcurrentSearchers`、tree phase、out-of-order backup 与 counter 时序。

每次只翻译一个连续参考区间，补对应回归，再提交。禁止重新引入原始指针或 `unsafe impl Send` 来
跨线程共享一个 `SearchWorker`。

## 验收

```powershell
cargo fmt --check
cargo test -p engin --lib
cargo build --release -p engin
```

DirectML 本地冒烟：

```powershell
@('uci', 'isready', 'position startpos', 'go nodes 1000', 'wait', 'quit') |
  .\target\release\engin.exe
```

P4 只有在真实 ONNX/DirectML 下 task worker 生命周期、固定 nodes、`stop`、`position ... moves ...`
和 tree in-flight 清理均对照 px0 通过后才能关闭。
