# temp

2026-07-15 收尾记录。

- 当前正式模型契约：`124x10x9 -> 2062 + WDL`；正式 UCI 只在 `WeightsFile` 成功加载 ONNX 后搜索，
  不回退到 `UniformBackend`。
- P4 单 worker 主线已可用：ONNX、MemCache、minibatch、prefetch、collision、shared-tree、watchdog、
  legacy clock manager 与 UCI 生命周期已接通。
- px0 默认 legacy 时钟预算已实现：`MoveOverheadMs`、`Slowmover`、
  `go wtime/btime/winc/binc/movestogo`。`depth/mate/ponder/ponderhit` 仍明确拒绝。
- 已知 P4 缺口：GPU task-worker 的 raw-pointer 翻译会在真实 ONNX 搜索中重复 `ExtendNode`，现已删除；当前强制
  `task_workers=0`。后续必须依照 px0 `src/search/classic/search.h:205-244,348-445` 与
  `search.cc:1069-1508` 重新建立无别名的 task 所有权，再恢复该路径。

本次代码验收：

```powershell
cargo fmt --check
cargo test --release -p engin --lib       # 88 passed
cargo test --release -p xiangqi_core      # 22 passed
cargo build --release -p engin
```

真实 ONNX UCI 冒烟已验证 `WeightsFile`、legacy 时钟预算和 `bestmove`：

```powershell
@('uci', 'setoption name MoveOverheadMs value 0', 'setoption name Slowmover value 1.5',
  'setoption name WeightsFile value data\x7.onnx', 'isready', 'position startpos',
  'go wtime 1000 btime 1000 winc 10 binc 10', 'wait', 'quit') |
  .\target\release\engin.exe
```
