# TODO

> 当前只做 px0 1:1 Rust 翻译。旧实现、lc0 和 KataGo 都不是当前兼容或优化目标。

## 已验证基建

- P0/P1：`src/chess/types.h`、`bitboard.h`、`board.h/.cc`、`position.h/.cc` 的 Rust 基础与
  legal move/history/rule60/RuleJudge 回归。
- P2：`src/chess/gamestate.*`、`uciloop.*`；`position ... moves ...` 保留完整历史。
- P3：`src/search/classic/node.*`、`params.*`、基本 selection/extend/backup/tree reuse。
- P4 已接通的部分：`src/neural/backend.h:45-138`、`src/neural/encoder.cc:118-217,229-481`、
  `src/neural/wrapper.cc:49-172`、`src/search/classic/search.cc:1142-1231,1268-1508,1551-1827,
  1977-2334` 的单 worker/minibatch/cache/prefetch/OOO/shared-tree 子集。
- 正式 UCI `WeightsFile` 生命周期：`src/engine.cc:137-197,206-219`，没有权重时明确拒绝搜索，
  不回退到 `UniformBackend`。
- NN 训练入口：参考 `pxzero-training/tf/train.py:110-126`、`tf/configs/example.yaml:4-31`，已收为
  单一 `dataset / model / training` YAML；当前不移植其 TensorFlow 兼容层或旧数据管道。

## P4：task-worker 生命周期，未完成

- [ ] 按 `src/search/classic/search.h:205-249,357-448` 翻译一个 `SearchWorker` + 每 task thread
  一个独占 `TaskWorkspace` 的 px0 所有权关系；禁止共享 Rust `&mut SearchWorker` 或 workspace。
- [ ] 按 `src/search/classic/search.cc:1069-1140,1268-1508` 翻译 task queue 的领取、执行、
  gathering/processing 回写和 `WaitForTasks`。
- [ ] 按 `src/search/classic/search.cc:1828-1897` 翻译 split、idle、退出和 join；
  `task_count=-1` 与 `exiting` 必须分开。
- [ ] 对照 `src/search/classic/search.cc:1142-1231,1977-2008,2109-2334` 验证多 SearchWorker +
  task worker 的 tree phase、in-flight、OOO 与 counters。
- [ ] 在 DirectML/ONNX 下补固定 nodes、`go infinite -> stop -> wait`、`position ... moves ...`、
  backend reload 的回归；结束时 root `NInFlight=0`。

## 约束

- 每个 Rust 函数的注释或本文件必须标注 px0 文件和连续行区间。
- 找不到 px0 对应参考时，记录缺口，不实现。
- `UniformBackend` 仅限单元测试和对拍，正式 UCI 永不使用。
