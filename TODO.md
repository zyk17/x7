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

当前：GPU task split 已接通；CPU 保持 px0 `task_workers_=0`。下一步验证多 SearchWorker、OOO、
stop 与真实 ONNX/DirectML 时序。

- [x] 按 `src/search/classic/search.cc:981-1034` 翻译 watchdog 的 counters-mutex/condition-variable
  等待与 `FireStopInternal` 唤醒；不再固定 1ms polling。
- [x] 按 `src/search/classic/search.cc:249-264,393-398,908-918,1213-1231` 翻译
  `nps_start_time_` 的 watchdog 初始化、UCI nps/eps 与 NPS limit 时钟归属。
- [x] 按 `src/search/classic/search.h:368-369, search.cc:596-610,908-922,981-1017,1268-1284`
  将 `latest_time_manager_hints_` 收为 SearchWorker-local、watchdog-local 两份；不得跨 worker
  共享 remaining-playouts hint。
- [x] 按 `src/search/classic/search.cc:596-610` 加入 root-first-visit stopper gate；未扩展根节点
  不能被 budget stopper 提前结束。

- [x] 按 `src/search/classic/search.h:435-445`、`search.cc:1069-1119,1464-1483` 翻译
  `task_taking_started`、claim、idle、wake、close 与重用；已补多线程唯一领取回归。
- [x] 按 `src/search/classic/search.cc:1142-1211,1494-1508` 将 Rust `NodeTree` 收为显式
  tree-phase 借用，删除 `active: *mut NodeTree`；direct 与 shared tree 均通过同一安全边界进入
  selection/process/fetch/backup。
- [x] 按 `src/search/classic/search.h:205-249,357-448, search.cc:1069-1140,1268-1508` 翻译一个
  `SearchWorker` + 每 task thread 一个独占 `TaskWorkspace` 的 lifecycle、gathering/processing
  回写与 `WaitForTasks`；GPU 回归确认 helper 实际领取任务。
- [x] 按 `src/search/classic/search.cc:1494-1501,1828-1897`、`src/search/classic/node.h:423-525,547-610`
  与 `src/utils/mutex.h:93-125` 建立受限 `TaskTreeBridge`：只允许 scoped task thread 在 active
  tree phase 内访问普通 Node；`task_count=-1` 与 `exiting` 继续分开。
- [ ] 对照 `src/search/classic/search.cc:1142-1231,1977-2008,2109-2334` 验证多 SearchWorker +
  task worker 的 tree phase、in-flight、OOO 与 counters。
- [ ] 在 DirectML/ONNX 下补固定 nodes、`go infinite -> stop -> wait`、`position ... moves ...`、
  backend reload 的回归；结束时 root `NInFlight=0`。

## 约束

- 每个 Rust 函数的注释或本文件必须标注 px0 文件和连续行区间。
- 找不到 px0 对应参考时，记录缺口，不实现。
- `UniformBackend` 仅限单元测试和对拍，正式 UCI 永不使用。
