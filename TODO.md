# TODO

> 当前只做 px0 1:1 Rust 翻译。旧实现、lc0 和 KataGo 都不是当前兼容或优化目标。

## 已验证基建

- P0/P1：`src/chess/types.h`、`bitboard.h`、`board.h/.cc`、`position.h/.cc` 的 Rust 基础与
  legal move/history/rule60/RuleJudge 回归。
- P2：`src/chess/gamestate.*`、`uciloop.*`；`position ... moves ...` 保留完整历史。
- P3：`src/search/classic/node.*`、`params.*`、基本 selection/extend/backup/tree reuse。
- P4 已接通的部分：`src/neural/backend.h:45-138`、`src/neural/encoder.cc:118-217,229-481`、
  `src/neural/wrapper.cc:49-172`、`src/neural/memcache.h:34-45,memcache.cc:38-190`、
  `src/search/classic/search.cc:1142-1231,1268-1508,1551-1827,1977-2334` 的 ONNX/MemCache、
  单 worker/minibatch/prefetch/OOO/shared-tree 子集。
- 正式 UCI `WeightsFile` 生命周期：`src/engine.cc:137-197,206-219`，没有权重时明确拒绝搜索，
  不回退到 `UniformBackend`。
- NN 训练入口：参考 `pxzero-training/tf/train.py:110-126`、`tf/configs/example.yaml:4-31`，已收为
  单一 `dataset / model / training` YAML；当前不移植其 TensorFlow 兼容层或旧数据管道。
- P1-P3 进入 P4 前复核：`cargo test --release -p xiangqi_core`（22 项 px0 规则/history 对拍）、
  `cargo test -p engin --lib`（79 项 P2/P3/controller/tree/UCT 回归）与 UCI/P4 生命周期集成测试已通过。

## P4：单 worker 搜索流水线可用；task-worker 待重构

单 worker/minibatch/OOO/cache/stop 与真实 ONNX/DirectML 时序已有回归和 release 冒烟。GPU
task split 不可用：当前 Rust raw-pointer 版本会让两个 task 重复扩展同一未扩展节点，已统一退回
`task_workers_=0`，不能作为 px0 对齐完成项。

- [x] 翻译 px0 `src/neural/memcache.cc:38-190`、`memcache.h:34-45` 为正式 ONNX 的
  `CachingBackend` wrapper：当前局面 hash 为 key、合法着数量防碰撞、cache miss 仅在
  `ComputeBlocking` 后写入、`ucinewgame` 清 cache；`NNCacheSize` 默认/范围为
  `src/neural/shared_params.cc:63-82` 的 `2000000` / `0..999999999`。

- [x] 按 `src/search/classic/search.cc:981-1034` 翻译 watchdog 的 counters-mutex/condition-variable
  等待与 `FireStopInternal` 唤醒；不再固定 1ms polling。
- [x] 按 `src/search/classic/search.cc:249-264,393-398,908-918,1213-1231` 翻译
  `nps_start_time_` 的 watchdog 初始化、UCI nps/eps 与 NPS limit 时钟归属。
- [x] 按 `src/search/classic/search.h:368-369, search.cc:596-610,908-922,981-1017,1268-1284`
  将 `latest_time_manager_hints_` 收为 SearchWorker-local、watchdog-local 两份；不得跨 worker
  共享 remaining-playouts hint。
- [x] 按 `src/search/classic/search.cc:596-610` 加入 root-first-visit stopper gate；未扩展根节点
  不能被 budget stopper 提前结束。

- [x] 在 DirectML/ONNX 下完成固定 nodes、`go infinite -> stop -> wait`、`position ... moves ...`、
  backend reload 的 release UCI 冒烟。另验证 `go infinite -> go nodes` 与
  `go infinite -> position ... -> go nodes`：旧搜索静默回收，只有最后一次 `go` 输出 `bestmove`。
  对照 `src/engine.cc:148-224`、`src/search/classic/wrapper.cc:100-140`。

## 后续：P4 task-worker 的安全所有权翻译

- [ ] 重新设计 Rust task 的所有权边界，对照 px0 `src/search/classic/search.h:205-244,348-445`、
  `search.cc:1069-1508`：task 只能拥有独立 workspace 与明确不重叠的 node/minibatch range，不能共享
  `&mut SearchWorker` 或通过 raw pointer 直接修改整棵树。
- [ ] 在固定 visits、`go movetime`、真实 ONNX/DirectML 下补重复 ExtendNode、`NInFlight==0`、stop/wait
  回归；只有该回归稳定后才能恢复 px0 GPU `task_workers_` 默认解析。

## 约束

- 每个 Rust 函数的注释或本文件必须标注 px0 文件和连续行区间。
- 找不到 px0 对应参考时，记录缺口，不实现。
- `UniformBackend` 仅限单元测试和对拍，正式 UCI 永不使用。
