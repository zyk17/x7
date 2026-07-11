# TODO

> 只保留未完成的搜索重建事项。旧 MCTS 任务已删除，不作为兼容目标。

## S0：节点与搜索容器

- [ ] 按 `C:\Users\Administrator\projects\lc0\src\search\classic\node.h`、`node.cc` 建立最小 Node/Edge 字段。
- [ ] 按 `search.h:50-203,209-419` 建立 `Search` / `SearchWorker` 边界。
- [ ] 为 visits、in-flight、parent、child、WDL 累加建立不变量测试。

## S1：单线程生命周期

- [ ] 按 `search.cc:921-1047` 译 `RunBlocking` 与 iteration 统计。
- [ ] 按 `search.h:254-300` 建立 `InitializeIteration -> Gather -> NN -> Fetch -> Backup` 数据流。
- [ ] root 不允许单独预展开；必须与普通节点走同一 ExtendNode 路径。

## S2：选择与扩展

- [ ] 按 `search.cc:1507-1920` 译 selection、collision、in-flight、multivisit。
- [ ] 按 `search.cc:1921-2149` 译 `ExtendNode`，再最小接入象棋合法着与终局规则。
- [ ] 在 `rule60`、重复与 RuleJudge 接入前，记录对应 px0 象棋参考的确切文件与行号。

## S3：推理与回传

- [ ] 按 `search.cc:2151-2216` 译 batch fetch；ONNX 不得持树锁。
- [ ] 按 `search.cc:2217-2373` 译 backup、PV、边界与计数。
- [ ] 对每个 budget 建立 `total_playouts == successful_backups`、搜索结束 `in_flight == 0` 测试。

## S4：外部行为

- [ ] 按 `search.cc:261-396,617-646,896-920` 恢复 UCI info、stop 与 worker 生命周期。
- [ ] 仅在单线程 trace 对齐后，按 `search.cc:1060-1490` 接 batch/task workers/shared tree。
- [ ] MultiPV、tree reuse、prefetch、NN cache 最后逐项恢复；每项都附 lc0 对照测试。
