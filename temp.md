# temp

2026-07-13：P4 第一检查点：真实 ONNX backend 已接入（尚未作为 UCI 默认 backend）。

- `PositionHistory -> 124x10x9`：px0 `encoder.cc:118-217`，含真实历史与 `FEN_ONLY`
- 2062 move table：由 px0 `encoder.cc:229-481` 机械提取，SHA256
  `884BEA3BBD05A119E7E8A2965993FAAAA564FD46440BD249E9D728336CC89924`
- `OnnxBackend`：px0 `wrapper.cc:49-172`，本地 `data/x7.onnx` 冒烟通过
- P4 尚未完成：树锁阶段拆分、完整 selection/collision/task worker/prefetch、UCI 权重配置均未完成

2026-07-12：P4 异步搜索 + UCI 接线完成。

验收（`cargo test -p engin -p xiangqi_core --release` 全绿）：

- `ClassicSearch` 多线程 `StartThreads` + `SearchWorker` 七阶段
- `go nodes` / `movetime` / `wtime` / `infinite`+`stop` → `bestmove`
- `nodes_budget` 精确控制 root visits；`TimeLimitStopper` 至少搜 1 node
- `p4_trace_test`：startpos 16 nodes 确定性
- `UniformBackend` NN cache 子集

未闭合：

- 完整 `PickNodesToExtendTask`、task workers、`RunTasks`
- 完整 `PrefetchIntoCache` 递归
- px0 二进制 trace golden
- 真实 ONNX / `pxzero-training` 导出接入

下一入口：`PickNodesToExtendTask` 或 `src/neural/*` ONNX。
