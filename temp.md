# temp

2026-07-12：P3 单线程 baseline 完成。`ClassicEngine` → `SearchSession`（PUCT + extend + backup）→ `bestmove`；`UniformBackend` 作确定性 stub NN。

验收：

- `search_trace_test`：固定 16/32 nodes，root N 达标
- `uci_search_test`：`position startpos` + `go nodes` / `go movetime` 输出绝对坐标的 `bestmove/ponder`
- root in-flight、terminal parent edge、tree reuse 无匹配 edge 分支均有回归覆盖

未闭合：

- 无 px0 二进制 trace 对拍
- 无碰撞 / OOO / task workers / 完整 stoppers / 异步 `StartThreads`
- 无真实 ONNX（P4）

下一入口：P4 `src/neural/*`、碰撞/worker 和完整 stopper。
