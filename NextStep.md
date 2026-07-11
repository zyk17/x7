# NextStep

## 当前阶段：P3 单线程 baseline 完成；进入 P4 后端与并发

P0/P1/P2 **规则与 UCI 外围已通过**。P3 **单线程 `go nodes` 已可用**：`ClassicEngine` + `UniformBackend` + `SearchSession`；`search_trace_test` / `uci_search_test` 通过。

P3 的验收范围是单线程、确定性 stub backend 下的 `go nodes` / `go movetime`：PUCT、extend、terminal、backup、tree reuse、绝对 UCI move/ponder 均已覆盖。碰撞/OOO/task workers、完整 stoppers、异步 `StartThreads` 与真实 NN 属 P4。

当前唯一工程参考：

- `C:\Users\Administrator\projects\px0`
- `C:\Users\Administrator\projects\pxzero-training`

| 阶段 | px0 参考 | Rust 目标 | 完成条件 |
|---|---|---|---|
| P0 | `src/chess/types.h`、`bitboard.h`、`board.h/.cc` | types、bitboard、ChessBoard、FEN、走子、合法着 | `board_test.cc` 与 legal move-set 对拍 |
| P1 | `src/chess/position.h/.cc` | Position、PositionHistory、重复、rule60、RuleJudge | `position_test.cc` 逐项移植 |
| P2 | `src/chess/gamestate.*`、`uciloop.*` | UCI 局面/history 语义 | `position ... moves ...` 与 px0 一致 |
| P3 | `src/search` | Node、Tree、Search、worker、选择/扩展/回传 | 单线程固定 FEN/budget trace 对照 |
| P4 | `src/search`、`src/neural` | minibatch、NN cache、prefetch、并发、tree reuse | 同局面统计和 bestmove 对照 |
| P5 | `src/engine.cc`、`src/chess/uciloop.cc` | UCI options、go/stop/info、默认行为 | UCI transcript 对照 |
| P6 | `pxzero-training` | 数据字段、训练/导出契约 | 训练与 ONNX I/O 对照 |

每个 Rust 函数必须标注 px0 文件和连续行区间；找不到参考不实现。

### P3 已落地

- `engin/src/search/classic/node.rs`：`Edge` prior、`Node` 统计、`NodeTree`（`node.cc`）
- `engin/src/search/classic/params.rs`：px0 默认参数子集
- `engin/src/search/classic/backend.rs`：`UniformBackend` stub
- `engin/src/search/classic/search.rs`：`SearchSession`、`ClassicSearch`
- `engin/src/engine.rs`：`ClassicEngine` 接线 UCI `go nodes` / `go movetime`
- `engin/tests/search_trace_test.rs`、`engin/tests/uci_search_test.rs`

### P4 入口

- `src/neural/*` + 真实 policy/value 推理
- `search.cc:1268-2331`：碰撞、OOO、task workers、batch gather/backup
- `classic/stoppers/*`、`search.cc:874-1055`：异步 `StartThreads`、`wait`/`stop`、完整时间管理
- px0 二进制 fixed-nodes trace 对拍（需同一 backend 策略）
