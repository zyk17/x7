# TODO

> 当前只做 px0 1:1 Rust 翻译。旧实现、lc0 和 KataGo 都不是当前兼容或优化目标。

## P0：棋规与棋盘

### 规则验收（对拍通过）

- [x] ChessBoard 主路径：FEN、走子、攻击、伪/合法着、长捉、杀子力
- [x] magic bitboard 走 **NO_PEXT** 路径（有意不实现 PEXT）
- [x] 移植 `board_test.cc`：perft d1–d5、FEN 校验、合法着集合

- [x] `ChessBoard::hash()`、`startpos_board()`（px0 `hashcat.h`、`board.cc:58`）
- [x] `bitboard::count_few` 稀疏路径（px0 `bitboard.h:76-88` NO_POPCNT 语义）
- [x] `types.h` 对应标量 API：File 默认无效、File/Rank/Square/Piece/Move 文本与翻转辅助（`types.h:31-222`）
- [x] FEN 一致性校验：保留，因为 px0 `board_test.cc:30-34,265-282` 明确要求非法兵/子力布局抛错

## P1：局面历史与重复规则

- [x] 翻译 `position.h`、`position.cc`（px0 `position.h:38-155`、`position.cc:31-197`）。
- [x] 移植 `PositionHistory`、重复计数、rule60、`RuleJudge`。
- [x] 移植 `position_test.cc:28-260`。

## P2：引擎外围

- [x] 删除旧 `engin` 的 history、ONNX、policy vocabulary、UCI 和旧 core API 调用。
- [x] 建立 `GameState` 骨架：`xiangqi_core/src/gamestate.rs` 对应 `gamestate.h:38-47`、`gamestate.cc:35-55`。
- [x] 建立 `GoParams`、`UciResponder`、`EngineController`、`UciLoop` 骨架：`engin/src/uci_loop.rs` 对应 `uciloop.h:42-116`。
- [x] 翻译 `GameState::CurrentPosition`、`GetPositions`（`gamestate.cc:35-55`），并添加逐步 moves 的位置序列对照测试。
- [x] 翻译 `ParseCommand`、`GetOrEmpty`、`GetNumeric`、`ContainsKey`（`uciloop.cc:81-168`）。
- [x] 翻译 `UciLoop::DispatchCommand`、`ProcessLine`（`uciloop.cc:178-261`）；`position ... moves ...` 保留完整 moves。
- [x] 翻译 String/Stdout responder 格式化（`uciloop.cc:263-337`，依赖 `callbacks.h:42-148`）。
- [x] 建立 stdin/stdout UCI 入口；P3 前 `go` 不返回伪搜索结果。

## P3：搜索

- [x] 建立 `SearchBase`、`ClassicSearch`、`Node`、`Edge`、`SearchParams` 文件与类型骨架。
- [x] `SearchBase` + `ClassicEngine` 取代 P2 RecordingEngine（`engine.rs`、`main.rs`）。
- [x] `Edge` prior、Node N/Q/in-flight、terminal、backup、tree reuse（`node.h/.cc`、`search.cc` 子集）。
- [x] `SearchParams` 默认参数子集（`params.h/.cc` 关键项）。
- [x] 单线程 gather → stub NN → backup，支持 `go nodes` / `go movetime`。
- [x] 固定 nodes、movetime、绝对 UCI move/ponder、tree reuse 回归。

## P4：并发与训练

- [ ] 翻译完整 `PickNodesToExtend`、碰撞、out-of-order、task workers（`search.cc:1268-2331`）。
- [ ] 翻译 `stoppers/*`、`StartThreads` 异步路径、完整 `wtime`/`infinite`/`stop` 语义。
- [ ] 对固定 FEN / fixed nodes 记录 **px0 二进制** node、PV、bestmove trace（当前仅 Rust + stub backend）。
- [ ] 逐函数翻译 px0 minibatch、NN cache、prefetch、tree reuse 与并发路径。
- [ ] 对齐 px0 UCI、bench、info 统计。
- [ ] 按 `pxzero-training` 对齐数据字段、训练和 ONNX 导出。

## 后续才允许做

- [ ] 记录与 lc0/KataGo 的明确差异，再决定是否吸收优化。
