# ARCHITECTURE

## 边界

`crates/xiangqi_core` 是唯一规则真相：棋盘、合法着、走子、重复与中国象棋规则。

`crates/engin` 当前只保留搜索重建所需的外围地基：

- `history.rs`：完整游戏历史；搜索将从 root history 派生临时路径
- `fen_tensor.rs`：px0 classical 的 `124 x 10 x 9` 网络输入
- `policy_onnx.rs`：`2062 + WDL` ONNX 推理
- `move_vocab.rs`：policy 索引与中国象棋着法映射
- `uci.rs`：最小 UCI 状态机；不包含旧搜索

`nn/` 负责训练与导出，不进入规则或搜索热路径。

## 搜索重建原则

搜索实现只以 modern lc0 classic 为骨架：

1. 一个 Rust 搜索函数必须映射到 lc0 的函数或连续小段。
2. root 与普通节点走同一 `pick -> extend -> NN -> backup` 生命周期。
3. 中国象棋只替换规则、走法、`rule60`、重复判定及网络编码；不改变搜索语义。
4. 单线程完整正确后，才接 batch、任务 worker、并发、tree reuse 与 MultiPV。

当前无正式 MCTS、benchmark、旧搜索统计或旧 UCI 搜索选项。它们会随移植阶段重新引入，而非兼容旧实现。
