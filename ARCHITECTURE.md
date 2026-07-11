# ARCHITECTURE

## 当前目标

这是一次 Rust 版 px0 的逐函数翻译，不是对旧实现做兼容或修补。

翻译顺序固定为：

1. `xiangqi_core`：px0 `src/chess` 的棋盘、合法着、FEN、Position、PositionHistory、RuleJudge。
2. `engin` 外围：`GameState`、UCI controller/loop；P2 已完成。后续接入 SearchBase，再接 history、124-plane 编码、policy 映射与 ONNX。
3. `engin/mcts`：px0 `src/search` 的单线程主线。
4. px0 已有的 batch、NN cache、prefetch、tree reuse 与并发。
5. `pxzero-training`：数据、训练与 ONNX 导出契约。

只有 px0 主线翻译完成并有对拍测试后，才允许比较 lc0 或 KataGo，并将明确记录的差异作为独立优化事项。

## 模块边界

`crates/xiangqi_core`：px0 `src/chess` 的 Rust 翻译，是唯一规则真相。

`crates/engin`：px0 的 UCI/controller、网络外围与 MCTS Rust 翻译；不在搜索内复制规则。当前 P2 UCI 已完成，P3 仅有搜索骨架。

`nn/`：pxzero-training 数据/训练/导出契约的 Rust/Python 侧接入；不进入规则或搜索热路径。

## 翻译纪律

- 每个 Rust 函数必须标明 px0 文件与连续行区间。
- 没有 px0 对照位置，不实现。
- 旧 Rust 代码不得作为参考或兼容目标。
- 规则测试优先移植 px0 `board_test.cc` 与 `position_test.cc`，搜索测试优先使用同 FEN、同 budget 的 px0 trace。
