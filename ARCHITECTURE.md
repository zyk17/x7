# ARCHITECTURE

## 当前目标

这是一次 Rust 版 px0 的逐函数翻译，不是对旧实现做兼容或修补。

翻译顺序固定为：

1. `xiangqi_core`：px0 `src/chess` 的棋盘、合法着、FEN、Position、PositionHistory、RuleJudge。
2. `engin` 外围：`GameState`、UCI controller/loop、`SearchBase` 与 px0
   `NetworkAsBackendComputation`。P4 的真实 history、124-plane 编码、policy 映射、ONNX batch 和
   `WeightsFile -> OnnxBackend` 子集已接入。
3. `engin/mcts`：px0 `src/search` 的 worker 主线；完整 collision/task-worker/prefetch 仍待翻译。
4. px0 已有的 NN cache、prefetch、tree reuse 与并发。
5. `pxzero-training`：数据、训练与 ONNX 导出契约。

只有 px0 主线翻译完成并有对拍测试后，才允许比较 lc0 或 KataGo，并将明确记录的差异作为独立优化事项。

## 模块边界

`crates/xiangqi_core`：px0 `src/chess` 的 Rust 翻译，是唯一规则真相。

`crates/engin`：px0 的 UCI/controller、网络外围与 MCTS Rust 翻译；不在搜索内复制规则。当前 P2 UCI、P3 单 worker 树与 P4 的 ONNX/worker 子集已存在；完整 collision/task-worker/prefetch 仍待逐函数翻译。`WeightsFile` 保持 px0 的 UCI 名称，但只接受本项目 ONNX 模型，不实现 px0 的 backend registry 或 protobuf weight discovery。P4 当前会在搜索结束时按 px0 `SendUciInfo` 生成深度、NPS/EPS、WDL、PV 和 MultiPV；实时 info 与 `ScoreType/WDL_mu` 要等 px0 OptionsDict 与 worker 并发边界翻译完成后再接入。

`nn/`：pxzero-training 数据/训练/导出契约的 Rust/Python 侧接入；不进入规则或搜索热路径。

## 翻译纪律

- 每个 Rust 函数必须标明 px0 文件与连续行区间。
- 没有 px0 对照位置，不实现。
- 旧 Rust 代码不得作为参考或兼容目标。
- 规则测试优先移植 px0 `board_test.cc` 与 `position_test.cc`，搜索测试优先使用同 FEN、同 budget 的 px0 trace。
