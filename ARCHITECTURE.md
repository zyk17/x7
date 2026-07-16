# ARCHITECTURE

## 当前目标

这是一次 Rust 版 px0 的逐函数翻译，不是对旧实现做兼容或修补。

翻译顺序固定为：

1. `xiangqi_core`：px0 `src/chess` 的棋盘、合法着、FEN、Position、PositionHistory、RuleJudge。
2. `engin` 外围：`GameState`、UCI controller/loop、`SearchBase` 与 px0
   `NetworkAsBackendComputation`。P4 的真实 history、124-plane 编码、policy 映射、ONNX batch 和
   `WeightsFile -> OnnxBackend` 子集已接入。
3. `engin/mcts`：px0 `src/search` 的 classic worker 主线；单 worker 的 minibatch、collision、
   prefetch、tree reuse 与 watchdog 已接线。GPU task-worker 的 Rust 所有权翻译尚未完成，当前强制
   `task_workers=0`，不得把已删除的 raw-pointer bridge 视为已接线实现。
4. prefetch、tree reuse、并发与真实 ONNX 的 px0 `MemCache` wrapper 已接线。缓存通用容器采用
   `quick_cache` 分片 S3-FIFO，value 以 `Arc<EvalResult>` 共享；px0 的 key/collision guard/completed-only
   回填时序不变，但淘汰策略不是严格 FIFO。后续改动只能在明确引用的 px0 语义上继续。
5. `pxzero-training`：数据、训练与 ONNX 导出契约。

只有 px0 主线翻译完成并有对拍测试后，才允许比较 lc0 或 KataGo，并将明确记录的差异作为独立优化事项。

## 模块边界

`crates/xiangqi_core`：px0 `src/chess` 的 Rust 翻译，是唯一规则真相。

`crates/engin`：px0 的 UCI/controller、网络外围与 MCTS Rust 翻译；不在搜索内复制规则。P2 UCI、P3 tree 与 P4 的 ONNX、MemCache、collision、prefetch、单 worker tree phase、watchdog 和 WDL display 已接入。GPU task-worker 仍是 P4 未完成项：此前 raw-pointer 移植在真实 ONNX 下会重复扩展节点，已从代码删除，现保持 `task_workers=0`。`WeightsFile` 保持 px0 的 UCI 名称，但只接受本项目 ONNX 模型，不实现 px0 的 backend registry、protobuf weight 或 autodiscover。P4 的 `SendUciInfo` 已生成深度、NPS/EPS、WDL、PV、MultiPV、ScoreType 与完整 WDL calibration display 语义。`ClassicEngine` 保持 px0 的会话边界：每个新 `go`、`position`、`ucinewgame` 都先回收旧搜索；`setoption` 只更新下一次 `go` 的参数快照，不中断当前搜索（`src/engine.cc:148-224`、`src/search/classic/wrapper.cc:100-140`）。

未完成对应 px0 stopper 或生命周期的 UCI 命令不得伪装支持：`nodes`、`movetime`、`infinite` 与
px0 factory 默认 legacy 时钟字段可启动搜索。`depth/mate` 仍等待完整 stopper 翻译，
`ponder/ponderhit` 仍等待 `engine.cc` 的 Ponder option/重设局面链路；`simple/smooth/alphazero`
时间管理器不暴露。

`nn/`：pxzero-training 的 `dataset / model / training` 配置布局为参考的独立 Python 训练子项目；训练从单一 YAML 启动，固定 `124x10x9 -> 2062 + WDL` 的纯 CNN 契约，不进入规则或搜索热路径。

## 翻译纪律

- 每个 Rust 函数必须标明 px0 文件与连续行区间。
- 没有 px0 对照位置，不实现。
- 旧 Rust 代码不得作为参考或兼容目标。
- 规则测试优先移植 px0 `board_test.cc` 与 `position_test.cc`，搜索测试优先使用同 FEN、同 budget 的 px0 trace。
