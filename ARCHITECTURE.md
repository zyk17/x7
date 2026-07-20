# ARCHITECTURE

## 当前目标

这是一次 Rust 版 px0 的逐函数翻译，不是对旧实现做兼容或修补。

翻译顺序固定为：

1. `xiangqi_core`：px0 `src/chess` 的棋盘、合法着、FEN、Position、PositionHistory、RuleJudge。
2. `engin` 外围：`GameState`、UCI controller/loop、`SearchBase` 与 px0
   `NetworkAsBackendComputation`。P4 的真实 history、124-plane 编码、policy 映射、ONNX batch 和
   `WeightsFile -> OnnxBackend` 子集已接入。
3. `engin/mcts`：px0 `src/search` 的 classic worker 主线；minibatch、collision、prefetch、tree reuse 与
   watchdog 已接线。task 已有 owned leaf path/history/workspace/result 的安全边界，owner 在
   `WaitForTasks` 后独占写 tree 和提交 backend；当前 task queue 仍由 owner 同步 drain，常驻并行
   task-worker 是 P4 的下一项逐段翻译任务。旧 scoped raw-pointer bridge 已删除。
4. prefetch、tree reuse、并发与真实 ONNX 的 px0 `MemCache` wrapper 已接线。缓存通用容器采用
   `quick_cache` 分片 S3-FIFO，value 以 `Arc<EvalResult>` 共享；px0 的 key/collision guard/completed-only
   回填时序不变，但淘汰策略不是严格 FIFO。后续改动只能在明确引用的 px0 语义上继续。
5. `pxzero-training`：数据、训练与 ONNX 导出契约。

## Stream 搜索（进行中）

`crates/engin/src/search/stream` 是独立于 `classic` 的新搜索实现。主线入口是
`search/stream/search.rs` 的 `Search`（常驻 Gather/Eval/Backprop）。它采用 LC3 文档描述的
streaming MCTS **形态**：sharded node repository、edge-local in-flight、owned `NodeEvent`。
第一版只使用 parent-key + move 的**树 key**，不合并成 DAG。协作式单线程 `pipeline` 已删除。

**现状边界**：库内可搜、可对拍；selection/bestmove 走项目批准的 X7↔classic 对照，不是正式 LC3
policy。每次搜索新建空 repository，**无 tree reuse / 无剪枝**；正式 UCI 仍只接 classic。

参考只限 LC3 官方设计文档（无本地 LC3 源码，禁止标称逐行翻译）：

- <https://lczero.org/dev/lc0/search/lc3/overview/>
- <https://lczero.org/dev/lc0/search/lc3/policy/>
- <https://lczero.org/dev/lc0/search/lc3/glossary/>

只有 px0 主线翻译完成并有对拍测试后，才允许比较 lc0 或 KataGo，并将明确记录的差异作为独立优化事项。

## 模块边界

`crates/xiangqi_core`：px0 `src/chess` 的 Rust 翻译，是唯一规则真相。

`crates/engin`：px0 的 UCI/controller、网络外围与 MCTS Rust 翻译；不在搜索内复制规则。P2 UCI、P3 tree 与 P4 的 ONNX、MemCache、collision、prefetch、owner tree phase、watchdog、WDL display 已接入。task 仅持有输入/workspace/result，主 worker 在 `WaitForTasks` 后写 tree 和提交 backend；当前 task queue 的执行仍是 owner 同步 drain，常驻 processing task-worker 尚待逐段翻译。这替代了已删除、会在真实 ONNX 长 `go movetime` 停顿的 scoped raw-pointer bridge。gathering 仍在 owner 的 tree phase 执行。`WeightsFile` 保持 px0 的 UCI 名称，但只接受本项目 ONNX 模型，不实现 px0 的 backend registry、protobuf weight 或 autodiscover。P4 的 `SendUciInfo` 已生成深度、NPS/EPS、WDL、PV、MultiPV、ScoreType 与完整 WDL calibration display 语义。`ClassicEngine` 保持 px0 的会话边界：每个新 `go`、`position`、`ucinewgame` 都先回收旧搜索；`setoption` 只更新下一次 `go` 的参数快照，不中断当前搜索（`src/engine.cc:148-224`、`src/search/classic/wrapper.cc:100-140`）。

未完成对应 px0 stopper 或生命周期的 UCI 命令不得伪装支持：`nodes`、`movetime`、`infinite` 与
px0 factory 默认 legacy 时钟字段可启动搜索。`depth/mate` 仍等待完整 stopper 翻译，
`ponder/ponderhit` 仍等待 `engine.cc` 的 Ponder option/重设局面链路；`simple/smooth/alphazero`
时间管理器不暴露。

`nn/`：pxzero-training 的 `dataset / model / training` 配置布局为参考的独立 Python 训练子项目；训练从单一 YAML 启动，固定 `124x10x9 -> 2062 + WDL` 的纯 CNN 契约。x7 v2 trunk 是 `3x3 stem + PreAct bottleneck + 两次 Global Broadcast` 的结构族；当前基准为 width=256、12 个 bottleneck（`BN/SiLU -> 1x1 256->112 -> 3x3 112->112 -> 3x3 112->112 -> 1x1 112->256 + identity`），并在三个 trunk stage 之间加入独立 Global Broadcast：`BN/SiLU -> 3x3 -> mean/max -> Linear -> x + bias`。YAML 可以试验其他正偶数 width 与不少于 3 个 blocks；恢复/初始化只接受尺寸与 checkpoint 元数据完全一致的 v2 权重。policy 保留 spatial 输出并注入 mean/max global bias；WDL 与 moves-left 共享 global readout：`1x1 conv -> mean/max pool -> FC` 后分为两个线性输出，不再展平 90 个格点。训练主线使用无权重的 policy CE、qMix WDL CE 和 pxzero 语义的 raw-plies Huber moves-left loss；学习率采用 YAML 分段计划，不使用隐藏的全程 cosine 衰减。训练 stream 使用有界 record shuffle，验证集保留文件级 10% 切分，并固定为持久化的 record-level material-stratified sample manifest，避免相邻局面相关性和每次只读取排序文件前缀；不进入规则或搜索热路径。

`model.bottleneck_channels` 是 YAML 的正式模型尺寸参数；省略时保持历史默认值 `width * 7 // 16`，当前
`width=256` 对应 `112`。该值与 width/blocks 一样写入 checkpoint，并在续训、初始化和 ONNX 导出时严格校验。

## 翻译纪律

- 每个 Rust 函数必须标明 px0 文件与连续行区间。
- 没有 px0 对照位置，不实现。
- 旧 Rust 代码不得作为参考或兼容目标。
- 规则测试优先移植 px0 `board_test.cc` 与 `position_test.cc`，搜索测试优先使用同 FEN、同 budget 的 px0 trace。
