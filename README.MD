# 77xiangqi_engine

中国象棋 `policy + WDL` 网络与 MCTS 引擎项目。

## 当前状态

旧 MCTS 已删除。它的 root 生命周期、选择、扩展、回传、计数与 lc0 classic 已发生结构性漂移，继续修补不具备可信度。

当前保留且可用的地基：

- `xiangqi_core`：局面、合法着、do/undo、规则
- `engin::history`：真实局面历史与 `fen_only` fallback
- `engin::fen_tensor`：`124 x 10 x 9` 编码
- `engin::policy_onnx`：ONNX Runtime 推理与 session pool
- `engin::move_vocab`：`2062` policy 索引映射
- 最小 UCI：`uci` / `isready` / `position` / `setoption PolicyFile`

搜索核心正在按 modern lc0 classic 逐函数移植。重建期间 `go` 明确输出 `bestmove 0000`，不返回旧搜索的伪结果。

## 正式模型契约

- 输入：`124 x 10 x 9`
- policy：`2062` logits
- value：`WDL[win, draw, loss]`
- 引擎 value：`q = win - loss`

## 命令

```powershell
cargo test -p xiangqi_core
cargo test -p engin
cargo run --release -p engin -- --onnx-smoke data\x7.onnx
cargo run --release -p engin
```

搜索重建顺序、lc0 对照位置见 [NextStep.md](/C:/projects/77xiangqi_engine/NextStep.md) 与 [TODO.md](/C:/projects/77xiangqi_engine/TODO.md)。
