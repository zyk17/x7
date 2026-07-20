# temp

## Stream 真实状态（2026-07-20，勿乐观）

- **能做什么**：独立 LC3-*形态* 的 Gather/Eval/Backprop（主线 `stream::Search` / `search.rs`）、
  分片 repository、edge in-flight、owned event/generation、ONNX minibatch；X7 批准的 classic 对照
  selection + 库内 bestmove/PV；`stream_compare` / `stream_behavior_compare`。
- **不能做什么**：不接 UCI；**无 tree reuse**（每手空表，上一步对下一步零帮助）；无剪枝/GC；
  无 Watchdog 周期 info；无 MultiPV；NN `m` 未做节点 backup 聚合。
  **明确不做** contempt / 非零 `draw_score`。
- **后续**：精简 stream 参数面（勿为 `compute_cpuct` 挂整份 classic `SearchParams`）。
- **模块**：主线在 `search/stream/search.rs`（`Search`）。
- **文档注意**：`NextStep.md`/`TODO.md` 里旧的「S2c 完全 blocked / 只有 root_stats」已过时；
  以本节与 `temp_stream_x7_policy.md` 为准。正式对局引擎仍是 classic。

---

2026-07-15 收尾记录。

2026-07-18：x7 仍为 v2，trunk 已更新为 `124->256 stem + 12x(256->112->112->256)` PreAct bottleneck，
block 4、8 后做独立 mean/max Global Broadcast。正式模型参数 `5,690,808`；旧
`katago_gpool_value_aux_v1` checkpoint 不能续训或导出，需从头训练。

- 当前正式模型契约：`124x10x9 -> 2062 + WDL`；正式 UCI 只在 `WeightsFile` 成功加载 ONNX 后搜索，
  不回退到 `UniformBackend`。
- P4 单 worker 主线已可用：ONNX、MemCache、minibatch、prefetch、collision、shared-tree、watchdog、
  legacy clock manager 与 UCI 生命周期已接通。
- px0 默认 legacy 时钟预算已实现：`MoveOverheadMs`、`Slowmover`、
  `go wtime/btime/winc/binc/movestogo`。`depth/mate/ponder/ponderhit` 仍明确拒绝。
- 当前语义审计不改变 px0 `TaskWorkers=-1` 与 backend 派生 `SearchWorkers` 的默认行为；task-worker 并发
  路径视为既有实现，本轮只核验 UCI、tree reuse、terminal、selection、NN result 与 backup 的主线语义。
  后续若重新调整 task 生命周期，仍必须对照 px0 `src/search/classic/search.h:205-244,348-445` 与
  `search.cc:1069-1508`，不得回退到 raw-pointer 共享 tree。

本次代码验收：

- 语义审计已修复：terminal WDL 方向/终局 rank、two-fold reuse 基线与 `MakeNotTerminal` base WDL、
  root null bestmove、`movetime`/bare `go`/`infinite` stopper、NN policy 回填顺序。
- 真实 `x7.onnx` 已验证连续 `go infinite -> stop -> wait -> position -> go nodes`：两次请求各输出一次
  `bestmove`，无旧搜索串答。

```powershell
cargo fmt --check
cargo test --release -p engin --lib       # 116 passed
cargo test --release -p xiangqi_core      # 22 passed
cargo clippy -p engin --all-targets -- -D warnings
cargo build --release -p engin
git diff --check
```

真实 ONNX UCI 冒烟已验证 `WeightsFile`、legacy 时钟预算和 `bestmove`：

```powershell
@('uci', 'setoption name MoveOverheadMs value 0', 'setoption name Slowmover value 1.5',
  'setoption name WeightsFile value data\x7.onnx', 'isready', 'position startpos',
  'go wtime 1000 btime 1000 winc 10 binc 10', 'wait', 'quit') |
  .\target\release\engin.exe
```
