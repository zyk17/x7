# 77xiangqi_engine

> 棋力不是来自网络，也不是来自搜索。棋力来自 Prediction 与 Evidence 的协同；Knowledge 负责产生 Prediction，Proof 负责产生新的 Evidence，Decision 在固定时间预算下融合二者，形成最终行动。

这是一个新的中国象棋研究引擎。项目目标是在有限计算资源、固定思考时间下研究上述协同，并只以固定时间 Elo 评估成果；网络架构、搜索算法和硬件分工都是实现选择，不是项目目标。

当前正式主线只有 stream tree MCTS；规则层参考 px0，模型契约固定为 `124 × 10 × 9 → 2062 policy + WDL + moves-left`，搜索值使用 `win - loss`。训练期辅助头不进入 ONNX 正式契约。研究原则、参考边界与架构约束见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 快速运行

正式 UCI 必须成功加载 ONNX 权重，不会回退到测试用的均匀网络。启动时依次查找可执行文件同目录、
当前目录和 `data/x7.onnx`；也可以明确指定权重：

```text
setoption name WeightsFile value C:/models/x7.onnx
position startpos
go nodes 1000
```

本机开发默认构建 DirectML。CUDA 保留为 NVIDIA 专用备用构建：`scripts/build-cuda.ps1` 默认从
`C:\projects\onnxruntime-cuda13\windows\runtimes\win-x64\native` 读取官方 ONNX Runtime 1.28 的 CUDA
DLL，复制到 `target/debug`、`target/release` 与 CUDA bundle；要换目录直接修改脚本中的路径。
本机仍需提供 CUDA 13 / cuDNN 9。
发行时显式选择两种互斥包：`build-directml.ps1` 构建通用 DirectML 包，`build-cuda.ps1` 构建 NVIDIA
CUDA 包。两者不是同一二进制的运行时回退链；CUDA 使用 ORT 的 `prefer_nhwc=1`，但当前端到端
吞吐仍不及 DirectML。

## UCI 范围

目前支持：

- `position startpos|fen ... [moves ...]`
- `go nodes`、`go movetime`、`go wtime/btime/winc/binc/movestogo`、`go infinite`、`go searchmoves`
- `stop`、`ucinewgame`、`quit`
- `UCI_ShowWDL`、`UCI_ShowEPS`、`WeightsFile`、`MiniBatchSize`、`MultiPV`
- `CPuct`、`CPuctBase`、`CPuctFactor`、`FpuReduction`、`GatherWorkers`、`EvalWorkers`、`BackpropWorkers`

`MiniBatchSize` 对应 px0/Lc0 的同名 option，范围 `0..=1024`，默认 `0`（由 backend
建议）。它限制一次 NN 调用最多合并的局面数；较大值通常提高吞吐，却可能因增加在途局面和
collision 降低固定时间棋力。它是引擎生命周期配置：一次 `setoption` 会影响之后启动的每次
`go`；正在运行的搜索保持其 worker。

`MultiPV` 是 `1..=500` 的整数，默认 `1`。它只控制每次 `info` 输出的根候选数量：候选沿用
bestmove 的既有根边排序与各自的 WDL、分数、将杀距离和 PV，不改变搜索预算、PUCT 或 worker。

`CPuct`（默认 `1.745`）控制初始探索强度；`CPuctBase`（默认 `38739`）决定 PUCT 随访问数增长何时显著，
`CPuctFactor`（默认 `3.894`）决定增长幅度；二者是当前主要搜索实验参数。`FpuReduction`
（默认 `0.330`）控制未访问边相对 parent Q 的降幅。原 X7 实验基线为 `CPuct=1.0`、`FpuReduction=0.220`，保留为对照参数而非默认值。`setoption` 的 option 名称与布尔值大小写不敏感。`GatherWorkers`、`EvalWorkers`、
`BackpropWorkers` 默认分别为 `4`、`4`、`1`，范围均为 `1..=64`。它们与 `MiniBatchSize` 一样是
引擎生命周期配置：`setoption` 影响之后的每次 `go`，当前 job 始终使用启动时的参数与 worker pool。

时钟使用固定中性的 px0 分配，固定 `MoveOverhead=200ms`，不额外暴露 UCI 调参项。`go movetime`
不可与时钟字段混用；`go depth`、`go mate`、`go ponder` 与 `ponderhit`
尚无完整语义，会明确拒绝。

## 基准测试

`benchmark` 用于比较 stream 流水线吞吐、batch 和 collision，不经过 UCI。代表局面保存在
`data/benchmark_positions.txt`；每行是“名称 | FEN”，使用时只复制 `|` 后的完整 FEN：

```powershell
cargo run -p engin --release --bin benchmark -- --fen "<FEN>" --movetime 3000 --repeat 3 --gathers 4 --evals 4 --backprops 1
```

要诊断实战中的评分突变，优先保留完整历史而不是只复制最终 FEN；分流参数使用独立的
`search_benchmark`，固定 `4/4/1` worker、backend 默认 batch 与 fresh tree：

```powershell
cargo run -p engin --release --bin search_benchmark -- --moves "c3c4 g6g5 ..." --playouts 2000 --root-top 12
```

固定节点扫描 cPUCT 形状时，`--cpuct`、`--cpuct-base` 与 `--cpuct-factor` 接受逗号列表并交叉运行；`--fpu-reduction` 每次只取一个值，比较两套 FPU 时分别运行：

```powershell
cargo run -p engin --release --bin search_benchmark -- --onnx data/x7_v3_b12c512_v697.onnx --fen "<FEN>" --playouts 1000 --cpuct 1.0,1.745 --cpuct-base 20000,38739,80000 --cpuct-factor 2.5,3.894,5.0 --fpu-reduction 0.330 --root-top 3
```

`--trace 128,256,512,...` 只用于固定 `--playouts`：它在同一棵 fresh tree 上输出各个
completed-visit 里程碑的 root `P/N/Q`，用于比较“新 Q 证据出现后，N 是否改变”。每个里程碑
会先 drain 在途事件再取快照，因此它是可重复的 completed-visit 轨迹，不等价于一段完全不中断的
流水线搜索。

`--track g6g9,i0g0,...` 可在每个 trace 里程碑额外输出指定根着的 `N` 排名和 `P/N/Q`；即使
该着不在 `--root-top` 中也会显示，适合观察 challenger 是未访问、被 Q 反驳，还是没有获得后续预算。

`--searchmoves "g6g9 i0g0 ..."` 与 UCI `go searchmoves` 使用相同的根过滤语义，可将比较限制在
指定候选中；它用于诊断候选获得预算后的 Q/N 轨迹，不把外部引擎的候选当作真值。

`--moves` 从 startpos（或指定的 `--fen`）依次应用 ICCS 着法；它保留重复与规则历史，
但每次 benchmark 仍从 fresh tree 开始，不模拟前几回合的 tree reuse。

诊断评分拐点时，可用本地 Pikafish 作为外部对照。`scripts/pikafish_compare.py` 只走标准
UCI 生命周期并打印最终 MultiPV；默认使用本机的 Pikafish 路径，也可用 `--engine` 覆盖：

```powershell
python scripts/pikafish_compare.py --moves "c3c4 g6g5 ..." --movetime 3000 --multipv 3
```

要比较指定根着的独立价值，可重复传入 `--searchmove`：

```powershell
python scripts/pikafish_compare.py --moves "c3c4 g6g5 ..." --searchmove c3c4 --movetime 3000
```

Pikafish 的 `score cp` 是对照分数；stream 的 `Q` 是未校准的网络搜索值，二者只比较候选
排序和趋势，不能直接作数值换算。

吞吐、EPS 和 collision 只能说明流水线利用率；棋力需要用固定局面或对局测试另行验证。

## 目录

| 目录 | 职责 |
| --- | --- |
| `crates/xiangqi_core` | 唯一的规则真相：棋盘、合法着、FEN、历史与裁判。 |
| `crates/engin` | UCI、ONNX 外围、stream 搜索、watchdog 与时钟管理。 |
| `nn` | 训练、权重处理与 ONNX 导出；不进入搜索热路径。 |

架构约束见 [ARCHITECTURE.md](ARCHITECTURE.md)，已完成研究尝试见
[Research.md](Research.md)。当前工作与待办只记录在本地的 [NextStep.md](NextStep.md) 和
[TODO.md](TODO.md)。训练说明见 [nn/README.md](nn/README.md)。

## 开发验证

```powershell
cargo fmt --check
cargo test -p engin --lib
cargo clippy -p engin --all-targets -- -D warnings
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build-cuda.ps1
```
