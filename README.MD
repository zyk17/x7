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
- `UCI_ShowWDL`、`UCI_ShowEPS`、`WeightsFile`、`VirtualLoss`

`VirtualLoss` 直接使用搜索参数 `V`，范围 `0.0..5.0`，默认 `3.0`，例如
`setoption name VirtualLoss value 2.5`。它是引擎
生命周期配置：一次 `setoption` 会影响之后启动的每次 `go`；正在运行的搜索保持启动时的参数快照。

时钟使用固定中性的 px0 分配，固定 `MoveOverhead=200ms`，不额外暴露 UCI 调参项。`go movetime`
不可与时钟字段混用；`go depth`、`go mate`、`go ponder` 与 `ponderhit`
尚无完整语义，会明确拒绝。

## 基准测试

`benchmark` 用于比较 stream 流水线吞吐、batch 和 collision，不经过 UCI。代表局面保存在
`data/benchmark_positions.txt`；每行是“名称 | FEN”，使用时只复制 `|` 后的完整 FEN：

```powershell
cargo run -p engin --release --bin benchmark -- --fen "<FEN>" --movetime 3000 --repeat 3 --gathers 4 --evals 4 --backprops 1 --virtual-loss 1
```

要诊断实战中的评分突变，优先保留完整历史而不是只复制最终 FEN：

```powershell
cargo run -p engin --release --bin benchmark -- --moves "c3c4 g6g5 ..." --movetime 3000 --repeat 3 --gathers 4 --evals 4 --backprops 1 --virtual-loss 1 --root-top 12
```

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

架构约束见 [ARCHITECTURE.md](ARCHITECTURE.md)。当前工作与待办只记录在本地的
[NextStep.md](NextStep.md) 和 [TODO.md](TODO.md)。训练说明见 [nn/README.md](nn/README.md)。

## 开发验证

```powershell
cargo fmt --check
cargo test -p engin --lib
cargo clippy -p engin --all-targets -- -D warnings
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build-cuda.ps1
```
