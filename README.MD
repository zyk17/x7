# 77xiangqi_engine

> 棋力不是来自网络，也不是来自搜索。棋力来自 Prediction 与 Evidence 的协同；Knowledge 负责产生 Prediction，Proof 负责产生新的 Evidence，Decision 在固定时间预算下融合二者，形成最终行动。

这是一个独立的中国象棋研究引擎（X7）。项目目标是在有限计算资源、固定思考时间下研究上述协同，并只以固定时间 Elo 评估成果；网络架构、搜索算法和硬件分工都是实现选择，不是项目目标。

第一版规则、网络外围、UCI 与训练格式参考过 px0 / Lc0 / LC3 公开文档；**当前正式搜索是本仓的 stream MCGS，不再是 px0 重写，也不承诺与 px0/LC3 行为等价**。搜索采用连续的 `Gather → Eval → NN → Eval → Backprop` 流水线：Gather 直接提交一个 owned leaf；Eval 处理终局、缓存、编码和 NN 输出解码；NN 只做 tensor 合批与推理。NN 空闲时立即执行队列中当前可得的 tensor，不等待逻辑搜索轮次。owner 最多保留两个 `MiniBatchSize` 的 in-flight owned event 窗口，避免无界过时统计。当前实战使用 `μ=FPU` 的临时 virtual mean，完成或取消时精确归还，不写入 completed Evidence；collision 直接取消 reservation。没有 multivisit、prefetch 或 tree-batch gather。`mcts` 分支可保留树 MCTS；本分支研究 stream MCGS。两者共用规则、网络与 UCI 外围，模型契约固定为 `124 × 10 × 9 → 2062 policy + WDL + moves-left`，搜索值使用 `win - loss`。训练期辅助头不进入 ONNX 正式契约。研究原则、参考边界与架构约束见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 快速运行

正式 UCI 必须成功加载 ONNX 权重，不会回退到测试用的均匀网络。启动时依次查找可执行文件同目录、
当前目录和 `data/x7.onnx`；也可以明确指定权重：

```text
setoption name WeightsFile value C:/models/x7.onnx
position startpos
go nodes 1000
```

本机日常开发默认构建 DirectML，也是通用发行线；打包使用 `scripts/build-directml.ps1`。大节点
benchmark 才显式使用 `--no-default-features --features tensorrt`。TensorRT 打包用 `scripts/build-tensorrt.ps1`
（脚本头部固定本机 ORT / CUDA / TensorRT 路径，换机器改那几行即可）。ORT GPU 包需带
TensorRT EP，且 TensorRT **必须是 10.x**（`nvinfer_10.dll`）；不要装 TRT 11。本机另需
CUDA 13 / cuDNN 9。TensorRT 包会复制配置 `tensorrt_libs` 目录中的全部 DLL（包括
`nvinfer_builder_resource_*`），因此目标机无需再单独安装 TensorRT。发行时显式选择两种互斥包：`build-directml.ps1` 与 `build-tensorrt.ps1`，
二者不是同一二进制的运行时回退链。

## UCI 范围

目前支持：

- `position startpos|fen ... [moves ...]`
- `go nodes`、`go movetime`、`go wtime/btime/winc/binc/movestogo`、`go infinite`、`go searchmoves`
- `stop`、`ucinewgame`、`quit`
- `UCI_ShowWDL`、`UCI_ShowEPS`、`WeightsFile`、`MiniBatchSize`、`NNCacheSizePowerOfTwo`、`MultiPV`
- `CPuct`、`CPuctBase`、`CPuctFactor`、`FpuReduction`、`LcbStdevs`、`LcbMinVisitFraction`、`Threads`

`MiniBatchSize` 沿用常见 UCI 同名 option（历史上与 px0/Lc0 同名），范围 `0..=1024`，默认 `0`
（由 backend 建议）。它限制一次 NN 调用最多合并的已编码局面，并派生连续流水线的两批 in-flight 上限；实际
NN batch 可能因终局、缓存、collision 或队列暂时不足更小。较大值通常提高吞吐，却也会增大 pending
近似对固定时间棋力的影响。它是引擎生命周期配置：一次 `setoption` 会影响之后启动的每次
`go`；正在运行的搜索保持其 worker。

`NNCacheSizePowerOfTwo` 对应 KataGo 的同名配置，范围 `0..=48`，默认 `20`，即
`1,048,576` 个槽（KataGo GTP 默认）。它是按当前棋盘 state key 的直映缓存：索引冲突时新结果替换旧结果，完整
64 位 key 与合法着数量仍会在命中时校验。它不把完整历史加入 key，与 MCGS 的 board node
identity 一致。`setoption` 在下一次 `go` 开始前替换缓存表；当前搜索继续使用原表。

`MultiPV` 是 `1..=500` 的整数，默认 `1`。它只控制每次 `info` 输出的根候选数量：候选沿用
bestmove 的既有根边排序与各自的 WDL、分数、将杀距离和 PV，不改变搜索预算、PUCT 或 worker。

`CPuct`、`CPuctBase`、`CPuctFactor` 默认分别为 `1.75`、`40000`、`4.0`。所有 node
共用同一条 cPUCT 曲线，`C(0)=1.75`，不再维护根专用参数；`CPuctBase` 决定增长何时显著，`CPuctFactor` 决定增长幅度。
三者仍可作为一个整体实验，但默认不再扩散扫描。`FpuReduction`（默认 `0.200`）控制未访问边相对 parent Q
的降幅；小网络可能有系统性偏差，故降低未知候选的首次入场门槛。LC0 对照值 `0.330` 与原 X7
基线 `CPuct=1.0`、`FpuReduction=0.220` 保留为实验参数而非默认值。`setoption` 的 option 名称与布尔值大小写不敏感。`Threads` 默认 `8`，只在 Gather 和 Eval 间近似平分（默认 `4/4`）；NN 与 Backprop 各固定一条线程，不计入该值。范围为 `2..=128`。它与 `MiniBatchSize` 一样是引擎生命周期配置：`setoption` 影响之后的每次 `go`，当前 job 始终使用启动时的参数与 worker pool。

`LcbStdevs=5.0`、`LcbMinVisitFraction=0.15` 只影响根的最终 Decision，不参与 PUCT 或 visit 分配。
根候选先按既有终局/N/Q/P 排名确定访问门槛；达到 N 第一候选 15% 的非终局候选再按 `Q - k·标准误`
比较。`LcbStdevs=0` 可退回既有 N-first。二阶矩遵循 MCGS shared-Q 的幂等重算，故这是 X7 对
KataGo LCB 思路的简化借鉴，不是其围棋 utility/weight 公式的移植。

时钟使用固定中性的时间分配（历史上参考 px0 legacy stopper 形状），固定 `MoveOverhead=200ms`，不额外暴露 UCI 调参项。`go movetime`
不可与时钟字段混用；`go depth`、`go mate`、`go ponder` 与 `ponderhit`
尚无完整语义，会明确拒绝。

## 基准测试

`benchmark` 用于比较 stream 流水线吞吐、batch 和 collision，不经过 UCI。代表局面保存在
`data/benchmark_positions.txt`；每行是“名称 | FEN”，使用时只复制 `|` 后的完整 FEN：

```powershell
cargo run -p engin --release --features benchmark --bin benchmark -- --fen "<FEN>" --movetime 3000 --repeat 3 --gathers 4 --evals 4
```

`benchmark` 默认每轮都使用 fresh graph 与 fresh backend/cache，适合比较流水线本身。仅诊断 NN
cache 跨图复用时加入 `--warm-cache`：每个 worker 配置的首轮填表，后续 `--repeat` 轮仍使用 fresh
graph、但共享同一 NN cache；输出的 `hit` 是实际跳过 NN 的次数，不能与 graph reuse 混淆。

需要对照 pure virtual visit 与 virtual mean 时，传入 `--virtual-mean-fpu-scale <x>`；不传就是
pure virtual visit。正式引擎当前固定 `x=1.0`，不暴露 UCI option。

要诊断实战中的评分突变，优先保留完整历史而不是只复制最终 FEN；分流参数使用独立的
`search_benchmark`，固定 `4/4` Search/Eval worker、backend 默认 batch 与 fresh graph：

```powershell
cargo run -p engin --release --bin search_benchmark -- --moves "c3c4 g6g5 ..." --playouts 2000 --root-top 12
```

固定节点扫描 cPUCT 形状时，`--cpuct`、`--cpuct-base` 与 `--cpuct-factor` 接受逗号列表并交叉运行；`--fpu-reduction` 每次只取一个值，比较两套 FPU 时分别运行：

```powershell
cargo run -p engin --release --bin search_benchmark -- --onnx data/x7_v3_b12c512_v697.onnx --fen "<FEN>" --playouts 1000 --cpuct 1.0,1.745 --cpuct-base 20000,38739,80000 --cpuct-factor 2.5,3.894,5.0 --fpu-reduction 0.200 --root-top 3
```

`--trace 128,256,512,...` 只用于固定 `--playouts`：它在同一个 fresh graph 上输出各个
completed-visit 里程碑的 root `P/N/Q`，用于比较“新 Q 证据出现后，N 是否改变”。每个里程碑
会先 drain 在途事件再取快照，因此它是可重复的 completed-visit 轨迹，不等价于一段完全不中断的
流水线搜索。

`--track g6g9,i0g0,...` 可在每个 trace 里程碑额外输出指定根着的 `N` 排名和 `P/N/Q`；即使
该着不在 `--root-top` 中也会显示，适合观察 challenger 是未访问、被 Q 反驳，还是没有获得后续预算。

`--searchmoves "g6g9 i0g0 ..."` 与 UCI `go searchmoves` 使用相同的根过滤语义，可将比较限制在
指定候选中；它用于诊断候选获得预算后的 Q/N 轨迹，不把外部引擎的候选当作真值。

`graph_shape` 从固定节点预算后的 fresh graph 递归打印形状：每一行是一个已访问 node，只显示进入该 node
的 `P/N/Q/M`；root 没有入边，故只显示 `N/Q/M`。`M` 是 moves-left 的平均 ply 数。`--depth` 限制展示深度，`--top` 限制每个节点展示的
已访问子节点数。首行同时给出总 collision 及其占所有提交尝试的比例；它用于看图的访问形状，而不是记录运行中的 collision 热点：

```powershell
cargo run -p engin --release --features benchmark --bin graph_shape -- --fen "<FEN>" --playouts 2000 --depth 4 --top 8
```

`--moves` 从 startpos（或指定的 `--fen`）依次应用 ICCS 着法；它保留重复与规则历史，
但每次 benchmark 仍从 fresh graph 开始，不模拟前几回合的 graph reuse。
固定的完整历史搜索诊断局面放在 `data/search_cases.txt`；其中 `punish_low_prior_01`
用于观察黑方失误后低先验候选 `g6g9` 是否能获得足够的后续访问，不能把外部引擎的
短 PV 或 mate 输出当作规则正确性的断言。

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
| `crates/engin` | UCI、ONNX 外围、stream 搜索与时钟管理。 |
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
powershell -ExecutionPolicy Bypass -File .\scripts\build-tensorrt.ps1
```
