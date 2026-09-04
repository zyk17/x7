# Commands

本文件只记录可复现的训练、验证、诊断与打包命令；目标和设计边界见 `README.MD` 与
`ARCHITECTURE.md`，当前工作队列见 `NextStep.md`。

默认先进入仓库根目录：

```powershell
Set-Location C:\projects\77xiangqi_engine
```

数据主线约定：

- Kaggle 数据集：`pikacat/px0data`
- 本地目录：`C:\work\px0data\{version}\`
- 数据准备入口：`nn\scripts\data\prepare_px0.py --config nn\configs\<name>.yaml`
- 训练入口：`nn\scripts\train\train_px0.py --config nn\configs\<name>.yaml`
- 准备脚本负责下载、解压、train/val 切分和固定 validation manifest；训练不会再做这些工作

## 1. 首次准备环境

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pip install -e "nn[train,dev]"
```

这会安装 PyTorch、ONNX、Kaggle、PyYAML、pytest 和 ruff。配置要求 `training.device: cuda` 时，
训练不会静默降级到 CPU；只有写为 `auto` 才允许回退。

## 2. 准备数据

每个 `px0_version + val_ratio + seed` 组合只需要准备一次。此步骤可能较慢：首次下载、解压和固定
train/validation chunk split 都在这里完成。

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\data\prepare_px0.py `
  --config nn\configs\x7_v3_01.yaml
```

## 3. 用 YAML 训练

```powershell
Copy-Item nn\configs\example.yaml nn\configs\x7_v3_01.yaml
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --config nn\configs\x7_v3_01.yaml
```

配置分为 `dataset`、`model`、`training` 三段，格式参考
`C:\Users\Administrator\projects\pxzero-training\tf\configs\example.yaml`，但只保留当前 PyTorch/PX0
主线需要的字段。正式契约固定为 `124x10x9 -> 2062 + WDL + moves-left`；x7 v2 的纯 CNN trunk、
正式 head 和 loss 语义不能通过配置切换。训练期 Auxiliary Soft Policy 与 root-WDL head 不进入 ONNX。
优化器固定为 AdamW：Conv/Linear weights 使用 decoupled weight decay，BatchNorm 与 bias 不 decay；学习率为
线性 warmup 后 cosine decay。
所有可配置字段、默认值与注释见 [nn/configs/example.yaml](C:/projects/77xiangqi_engine/nn/configs/example.yaml)。
`example.yaml` 是唯一提交到 Git 的配置文件；复制出的实验 YAML 被忽略。

训练只加载已准备 manifest；若准备结果缺失或 YAML 的 dataset 参数变了，会立即失败而不是自动等待。

## 4. 继续训练

再次运行同一条 YAML 命令即可。`training.out` 已存在时自动恢复；只需把 YAML 中的
`training.steps` 调大。

## 5. 从旧权重初始化新实验

复制 YAML，修改 `name`、`training.out`、`training.init_from` 和 `training.steps`。`init_from` 只加载模型权重；
width、blocks、bottleneck_channels 必须与来源 checkpoint 一致。

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --config nn\configs\x7_v3_01.yaml
```

## 6. 强制重下并重建某个版本

把 YAML 的 `dataset.force_download` 改为 `true` 后，运行第 2 节准备命令；完成后改回 `false`。

## 7. 导出 best checkpoint 为 ONNX

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\export\export_onnx.py `
  --checkpoint data\checkpoints\x7_v2_01.best.pt `
  --out data\x7.onnx `
  --precision mixed-fp16
```

说明：

- `value` 输出是 `WDL` 概率
- 引擎侧按 `q = W - L` 消费 value
- 当前 trunk：`pre-activation bottleneck residual + two Global Broadcast`
- 当前 policy：`pure CNN conv head`
- 当前引擎正式消费：`policy + WDL + moves-left`
- Auxiliary Soft Policy 与 root-WDL 是训练期 head，不在 ONNX 中

## 8. 检查 ONNX 合约

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pytest nn\tests\test_policy_onnx_contract.py
```

## 9. 当前引擎 UCI 冒烟

当前引擎提供正式 UCI stdin/stdout 入口；另有 `nn_eval`、`benchmark`、`search_benchmark` 和 Pikafish 对照脚本用于本地诊断。权重由 `WeightsFile` 在下一条 `position` 前加载。

```powershell
@'
uci
setoption name WeightsFile value C:/projects/77xiangqi_engine/data/x7.onnx
isready
position startpos
go nodes 64
quit
'@ | C:\projects\77xiangqi_engine\target\release\x7.exe
```

验证真实 history：

```powershell
@'
uci
setoption name WeightsFile value C:/projects/77xiangqi_engine/data/x7.onnx
isready
position startpos moves h2e2 h7e7
go nodes 64
quit
'@ | C:\projects\77xiangqi_engine\target\release\x7.exe
```

默认不指定 `--bin` 时，`cargo run -p engin` 仍然启动主 UCI 引擎：

```powershell
cargo run --release -p engin
```

当前 UCI options 是：`WeightsFile`、`NnBatchSize`、`NnCacheSizePowerOfTwo`、`MultiPV`、`UCI_ShowWDL`、`UCI_ShowEPS`、
`CPuct`、`CPuctBase`、`CPuctFactor`、`FpuReduction`、`VarianceBonusScale`、`NnWindow`、`VirtualMeanFpuScale`、`DecisionRule`、`DecisionLcbStdevs`、`DecisionUcbStdevs`、`DecisionMixNWeight`、`Threads`。
`NnBatchSize` 使用 `0..=1024` 的整数，默认 `0`（backend 建议值）；一次 `setoption` 影响之后启动的每次 `go`，已运行搜索保留其 worker。
搜索参数默认 `CPuct=2.4`、`CPuctBase=40000`、`CPuctFactor=0`、`FpuReduction=0.225`、
`NnWindow=2.25`、`VarianceBonusScale=1.5`；
`DecisionLcbStdevs=1.0`、`DecisionUcbStdevs=1.0` 是一倍 SE 的温和置信修正，`DecisionMixNWeight=0.25` 表示最多 0.25 个 Q 单位的归一化 N 偏好；默认 `DecisionRule=Auto`（即 `MaxN`），因此三者只在切换到相应规则后生效。
`DecisionRule=Auto` 与 `MaxN` 都直接取最大 completed N；`MaxQ` 直接取最大 Q；`Lcb/Ucb` 分别直接取最大
`Q - DecisionLcbStdevs * SE` / `Q + DecisionUcbStdevs * SE`；`MixNQ` 直接取最大
`Q + DecisionMixNWeight * N/max(root child N)`。所有规则使用同一套 N、Q、prior 作为并列时的稳定次序，
已证明必胜的 terminal root child 优先于普通候选，并在多个必胜着中选最短 mate；其余候选不对
terminal、样本数或候选比例作额外覆盖或过滤。未满两个样本时 `SE=0`。
将 `VarianceBonusScale=0` 可关闭独立 SE verification bonus；非零时对至少两次 completed evidence 的 edge 加上不含 prior 的 `scale * SE`。它只为降低尚未收敛的 Q 的标准误提供平滑复核机会；`SE` 会随自身 evidence 增加自然衰减，常规 PUCT 的 Q 与 U 不变。

相关公式为：`score = Q_mean + U + VarianceBonusScale * SE`，其中 `SE=std/sqrt(N)`，
`std=sqrt(max(0, mean(w²)-Q_mean²))`，全部来自同一 edge 的原始 completed `wl` 样本。
`Threads=8` 只在 Gather/Eval 间近似平分，NN 与 Backprop 固定各一条线程。option 名称和布尔值大小写不敏感。

当前支持 `go nodes`、`movetime`、`wtime/btime/winc/binc/movestogo`、`infinite` 与 `searchmoves`。
`movetime` 不可与时钟字段混用；`infinite` 不可与其他预算混用。`depth`、`mate`、`ponder` 仍会明确报错。

连续命令生命周期冒烟。按 px0 语义，前一条无限搜索被静默回收，只有最后一条 `go` 返回
`bestmove`：

```powershell
@'
uci
setoption name WeightsFile value C:/projects/77xiangqi_engine/data/x7.onnx
isready
position startpos
go infinite
position startpos moves h2e2 h7e7
go nodes 64
wait
quit
'@ | C:\projects\77xiangqi_engine\target\release\x7.exe
```

`setoption` 示例：

```powershell
@'
uci
setoption name NnBatchSize value 64
setoption name WeightsFile value C:/projects/77xiangqi_engine/data/x7.onnx
isready
position startpos
go nodes 64
quit
'@ | C:\projects\77xiangqi_engine\target\release\x7.exe
```

## 10. 搜索与 ONNX 诊断

`stream` 已是正式 UCI 搜索。`benchmark` 从 fresh tree 运行，输出吞吐 / NN·cache / 队列 / 深度 / root；
可选 `--collision-dist`、`--tree-depth`、`--trace`：

```powershell
cargo run --release -p engin --bin benchmark -- `
  --movetime 3000 --repeat 3 `
  --gathers 3 --evals 5
```

它也可在同一 fresh-tree 批次内固定 `--cpuct`、`--cpuct-factor`、`--fpu-reduction`、`--nn-window` 与
`--virtual-mean-fpu-scale`；这用于联合观察基础树形和流水线窗口，不替代 `search_benchmark` 的参数扫描。

`search_benchmark` 固定 `4/4` Search/Eval worker 和 backend 默认 batch，只比较 cPUCT/FPU 下的 fresh-tree 根部分流。使用完整历史诊断评分拐点：

```powershell
cargo run --release -p engin --bin search_benchmark -- `
  --moves "c3c4 g6g5 ..." --playouts 2000 --root-top 12
```

扫描独立的目标 SE bonus（每组均为 fresh tree）：

```powershell
cargo run --release -p engin --bin search_benchmark -- `
  --moves "c3c4 g6g5 ..." --playouts 20000 --root-top 12 `
  --variance-bonus-scale 0,0.05,0.1,0.2
```

查看固定参数下的根访问集中度、collision 和主干树形；`--repeat` 必须大于 1，避免将异步调度的单次分流当作结论：

```powershell
cargo run --release -p engin --bin benchmark -- `
  --fen "2bak4/3PaP1P1/9/4n4/2b6/6B2/3p5/4BA3/5C3/3K1A3 w - - 0 1" `
  --playouts 20000 --repeat 3 --gathers 4 --evals 4 `
  --variance-bonus-scale 0.1 `
  --root-top 8 --tree-depth 3 --tree-top 3
```

`benchmark` 输出正常 cache hit（`hit`）和流水线队列数据；`search_benchmark` 输出根候选的
`P / completed-N / in-flight / Q`，可加 `--trace` 和 `--track` 观察固定节点轨迹。两者都不模拟实战 tree reuse。`nn_eval` 可单独检查 ONNX：

```powershell
cargo run --release -p engin --bin nn_eval -- --onnx data\x7.onnx --bench 20
```

以本地 Pikafish 对照候选排序：

```powershell
python scripts\pikafish_compare.py --moves "c3c4 g6g5 ..." --movetime 3000 --multipv 3
```

脚本固定设置 Pikafish `Threads=4`、`Hash=512 MB`；不要把其原始默认的 `1` thread、`16 MB`
当作搜索质量对照。

## 11. 质量检查

在仓库根目录执行：

```powershell
cargo fmt --check
cargo test -p engin --lib
cargo clippy -p engin --all-targets -- -D warnings
```

规则层的 release perft 对拍较慢，按需单独运行：

```powershell
cargo test --release -p xiangqi_core
```

NN 检查使用项目自己的虚拟环境：

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m ruff check nn\src nn\scripts nn\tests
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pytest nn\tests -q
```

需要检查格式而非修正格式时：

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m ruff format --check nn\src nn\scripts nn\tests
```

## 12. Windows DirectML 打包

清理 Rust 构建产物、以 DirectML-only `ort` 重编译并生成可分发目录。脚本会复制实际需要的
DirectML provider DLL，并确认 bundle 内的 UCI 搜索没有回退 CPU：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1
```

默认读取 `data\x7.onnx`，输出到 `bundle-directml\`。可指定其他模型或输出目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1 `
  -ModelPath C:\models\x7.onnx `
  -BundleDir C:\dist\x7-directml
```

本机日常开发默认使用 DirectML。大量 playout 的 benchmark 才显式使用 TensorRT；它需要 CUDA 13（含
nvcc）、cuDNN 9、TensorRT 10（`nvinfer_10.dll`）。路径写在 `scripts/build-tensorrt.ps1` 与
`crates/engin/build.rs` 头部（`CUDA_PATH` / `X7_MSVC_BIN` 可覆盖）。

开发构建：

```powershell
cargo run -p engin --release

# 大节点 benchmark：显式切 TensorRT。
cargo run -p engin --release --no-default-features --features tensorrt --bin search_benchmark -- --playouts 25000
```

打包：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-tensorrt.ps1
```

它输出到 `bundle-tensorrt\`，并复制配置 `tensorrt_libs` 目录的全部 DLL（包括
`nvinfer_builder_resource_*`）；目标机仍须提供 CUDA 13 / cuDNN 9。TensorRT 与 DirectML 是两份互斥包，不构成运行时回退链。
发行包里的 `trt_cache` 保持为空；engine 由用户首跑按本机 GPU 构建。
