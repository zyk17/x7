# Commands

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

每个 `px0_version + val_ratio + seed + validation_*` 组合只需要准备一次。此步骤可能较慢：首次下载、解压和
validation manifest 的局面扫描都在这里完成。

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\data\prepare_px0.py `
  --config nn\configs\x7_v2_01.yaml
```

## 3. 用 YAML 训练

```powershell
Copy-Item nn\configs\example.yaml nn\configs\x7_v2_01.yaml
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --config nn\configs\x7_v2_01.yaml
```

配置分为 `dataset`、`model`、`training` 三段，格式参考
`C:\Users\Administrator\projects\pxzero-training\tf\configs\example.yaml`，但只保留当前 PyTorch/PX0
主线需要的字段。`124x10x9 -> 2062 + WDL`、纯 CNN trunk 与 loss 语义固定，不能通过配置切换。
优化器固定为 pxzero-training 的 `SGD(momentum=0.9, nesterov=true)`。
所有可配置字段、默认值与注释见 [nn/configs/example.yaml](C:/projects/77xiangqi_engine/nn/configs/example.yaml)。
`example.yaml` 是唯一提交到 Git 的配置文件；复制出的实验 YAML 被忽略。

训练只加载已准备 manifest；若准备结果缺失或 YAML 的 dataset 参数变了，会立即失败而不是自动等待。

## 4. 继续训练

再次运行同一条 YAML 命令即可。`training.out` 已存在时自动恢复；只需把 YAML 中的
`training.steps` 调大。

## 5. 开启新阶段训练（从旧权重起步）

复制 YAML，修改 `name`、`training.out`、`training.init_from`、`training.q_ratio` 和 `training.steps`。
例如先用 `q_ratio=0.0` 训出第一阶段，再切到新的 `q_ratio`：

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --config nn\configs\x7_v2_01.yaml
```

## 6. 强制重下并重建某个版本

把 YAML 的 `dataset.force_download` 改为 `true` 后，运行第 2 节准备命令；完成后改回 `false`。

## 7. 导出 best checkpoint 为 ONNX

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\export\export_onnx.py `
  --checkpoint data\checkpoints\x7_v2_01.best.pt `
  --out data\x7.onnx
```

说明：

- `value` 输出是 `WDL` 概率
- 引擎侧按 `q = W - L` 消费 value
- 当前 trunk：`pre-activation residual + global-pooling residual`
- 当前 policy：`pure CNN conv head`
- 当前引擎正式只消费：`policy + WDL`
- `q_ratio=0.0` 表示纯最终结果监督
- `q_ratio=1.0` 表示纯搜索监督

## 8. 检查 ONNX 合约

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pytest nn\tests\test_policy_onnx_contract.py
```

## 9. 当前引擎 UCI 冒烟

当前引擎只提供正式 UCI stdin/stdout 入口；没有独立 ONNX evaluator、bench CLI 或 px0 trace
对拍脚本。权重由 `WeightsFile` 在下一条 `position` 前加载。

```powershell
@'
uci
setoption name WeightsFile value C:/projects/77xiangqi_engine/data/x7.onnx
isready
position startpos
go nodes 64
quit
'@ | C:\projects\77xiangqi_engine\target\release\engin.exe
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
'@ | C:\projects\77xiangqi_engine\target\release\engin.exe
```

默认不指定 `--bin` 时，`cargo run -p engin` 仍然启动主 UCI 引擎：

```powershell
cargo run --release -p engin
```

当前已翻译的 UCI options 是：`WeightsFile`、`MultiPV`、`PerPVCounters`、
`ScoreType`、`UCI_ShowWDL`、`UCI_ShowEPS`、`UCI_ShowMovesLeft`、`NNCacheSize`。其余搜索参数仍
是 Rust 内部 `SearchParams`，在对应 px0 option 层完整翻译前不伪造公开 UCI 选项。

当前已实现的 `go` 预算只有 `nodes`、`movetime`、`infinite`。`depth`、`mate`、`ponder` 与
`wtime/btime/winc/binc/movestogo` 会明确报错，直到 px0 对应 stopper、ponder 生命周期和
`SimpleTimeManager` 被逐函数翻译；不要把它们与 `nodes` 混用。

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
'@ | C:\projects\77xiangqi_engine\target\release\engin.exe
```

`setoption` 示例：

```powershell
@'
uci
setoption name MultiPV value 2
setoption name WeightsFile value C:/projects/77xiangqi_engine/data/x7.onnx
isready
position startpos
go nodes 64
quit
'@ | C:\projects\77xiangqi_engine\target\release\engin.exe
```

## 10. 质量检查

## 11. Windows DirectML 打包

清理 Rust 构建产物、以 DirectML-only `ort` 重编译并生成可分发目录。脚本会复制实际需要的
DirectML provider DLL，并确认 bundle 内的 UCI 搜索没有回退 CPU：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1
```

默认读取 `data\x7.onnx`，输出到 `bundle\`。可指定其他模型或输出目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-directml.ps1 `
  -ModelPath C:\models\x7.onnx `
  -BundleDir C:\dist\x7-directml
```

```powershell
cargo check
```

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m ruff check nn
```

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pytest nn\tests
```

```powershell
cargo test -p engin --lib
```

P1 px0 规则对拍包含 depth-5 perft，建议只在 release 运行：

```powershell
cargo test --release -p xiangqi_core
```
