# Commands

默认先进入仓库根目录：

```powershell
Set-Location C:\projects\77xiangqi_engine
```

数据主线约定：

- Kaggle 数据集：`pikacat/px0data`
- 本地目录：`C:\work\px0data\{version}\`
- 训练入口：`nn\scripts\train\train_px0.py --config nn\configs\<name>.yaml`
- 如果本地已有 `training.*.gz`，直接复用
- 如果只有 `archive.zip` / `data.bin`，自动解压整理
- 如果目录为空，自动从 Kaggle 下载

## 1. 首次准备环境

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pip install -e "nn[train,dev]"
```

这会安装 PyTorch、ONNX、Kaggle、PyYAML、pytest 和 ruff。配置要求 `training.device: cuda` 时，
训练不会静默降级到 CPU；只有写为 `auto` 才允许回退。

## 2. 用 YAML 训练

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --config nn\configs\x7_qmix_075_01.yaml
```

配置分为 `dataset`、`model`、`training` 三段，格式参考
`C:\Users\Administrator\projects\pxzero-training\tf\configs\example.yaml`，但只保留当前 PyTorch/PX0
主线需要的字段。`124x10x9 -> 2062 + WDL`、纯 CNN trunk 与 loss 语义固定，不能通过配置切换。
所有可配置字段、默认值与注释见 [nn/configs/example.yaml](C:/projects/77xiangqi_engine/nn/configs/example.yaml)。

## 3. 继续训练

再次运行同一条 YAML 命令即可。`training.out` 已存在时自动恢复；只需把 YAML 中的
`training.steps` 调大。

## 4. 开启新阶段训练（从旧权重起步）

复制 YAML，修改 `name`、`training.out`、`training.init_from`、`training.q_ratio` 和 `training.steps`。
例如先用 `q_ratio=0.0` 训出第一阶段，再切到新的 `q_ratio`：

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --config nn\configs\x7_qmix_025_01.yaml
```

## 5. 强制重下并重建某个版本

把 YAML 的 `dataset.force_download` 改为 `true` 后，运行第 2 节命令；完成后改回 `false`。

## 6. 导出 best checkpoint 为 ONNX

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\export\export_onnx.py `
  --checkpoint data\checkpoints\x7_qmix_075_01.best.pt `
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

## 7. 检查 ONNX 合约

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pytest nn\tests\test_policy_onnx_contract.py
```

## 8. 引擎 ONNX 冒烟

```powershell
cargo run --release -p engin -- --onnx-smoke data\x7.onnx
```

这条命令只验证最小推理链路，不代表 GUI 正式接入效果。

## 8.1 UCI 使用正式 ONNX 权重

`WeightsFile` 沿用 px0 的 UCI 名称，但本项目只接受 ONNX。权重会在下一条
`position` 前加载：

```powershell
cargo run --release -p engin
```

```text
uci
setoption name WeightsFile value data/x7.onnx
isready
position startpos
go nodes 1000
```

## 9. 独立 ONNX 局面评估

说明：

- 这是独立评估工具，不走 `engin` 主 UCI 入口
- 同时输出：
  - `policy_topk_legal`
  - `wdl`
  - `q`
- 输入支持：
  - 单行 `FEN`
  - `position startpos moves ...`
  - `position fen ... moves ...`

单局面：

```powershell
cargo run --release -p engin --bin onnx_eval -- `
  --onnx data\x7.onnx `
  --fen "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1" `
  --topk 8
```

批量文件：

```powershell
cargo run --release -p engin --bin onnx_eval -- `
  --onnx data\x7.onnx `
  --input data\eval_positions.txt `
  --topk 8 `
  --out data\eval_positions.ndjson
```

## 10. 引擎 bench

bench CLI（与 [`main.rs`](crates/engin/src/main.rs) 一致）：

- `--playouts N` — playout 上限
- `--nodes N` — UCI nodes 口径上限（含复用树 initial_visits）
- `--movetime MS` — 思考时间上限（毫秒）
- `--cpuct F` — PUCT 强度
- `--minibatch-size N` — gather batch（`0` = backend auto）
- `--threads N` — 搜索线程（`0` = auto）
- `--onnx PATH` / `--require-onnx` / `--fen FEN` / `--data-dir PATH`

```powershell
cargo run --release -p engin -- --bench --playouts 64 --onnx data\x7.onnx --require-onnx
```

固定 batch 对照：

`MinibatchSize=16`

```powershell
cargo run --release -p engin -- --bench `
  --onnx data\x7.onnx `
  --require-onnx `
  --movetime 2000 `
  --minibatch-size 16
```

`MinibatchSize=32`

```powershell
cargo run --release -p engin -- --bench `
  --onnx data\x7.onnx `
  --require-onnx `
  --movetime 2000 `
  --minibatch-size 32
```

`MinibatchSize=64`

```powershell
cargo run --release -p engin -- --bench `
  --onnx data\x7.onnx `
  --require-onnx `
  --movetime 2000 `
  --minibatch-size 64
```

说明：

- `nps` 这里按 `playouts / sec`（仅 completed playout，不含 collision / 未 backup 的 reservation）
- UCI `nodes` 报告 `本轮 playouts + 复用树 initial_visits`（lc0 `VisitsStopper` 口径）
- `go nodes N`：总 visits（含复用树）达到 N 时停止
- `minibatch_size` 会写进输出 JSON，便于后续对照
- `eval_cache.hits/misses` 使用 lookup 口径（可直接比较）
- `eval_cache.miss_keys` 是去重后真实未命中 key 数（更接近实际 NN 负载）
- `retry_without_playout`：预算未耗尽时 gather 返回 0 playout 的重试次数

P2 root 访问分配 A/B（固定局面）建议：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\search_regression.ps1
```

重点对比输出里的：

- `pv` 长度
- `seldepth`
- `root_moves[].visits` 分布

## 11. px0 Fixed-Nodes 对拍记录

这条命令分别运行本机 px0 二进制和本仓库 `engin`，为同一 FEN 和相同
`go nodes` 采集原始 UCI transcript。它不比较 score：px0 当前读取 `pb.gz`，
本仓库读取 ONNX，二者并非完全等价权重。重点人工对照 `nodes`、`depth`、
`seldepth`、PV 与 `bestmove`。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\compare_px0_trace.ps1 `
  -Nodes 10000
```

默认写入被 Git 忽略的 `logs\trace\`。可传入不同 FEN：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\compare_px0_trace.ps1 `
  -Nodes 10000 `
  -Fen "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
```

## 12. 最小 UCI 联调

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
`ScoreType`、`UCI_ShowWDL`、`UCI_ShowEPS`、`UCI_ShowMovesLeft`。其余搜索参数仍
是 Rust 内部 `SearchParams`，在对应 px0 option 层完整翻译前不伪造公开 UCI 选项。

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

## 13. 质量检查

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
