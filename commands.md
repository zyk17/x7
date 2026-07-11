# Commands

默认先进入仓库根目录：

```powershell
Set-Location C:\projects\77xiangqi_engine
```

数据主线约定：

- Kaggle 数据集：`pikacat/px0data`
- 本地目录：`C:\work\px0data\{version}\`
- 训练入口：`nn\scripts\train\train_px0.py --px0-version <version>`
- 如果本地已有 `training.*.gz`，直接复用
- 如果只有 `archive.zip` / `data.bin`，自动解压整理
- 如果目录为空，自动从 Kaggle 下载

## 1. 首次准备环境

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pip install kagglehub
```

## 2. 检查某个 px0 版本是否可读

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\data\inspect_px0.py `
  --px0-version 710 `
  --max-files 8 `
  --max-samples 256
```

## 3. 直接训练指定版本

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --px0-version 710 `
  --out data\checkpoints\x7_qmix_000_01.pt `
  --width 160 `
  --blocks 10 `
  --batch-size 256 `
  --steps 200000 `
  --eval-every 1000 `
  --val-batches 32 `
  --num-workers 4 `
  --device cuda `
  --q-ratio 0.0
```

## 4. 继续训练

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --px0-version 710 `
  --out data\checkpoints\x7_qmix_000_01.pt `
  --width 160 `
  --blocks 10 `
  --batch-size 256 `
  --steps 200000 `
  --eval-every 1000 `
  --val-batches 32 `
  --num-workers 4 `
  --device cuda `
  --q-ratio 0.0
```

说明：

- `--out` 文件已存在时，脚本默认自动续训
- 不需要再传 `--resume`

## 5. 开启新阶段训练（从旧权重起步）

例如先用 `q_ratio=0.0` 训出第一阶段，再切到新的 `q_ratio`：

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --px0-version 710 `
  --init-from data\checkpoints\x7_qmix_075_01.best.pt `
  --out data\checkpoints\x7_qmix_025_01.pt `
  --width 160 `
  --blocks 10 `
  --batch-size 256 `
  --steps 30000 `
  --eval-every 1000 `
  --val-batches 32 `
  --num-workers 4 `
  --device cuda `
  --q-ratio 0.25
```

## 6. 强制重下并重建某个版本

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --px0-version 710 `
  --px0-force-download `
  --out data\checkpoints\x7_qmix_000_01.pt `
  --width 160 `
  --blocks 10 `
  --batch-size 256 `
  --steps 200000 `
  --eval-every 1000 `
  --val-batches 32 `
  --num-workers 4 `
  --device cuda `
  --q-ratio 0.0
```

## 7. 如果你确实要手动生成 train/val manifest

训练主线不需要这一步；只有想固定文件切分时才用：

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\data\split_px0_files.py `
  --px0-version 710 `
  --out-train data\rounds\px0_train_v1.json `
  --out-val data\rounds\px0_val_v1.json `
  --val-ratio 0.1 `
  --seed 42
```

## 8. 导出 best checkpoint 为 ONNX

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

## 9. 检查 ONNX 合约

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pytest nn\tests\test_policy_onnx_contract.py
```

## 10. 引擎 ONNX 冒烟

```powershell
cargo run --release -p engin -- --onnx-smoke data\x7.onnx
```

这条命令只验证最小推理链路，不代表 GUI 正式接入效果。

## 11. 独立 ONNX 局面评估

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

## 12. 引擎 bench

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

## 13. 最小 UCI 联调

```powershell
@'
uci
setoption name PolicyFile value C:/projects/77xiangqi_engine/data/x7.onnx
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
setoption name PolicyFile value C:/projects/77xiangqi_engine/data/x7.onnx
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

当前主 UCI 公开选项建议只关注：

- `PolicyFile`
- `MctsPlayouts`
- `CPuct`
- `FpuValue`
- `MinibatchSize`
- `NNCacheSize`
- `Threads`
- `MultiPV`

说明：

- 不再兼容旧别名：`Playouts / Visits / MctsCpuct / MctsFpuReduction / MctsBatchCap / MctsWorkers / SearchBatchSize`
- 当前默认 `Threads=0`、`MinibatchSize=0`
- `Threads=0` 表示按 backend attrs 自动推导
- `MinibatchSize=0` 表示按 backend `recommended_batch_size`
- `MultiPV=1` 为默认；`>1` 时每条 `info` 带 `multipv N`，共用 `depth/seldepth/time/nodes/nps`
- 当前不含 lc0 `PerPVCounters`（每条线独立 `nodes`）；默认与 lc0 `PerPVCounters=false` 一致

`setoption` 示例：

```powershell
@'
uci
setoption name CPuct value 1.745
setoption name Threads value 4
setoption name MultiPV value 2
setoption name PolicyFile value C:/projects/77xiangqi_engine/data/x7.onnx
isready
position startpos
go nodes 64
quit
'@ | C:\projects\77xiangqi_engine\target\release\engin.exe
```

## 14. 质量检查

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
