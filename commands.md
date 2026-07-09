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
  --out data\checkpoints\baseline_px0_katago_v1.pt `
  --width 128 `
  --blocks 8 `
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
  --out data\checkpoints\baseline_px0_katago_v1.pt `
  --width 128 `
  --blocks 8 `
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
  --init-from data\checkpoints\baseline_px0_katago_v1.best.pt `
  --out data\checkpoints\baseline_px0_katago_v1_qmix025.pt `
  --width 128 `
  --blocks 8 `
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
  --out data\checkpoints\baseline_px0_katago_v1.pt `
  --width 128 `
  --blocks 8 `
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
  --checkpoint data\checkpoints\baseline_px0_katago_v1.best.pt `
  --out data\policy.onnx
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
cargo run --release -p engin -- --onnx-smoke data\policy.onnx
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
  --onnx data\policy.onnx `
  --fen "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1" `
  --topk 8
```

批量文件：

```powershell
cargo run --release -p engin --bin onnx_eval -- `
  --onnx data\policy.onnx `
  --input data\eval_positions.txt `
  --topk 8 `
  --out data\eval_positions.ndjson
```

## 12. 引擎 bench

```powershell
cargo run --release -p engin -- --bench --playouts 64 --onnx data\policy.onnx --require-onnx
```

固定搜索对照建议直接这样跑：

`MctsBatchCap=16`

```powershell
cargo run --release -p engin -- --bench `
  --onnx data\policy.onnx `
  --require-onnx `
  --movetime 2000 `
  --search-batch-size 16
```

`MctsBatchCap=32`

```powershell
cargo run --release -p engin -- --bench `
  --onnx data\policy.onnx `
  --require-onnx `
  --movetime 2000 `
  --search-batch-size 32
```

`MctsBatchCap=64`

```powershell
cargo run --release -p engin -- --bench `
  --onnx data\policy.onnx `
  --require-onnx `
  --movetime 2000 `
  --search-batch-size 64
```

说明：

- `nps` 这里按 `playouts / sec`
- `nodes` 是树节点总量
- `search_batch_size` 会写进输出 JSON，便于后续对照

## 13. 最小 UCI 联调

```powershell
@'
uci
setoption name PolicyFile value C:/projects/77xiangqi_engine/data/policy.onnx
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
setoption name PolicyFile value C:/projects/77xiangqi_engine/data/policy.onnx
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
- `MctsCpuct`
- `MctsFpuReduction`
- `MctsBatchCap`
- `MctsWorkers`

说明：

- 不再兼容 `Playouts / Visits / Cpuct / FpuReduction / SearchBatchSize / Threads`
- 当前默认 `MctsWorkers=8`
- 默认 `MctsBatchCap=2048`
- `MctsBatchCap` 是 batch 上限，不是固定目标值
- 搜索过程中的 `info string root_moves_top5` 现在格式为 `move:visits:q:prior`
- 目前建议优先测试的 `MctsBatchCap` 档位是 `64 / 128 / 256 / 512 / 1024 / 2048 / 4096 / 8192`

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
