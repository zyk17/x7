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
  --px0-version 7 `
  --max-files 8 `
  --max-samples 256
```

## 3. 直接训练指定版本

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --px0-version 7 `
  --out data\checkpoints\baseline_px0_wdl_v1.pt `
  --width 96 `
  --blocks 6 `
  --batch-size 256 `
  --steps 200000 `
  --eval-every 1000 `
  --val-batches 32 `
  --num-workers 4 `
  --device cuda `
  --q-ratio 1.0
```

## 4. 继续训练

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --px0-version 7 `
  --out data\checkpoints\baseline_px0_wdl_v1.pt `
  --width 96 `
  --blocks 6 `
  --batch-size 256 `
  --steps 200000 `
  --eval-every 1000 `
  --val-batches 32 `
  --num-workers 4 `
  --device cuda `
  --q-ratio 1.0 `
  --resume
```

## 5. 强制重下并重建某个版本

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --px0-version 7 `
  --px0-force-download `
  --out data\checkpoints\baseline_px0_wdl_v1.pt `
  --width 96 `
  --blocks 6 `
  --batch-size 256 `
  --steps 200000 `
  --eval-every 1000 `
  --val-batches 32 `
  --num-workers 4 `
  --device cuda `
  --q-ratio 1.0
```

## 6. 如果你确实要手动生成 train/val manifest

训练主线不需要这一步；只有想固定文件切分时才用：

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\data\split_px0_files.py `
  --px0-version 7 `
  --out-train data\rounds\px0_train_v1.json `
  --out-val data\rounds\px0_val_v1.json `
  --val-ratio 0.1 `
  --seed 42
```

## 7. 导出 best checkpoint 为 ONNX

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\export\export_onnx.py `
  --checkpoint data\checkpoints\baseline_px0_wdl_v1.best.pt `
  --out data\checkpoints\baseline_px0_wdl_v1.best.onnx
```

说明：

- `value` 输出是 `WDL` 概率
- 引擎侧按 `q = W - L` 消费 value
- `q_ratio=1.0` 表示纯搜索监督

## 8. 检查 ONNX 合约

```powershell
Copy-Item data\checkpoints\baseline_px0_wdl_v1.best.onnx data\policy.onnx -Force
```

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pytest nn\tests\test_policy_onnx_contract.py
```

## 9. 引擎 ONNX 冒烟

```powershell
cargo run --release -p engin -- --onnx-smoke data\checkpoints\baseline_px0_wdl_v1.best.onnx
```

这条命令只验证最小推理链路，不代表 GUI 正式接入效果。

## 10. 引擎 bench

```powershell
cargo run --release -p engin -- --bench --playouts 64 --onnx data\checkpoints\baseline_px0_wdl_v1.best.onnx --require-onnx
```

## 11. 最小 UCI 联调

```powershell
@'
uci
setoption name PolicyFile value C:/projects/77xiangqi_engine/data/checkpoints/baseline_px0_wdl_v1.best.onnx
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
setoption name PolicyFile value C:/projects/77xiangqi_engine/data/checkpoints/baseline_px0_wdl_v1.best.onnx
isready
position startpos moves h2e2 h7e7
go nodes 64
quit
'@ | C:\projects\77xiangqi_engine\target\release\engin.exe
```

## 12. 质量检查

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

## 13. 清理训练产物

```powershell
Remove-Item -Force data\checkpoints\*.pt, data\checkpoints\*.onnx
Remove-Item -Force data\policy.onnx
```
