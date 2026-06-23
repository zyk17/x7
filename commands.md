# Commands

以下命令默认在仓库根目录执行：

```powershell
Set-Location C:\projects\77xiangqi_engine
```

## 1. 检查 `px0data` 是否可读

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\data\inspect_px0.py `
  --glob "C:\work\px0data\unpacked\run1\training.*.gz" `
  --max-files 8 `
  --max-samples 256
```

## 2. 生成 `px0` train/val 文件清单

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\data\split_px0_files.py `
  --glob "C:\work\px0data\unpacked\run1\training.*.gz" `
  --out-train data\rounds\px0_train_v1.json `
  --out-val data\rounds\px0_val_v1.json `
  --val-ratio 0.1 `
  --seed 42
```

## 3. 启动新的 `px0` baseline 训练

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\train\train_px0.py `
  --train-list data\rounds\px0_train_v1.json `
  --val-list data\rounds\px0_val_v1.json `
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
  --train-list data\rounds\px0_train_v1.json `
  --val-list data\rounds\px0_val_v1.json `
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

## 5. 导出当前 best baseline ONNX

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe nn\scripts\export\export_onnx.py `
  --checkpoint data\checkpoints\baseline_px0_wdl_v1.best.pt `
  --out data\checkpoints\baseline_px0_wdl_v1.best.onnx
```

说明：

- 现在导出的 `value` 是 `WDL` 概率，不再是单标量
- 引擎侧按 `q = W - L` 消费 value
- `q_ratio` 采用 lc0/px0 口径：`1.0=纯搜索`，`0.0=纯最终结果`

## 6. 检查 ONNX 合约

测试脚本默认检查 `data\policy.onnx`，所以先复制：

```powershell
Copy-Item data\checkpoints\baseline_px0_wdl_v1.best.onnx data\policy.onnx -Force
```

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pytest nn\tests\test_policy_onnx_contract.py
```

## 7. 引擎 ONNX 冒烟

```powershell
cargo run --release -p engin -- --onnx-smoke data\checkpoints\baseline_px0_wdl_v1.best.onnx
```

## 8. 引擎 MCTS 基准

```powershell
cargo run --release -p engin -- --bench --playouts 64 --onnx data\checkpoints\baseline_px0_wdl_v1.best.onnx --require-onnx
```

## 9. 最小 UCI 联调

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

## 10. 当前 baseline 结果

这一轮 `baseline_px0_wdl_v1` 的最终结论：

- 当前主线配置：`width=96 blocks=6 batch=256 q_ratio=1.0`
- 训练轮次：`200k steps`
- 当前 best checkpoint：`data\checkpoints\baseline_px0_wdl_v1.best.pt`
- 终盘附近最好验证指标：
  - `val_policy ~= 2.4583`
  - `val_value_ce ~= 0.7199`
  - `val_value_q_mse ~= 0.0654`

结论：

- 这版已经可作为当前正式 baseline
- 同配置继续硬跑收益已很小
- 下一步优先做 `导出 ONNX -> 引擎联调 -> 小规模对照实验`

## 11. 常用质量检查

```powershell
cargo check
```

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m ruff check nn
```

```powershell
C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pytest nn\tests\test_px0_record.py nn\tests\test_dataset_px0.py nn\tests\test_nn_smoke.py nn\tests\test_train_policy_unpack.py
```

```powershell
cargo test -p engin --lib
```

## 12. 清理训练产物

```powershell
Remove-Item -Force data\checkpoints\*.pt, data\checkpoints\*.onnx
Remove-Item -Force data\policy.onnx
```
