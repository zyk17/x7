# nn

当前 Python 训练栈主线已经固定为：

- `PX0 v6 chunks`
- 小型 `policy + value`
- 输入 `124 x 10 x 9`
- 输出 `2062 + WDL`

## 当前目录职责

- `src/nn/px0_record.py`
  解析最小 `px0 v6 classical` record
- `src/nn/dataset_px0.py`
  流式读取 `px0` chunks
- `scripts/data/inspect_px0.py`
  快速检查 chunk 是否可读
- `scripts/data/split_px0_files.py`
  生成 train / val 文件清单
- `scripts/train/train_px0.py`
  当前主线训练入口
- `scripts/export/export_onnx.py`
  导出 ONNX

## 当前原则

- 默认先纯 `px0`
- 不预设人类数据混入
- 不把 XRSH 当主训练数据路线
- value 主语义为 `WDL + qMix`
- 默认 `q_ratio=1.0`，先以搜索 WDL 为主监督

如果后续你彻底放弃 XRSH 支线，那么相关数据可以删掉，代码是否继续保留再单独决定。
