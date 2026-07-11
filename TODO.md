# TODO

> 只保留未完成事项。  
> 每个改动前，先补 `lc0` 参考文件、行号、本仓库落点。

## P3 lc0 基建剩余差距

### P3.1 time manager

- [ ] 把 `go wtime / btime / winc / binc / movestogo` 接成真正可执行的 stopper 组合
- [ ] 明确象棋侧与 lc0 国际象棋侧的最小必要偏差
- [ ] 补对应 UCI 测试

参考：

- `C:\Users\Administrator\projects\lc0\src\chess\uciloop.cc:221-243`
- `C:\Users\Administrator\projects\lc0\src\search\classic\stoppers\common.cc:118-165`
- 本仓库：
  - `C:\projects\77xiangqi_engine\crates\engin\src\uci.rs`
  - `C:\projects\77xiangqi_engine\crates\engin\src\mcts\config.rs`

### P3.2 UCI 生命周期补齐

- [ ] `ponder` 语义补齐
- [ ] `searchmoves` 语义补齐
- [ ] `mate` 语义补齐
- [ ] 补对应 `stop / new position / go` 回归测试

参考：

- `C:\Users\Administrator\projects\lc0\src\chess\uciloop.cc:167-245`
- `C:\Users\Administrator\projects\lc0\src\engine.cc:200-257`
- 本仓库：
  - `C:\projects\77xiangqi_engine\crates\engin\src\uci.rs`

### P3.3 并行细节继续对齐 lc0

- [ ] 核对当前 `shared-tree` worker 流水线与 lc0 `SearchWorker` 剩余差距
- [ ] 评估是否需要继续补 task workers / node-lock 细节
- [ ] 如果要补，先把参考位置和本地落点写清楚再改

参考：

- `C:\Users\Administrator\projects\lc0\src\search\classic\search.h:216-303`
- `C:\Users\Administrator\projects\lc0\src\search\classic\search.cc:1209-1439`
- `C:\Users\Administrator\projects\lc0\src\search\classic\search.cc:2018-2377`
- 本仓库：
  - `C:\projects\77xiangqi_engine\crates\engin\src\mcts\search.rs`
  - `C:\projects\77xiangqi_engine\crates\engin\src\mcts\worker.rs`
  - `C:\projects\77xiangqi_engine\crates\engin\src\policy_onnx.rs`

### P3.4 回归护栏

- [ ] 补 engine 侧 integration 回归测试
- [ ] 固定 bench 对照样例与输出字段检查
- [ ] 收口文档与真实接口同步检查流程

参考：

- `C:\Users\Administrator\projects\lc0\src\chess\uciloop.cc`
- `C:\Users\Administrator\projects\lc0\src\engine.cc`
- `C:\Users\Administrator\projects\lc0\src\search\classic\search.cc`
- 本仓库：
  - `C:\projects\77xiangqi_engine\crates\engin\tests\p3_integration.rs`
  - `C:\projects\77xiangqi_engine\crates\engin\src\benchmark.rs`
  - `C:\projects\77xiangqi_engine\crates\engin\src\uci.rs`

## P4 模型与训练

- [ ] 保持 `124x10x9 -> 2062 + WDL` 契约稳定
- [ ] 继续使用 `px0` 数据训练 baseline
- [ ] 基建稳定后再评估 `q_ratio` 分阶段脚本设计
- [ ] 基建稳定后再评估中残局 value 问题

参考：

- `C:\Users\Administrator\projects\lczero-training\README.md:37-57`
- `C:\Users\Administrator\projects\lczero-training\csrc\loader\stages\tensor_generator.cc:85-119`
- 本仓库：
  - `C:\projects\77xiangqi_engine\nn\scripts\train\train_px0.py`
  - `C:\projects\77xiangqi_engine\nn\src`
