# 77xiangqi_engine

X7 是独立的中国象棋研究引擎。目标是在有限计算资源和固定思考时间下，让网络提供
Prediction、搜索产生 Evidence，并以固定时间 Elo 衡量二者协同的效果。网络结构、搜索算法和
硬件分工都是实现选择，不是目标本身。

第一版规则、UCI 与训练外围曾参考 px0/Lc0；当前实现不是 px0 重写，也不承诺与 px0、LC3
行为等价。正式模型契约固定为 `124 x 10 x 9 -> 2062 policy + WDL + moves-left`。

## 当前结构

| 模块 | 职责 |
| --- | --- |
| `crates/xiangqi_core` | 唯一规则真相：棋盘、合法着、FEN、历史、重复与裁判。 |
| `crates/engin` | UCI、ONNX 外围、stream 路径树搜索与时钟。 |
| `nn` | 训练、权重处理、ONNX 导出；不进入搜索热路径。 |

当前搜索是单一 stream 路径树：`NodeKey = hash_cat(parent, move)`，不合并换位。流水线为
`Gather -> Eval -> NN -> Eval -> Backprop`；每次 Gather 只提交一个叶子，规则终局在 Eval
阶段按 event 的完整 variation/history 裁决。NN 只合并已经编码的 tensor，不等待逻辑搜索轮次。

## 文档

- [ARCHITECTURE.md](ARCHITECTURE.md)：目标、边界和当前架构约束。
- [Research.md](Research.md)：已完成或否决的历史实验。
- [commands.md](commands.md)：训练、导出、UCI、诊断和打包命令。
- [NextStep.md](NextStep.md)：当前优先级和待验证假设。

## 快速验证

```powershell
cargo fmt --check
cargo test -p xiangqi_core
cargo test -p engin --lib
cargo clippy -p engin --all-targets -- -D warnings
```

正式 UCI 必须加载 ONNX 权重，不会回退到测试用的 `UniformBackend`。常用运行和打包方式见
[commands.md](commands.md)。
