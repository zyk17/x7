# Value 小评测集（`--value-probe`）

用于对照 **`NNLeafMode::Off`** 与 **`MainLeafOnly`** 在同一组固定 FEN 上的 `bestmove`，判断叶子 value 是否值得留在搜索主线。

注意：当前短期主产品是**复盘系统**。因此 `value-probe` 属于**长期搜索验证工具**，不是近期主阻塞项。

## 运行

**必须**能解析到并成功加载 `policy.onnx`（与 `--bench` 相同：`./data/`、`ENGIN_DATA_DIR`、`--data-dir`、`--onnx`）。未找到或未加载时进程直接退出，**不会**输出「假对比」表。

```bash
cargo run --release -p engin -- --value-probe --depth 4
```

可选：`--vocab`、`--nn-eval-budget N`、`--no-policy-ordering`。

## 输出

标准输出为一篇 **Markdown 表**，列含义：

| 列 | 含义 |
|----|------|
| off / main bestmove | 各模式下迭代加深结束时的 UCI 根着 |
| 自动判定 | `一样`：两列 UCI 相同；`不同`：需进一步判断 |
| 人填更好/更差 | 打印为占位符，请结合棋理或强引擎对照后手写 |

局面列表定义在 `crates/engin/src/value_probe.rs` 的 `VALUE_PROBE_CASES`（均为 **FEN 可解析且棋规合法** 的样例：开局 / 车抬头 / 中路有子隔断的残面 / 子力失衡）。

## 解读建议

- 若多数局面为「一样」且 `main` 侧 nps 可接受，value 更像**温和正则项**。
- 若「不同」集中在战术/危险类标签，应对照具体着法是否更合理，再决定是否加大预算或改为 `AllLeaf` 做消融。
