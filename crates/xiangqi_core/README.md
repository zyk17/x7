# xiangqi_core

中国象棋 **规则、位棋盘表示、合法着生成** 的 Rust 库。着法 UCI 串为 `a0`～`i9` 形式，与常见皮卡鱼族引擎的坐标约定一致，便于对拍与联调。

## API 摘要

- `Position::from_fen(&str)` / `set_fen`：局面
- `legal_moves_uci(&Position) -> Vec<String>`：合法着 UCI 串；**纵坐标为 0～9**（`a0`～`i9`）
- `parse_move_uci(s: &str) -> Option<Move>`：解析着法串（不校验合法）；`uci_to_move` 需局面
- `movegen::generate(..., GenType::Legal, ...)`：内部 `Move` 枚举
- `uci_format::START_FEN`：起始 FEN

## 测试

```bash
cargo test -p xiangqi_core
```

含 perft(1–3) 与走子/撤销回归；数值可与参考实现对照，用于防止无意回归。

## 与 pyffish 合法集对拍

辅助二进制 **`legal_moves_dump`**：对给定 **根 FEN** 与可选 **pyffish UCI 前缀**（空格分隔，从根局面依次执行），将排序后的合法 UCI 逐行打印，便于与 `pyffish.legal_moves("xiangqi", fen, prefix_list)` 对比。

```bash
# 仓库根目录
cargo run -p xiangqi_core --bin legal_moves_dump -- --help
cargo run -q -p xiangqi_core --bin legal_moves_dump -- \
  --fen "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1" \
  --prefix "b1c3"
```

一键脚本（需 Python **`pyffish`**、`cargo` 在 PATH）：见 **`nn/scripts/parity/pyffish_xiangqi_core_parity.py`**；或在 **`nn/`** 下 `pytest tests/test_pyffish_xiangqi_core_parity.py`。

说明：个别路径涉及亚洲象棋「捉」「闲着」「不变作和」等时，仅靠静态 FEN 无法复原完整历史时可能与 pyffish 分歧；本脚本用例选用 **根 FEN + 前缀重演** 的种子局面以降低此类噪声。
