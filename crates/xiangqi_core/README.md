# xiangqi_core

象棋 **规则、位棋盘、合法着生成**（UCI 字符串，与 pyffish / Pikafish 坐标一致）。

## 来源与许可说明

`src/types.rs`、`src/misc.rs`、`src/board.rs`、`src/movegen.rs` 自本地 **pikafish-rust**（Pikafish 的 Rust 移植）拷贝，并增加全局 Zobrist、`Position::from_fen` / `new_with_global_zobrist`、`legal_moves_uci` 等库 API。上游 Pikafish 通常为 **GPL-3.0**；若发布衍生作品，请自行核对与仓库根目录许可证字段是否一致。

## API 摘要

- `Position::from_fen(&str)` / `set_fen`：局面
- `legal_moves_uci(&Position) -> Vec<String>`：合法着 UCI 串；**纵坐标为 1～10**（与 pyffish 一致，非 ICCS 的 0～9）
- `movegen::generate(..., GenType::Legal, ...)`：内部 `Move` 枚举
- `uci_format::START_FEN`：起始 FEN

## 测试

```bash
cargo test -p xiangqi_core
```

含 perft(1–3) 与 do/undo，与 pikafish-rust 测试数值对齐。
