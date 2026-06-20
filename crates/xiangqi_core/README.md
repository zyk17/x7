# xiangqi_core

Rust 规则内核：

- FEN / 局面表示
- 合法着生成
- do / undo
- 基础终局判断

## 测试

```bash
cargo test -p xiangqi_core
```

## 调试

仓库里保留了 `legal_moves_dump` 这样的最小辅助二进制，供规则联调用。
