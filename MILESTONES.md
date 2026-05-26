# 里程碑（P0–P7）

与 **`ARCHITECTURE.md`** 配套；中期目标见 **`工程目标.md`**；近期执行顺序见 **`NEXT_STEPS.md`**。

| 阶段 | 目标 | 主要交付 / Crate |
|------|------|------------------|
| **P0** | 完整象棋规则 + 合法 UCI | `crates/xiangqi_core` |
| **P1** | canonical 词表 + PGN → XRSH | `crates/xiangqi_dataset` |
| **P2** | Python 训练接入 XRSH + 多头网络 | `nn/` |
| **P3** | UCI 引擎与搜索基础设施 | `crates/engin` |
| **P4** | 在强 trunk 上分别读出 `value / danger / attack`，形成复盘 MVP | `nn/` + `docs/review-system.md` |
| **P5** | 真理引擎 + 滑动窗口 + 模型头，收敛成稳定复盘系统 | 复盘编排层 |
| **P6** | 把复盘中验证有效的头逐步转成搜索收益 | `crates/engin` |
| **P7** | 蒸馏与动态搜索 | 数据 / 训练 / 引擎联动 |

## 依赖顺序

```text
P0 -> P1 -> P2 -> P4 -> P5 -> P6 -> P7
 \-----------------> P3 -----^
```

说明：

- `P3` 仍然重要，但当前不是短期主阻塞项
- 当前短期最重要的是 `P4` 与 `P5`

## 当前产品节奏

1. 先做强 `policy + trunk`
2. 再做 `value`
3. 再做 `danger`
4. 再做 `attack`
5. 再把它们与真理引擎、滑动窗口整合
6. 最后才做搜索收益验证
