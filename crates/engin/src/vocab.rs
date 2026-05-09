//! `move_vocab.json`：`{ "moves": [ "a0a1", ... ], "size": N }`（与 `build_vocab.py` 一致）。

use std::collections::HashMap;
use std::path::Path;

fn moves_array_from_json(path: &Path) -> Result<Vec<String>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("读取词表失败: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| format!("词表 JSON 无效: {e}"))?;
    let arr = v
        .get("moves")
        .and_then(|x| x.as_array())
        .ok_or_else(|| "词表缺少 moves 数组".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let s = item
            .as_str()
            .ok_or_else(|| format!("moves[{i}] 不是字符串"))?
            .to_string();
        out.push(s);
    }
    Ok(out)
}

/// 按下标 0..V-1 与 `logits` 对齐的着法表。
pub fn load_move_vocab_ordered(path: &Path) -> Result<Vec<String>, String> {
    moves_array_from_json(path)
}

/// 返回 `(move → 词表下标, moves.len())`；下标与 ONNX `logits` 维一致。
pub fn load_move_vocab(path: &Path) -> Result<(HashMap<String, usize>, usize), String> {
    let ordered = moves_array_from_json(path)?;
    let n = ordered.len();
    let mut m = HashMap::with_capacity(n);
    for (i, s) in ordered.into_iter().enumerate() {
        m.insert(s, i);
    }
    Ok((m, n))
}
