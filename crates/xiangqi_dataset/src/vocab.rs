//! 词表 JSON（`nn/scripts/vocab/build_vocab.py`：`{ "moves": [...] }`，可选 `size` 等冗余字段）与 SHA256 指纹。

use anyhow::{Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct VocabFile {
    pub moves: Vec<String>,
}

/// 解析 JSON 字符串（测试 / 内存调用）。
pub fn load_vocab_json_str(text: &str) -> Result<(HashMap<String, i32>, [u8; 32])> {
    let v: VocabFile =
        serde_json::from_str(text).context("词表 JSON 解析失败（需要 moves 数组）")?;
    Ok(hash_vocab_moves(v.moves))
}

/// 加载词表并计算与 Python `vocab_fingerprint_ordered_moves` 一致的 SHA-256（32 字节）。
pub fn load_vocab(path: &Path) -> Result<(HashMap<String, i32>, [u8; 32])> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("读取词表 {}", path.display()))?;
    load_vocab_json_str(&text)
}

fn hash_vocab_moves(moves: Vec<String>) -> (HashMap<String, i32>, [u8; 32]) {
    let mut hasher = Sha256::new();
    for m in &moves {
        hasher.update(m.as_bytes());
        hasher.update(b"\0");
    }
    let hash: [u8; 32] = hasher.finalize().into();
    let map: HashMap<String, i32> = moves
        .into_iter()
        .enumerate()
        .map(|(i, m)| (m, i as i32))
        .collect();
    (map, hash)
}

pub fn vocab_sha256_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_hash_stable() {
        let j = r#"{"moves":["a1b2","h2h3"]}"#;
        let (_, h1) = load_vocab_json_str(j).unwrap();
        let (_, h2) = load_vocab_json_str(j).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(vocab_sha256_hex(&h1).len(), 64);
    }
}
