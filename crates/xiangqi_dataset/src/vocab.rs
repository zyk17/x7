//! 词表 JSON（`build_vocab.py` 输出：`{ "moves": [...] }`）与 SHA256 指纹。

use anyhow::{Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct VocabFile {
    pub moves: Vec<String>,
    #[serde(default)]
    pub size: usize,
}

/// 加载词表并计算与 Python `vocab_fingerprint_ordered_moves` 一致的 SHA-256（32 字节）。
pub fn load_vocab(path: &Path) -> Result<(HashMap<String, i32>, [u8; 32])> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("读取词表 {}", path.display()))?;
    let v: VocabFile =
        serde_json::from_str(&text).context("词表 JSON 解析失败（需要 moves 数组）")?;
    let moves = v.moves;
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
    Ok((map, hash))
}

pub fn vocab_sha256_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}
