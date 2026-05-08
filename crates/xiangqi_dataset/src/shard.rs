//! 二进制分片 `XQB` v1：按局聚合，便于按 `game_id` 并行生成。

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub const FORMAT_NAME: &str = "xqb_v1";

/// 单样本（一行训练数据）。
#[derive(Debug, Clone)]
pub struct EncodedRow {
    pub fen: String,
    pub root_fen: String,
    pub uci_prefix: Vec<String>,
    pub target_idx: i32,
    pub legal_idx: Vec<i32>,
    pub ply: u16,
}

/// 一局内的所有样本。
#[derive(Debug, Clone)]
pub struct EncodedGame {
    pub game_id: String,
    pub rows: Vec<EncodedRow>,
}

fn write_str_u16(w: &mut impl Write, s: &str) -> Result<()> {
    let b = s.as_bytes();
    if b.len() > u16::MAX as usize {
        bail!("字符串过长 (>65535)：{}", s.chars().take(40).collect::<String>());
    }
    w.write_all(&(b.len() as u16).to_le_bytes())?;
    w.write_all(b)?;
    Ok(())
}

fn write_prefix_list(w: &mut impl Write, pfx: &[String]) -> Result<()> {
    if pfx.len() > u16::MAX as usize {
        bail!("uci_prefix 过长");
    }
    w.write_all(&(pfx.len() as u16).to_le_bytes())?;
    for s in pfx {
        let b = s.as_bytes();
        if b.len() > u8::MAX as usize {
            bail!("单步 UCI 过长 (>255)");
        }
        w.write_all(&[b.len() as u8])?;
        w.write_all(b)?;
    }
    Ok(())
}

/// 写入单个分片文件：`shard_{index:05}.xqb`
pub fn write_shard(path: &Path, vocab_hash: &[u8; 32], games: &[EncodedGame]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| parent.display().to_string())?;
    }
    let f = File::create(path).with_context(|| path.display().to_string())?;
    let mut w = BufWriter::new(f);

    // 头 64 字节
    w.write_all(b"XQB\0")?;
    w.write_all(&1u32.to_le_bytes())?;
    w.write_all(vocab_hash)?;
    w.write_all(&(games.len() as u32).to_le_bytes())?;
    w.write_all(&[0u8; 24])?; // 保留

    for g in games {
        write_str_u16(&mut w, &g.game_id)?;
        if g.rows.len() > u32::MAX as usize {
            bail!("单局样本数溢出");
        }
        w.write_all(&(g.rows.len() as u32).to_le_bytes())?;
        for r in &g.rows {
            write_str_u16(&mut w, &r.fen)?;
            write_str_u16(&mut w, &r.root_fen)?;
            write_prefix_list(&mut w, &r.uci_prefix)?;
            w.write_all(&r.target_idx.to_le_bytes())?;
            if r.legal_idx.len() > u16::MAX as usize {
                bail!("合法着数量溢出");
            }
            w.write_all(&(r.legal_idx.len() as u16).to_le_bytes())?;
            for &idx in &r.legal_idx {
                w.write_all(&idx.to_le_bytes())?;
            }
            w.write_all(&r.ply.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

/// 写出 `pack_meta.json`（与 Python 侧 `pack_meta.json` 字段风格接近）。
pub fn write_pack_meta(
    out_dir: &Path,
    vocab_hash: &[u8; 32],
    shard_count: usize,
    source_note: &str,
) -> Result<()> {
    let hex: String = vocab_hash.iter().map(|b| format!("{b:02x}")).collect();
    let meta = serde_json::json!({
        "format": FORMAT_NAME,
        "format_version": 1,
        "vocab_sha256": hex,
        "shard_count": shard_count,
        "source": source_note,
    });
    let p = out_dir.join("pack_meta.json");
    std::fs::write(&p, serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}
