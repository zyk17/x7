//! 二进制分片 **XRSH**（Xiangqi Review Shard）v1：按局聚合，便于按 `game_id` 并行生成。
//! 扩展名 `.xrsh`，与市面其它 `.xqb` 棋谱格式区分。

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

/// 当前写入格式：`pack_meta.json` 的 `format` 字段。
pub const FORMAT_NAME: &str = "xrsh_v3";

/// 分片文件头内的二进制版本号（与 `pack_meta.format_version` 一致）。
pub const SHARD_FILE_VERSION: u32 = 3;

/// 单样本（一行训练数据）。
#[derive(Debug, Clone)]
pub struct EncodedRow {
    pub fen: String,
    pub root_fen: String,
    pub uci_prefix: Vec<String>,
    pub target_idx: i32,
    pub legal_idx: Vec<i32>,
    pub ply: u16,
    /// 由 `xiangqi_core` 预计算，与 `nn.aux_pseudo_labels` 数值对齐。
    pub aux_attack: f32,
    pub aux_danger: f32,
    pub aux_tactical: f32,
    /// PGN `[Result]` 红方视角：`1` 红胜、`-1` 黑胜、`0` 和、`2` 未知（`*` 或未标注）。
    pub game_result_red: i8,
    /// 本局总着数（与 `uci_moves.len()` 一致），用于 value 时间折扣等。
    pub ply_total: u16,
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

/// 写入单个分片文件：`shard_{index:05}.xrsh`
pub fn write_shard(path: &Path, vocab_hash: &[u8; 32], games: &[EncodedGame]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| parent.display().to_string())?;
    }
    let f = File::create(path).with_context(|| path.display().to_string())?;
    let mut w = BufWriter::new(f);

    // 头 64 字节（魔数 `"XRSH"`）
    w.write_all(b"XRSH")?;
    w.write_all(&SHARD_FILE_VERSION.to_le_bytes())?;
    w.write_all(vocab_hash)?;
    w.write_all(&(games.len() as u32).to_le_bytes())?;
    w.write_all(&[0u8; 20])?; // 保留（共 64 字节头）

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
            w.write_all(&r.aux_attack.to_le_bytes())?;
            w.write_all(&r.aux_danger.to_le_bytes())?;
            w.write_all(&r.aux_tactical.to_le_bytes())?;
            w.write_all(&r.game_result_red.to_le_bytes())?;
            w.write_all(&r.ply_total.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

/// 写出 `pack_meta.json`（与 Python 侧 `pack_meta.json` 字段风格接近）。
pub fn write_pack_meta(out_dir: &Path, vocab_hash: &[u8; 32], shard_count: usize, source_note: &str) -> Result<()> {
    let hex: String = vocab_hash.iter().map(|b| format!("{b:02x}")).collect();
    let meta = serde_json::json!({
        "format": FORMAT_NAME,
        "format_version": SHARD_FILE_VERSION,
        "vocab_sha256": hex,
        "shard_count": shard_count,
        "source": source_note,
    });
    let p = out_dir.join("pack_meta.json");
    std::fs::write(&p, serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

/// 校验文件头（魔数、格式版本、词表哈希、本文件内对局数）。
pub fn read_shard_header(path: &Path) -> Result<(u32, [u8; 32], u32)> {
    let mut f = File::open(path).with_context(|| path.display().to_string())?;
    let mut buf = [0u8; 64];
    f.read_exact(&mut buf)
        .with_context(|| format!("读取头 64 字节 {}", path.display()))?;
    if buf[0..4] != *b"XRSH" {
        bail!("非 XRSH review shard 文件: {}", path.display());
    }
    let ver = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&buf[8..40]);
    let n_games = u32::from_le_bytes(buf[40..44].try_into().unwrap());
    Ok((ver, hash, n_games))
}
