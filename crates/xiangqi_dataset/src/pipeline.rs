//! 从 PGN / JSONL 生成分片（供 CLI 与测试复用）。
//!
//! JSONL 路径：**按行并行**解析与编码（Rayon）；`--jobs 0` 表示使用
//! [`std::thread::available_parallelism`] 作为线程数。

use crate::encode::{encode_game, encode_jsonl_line};
use crate::pgn::read_pgn_games;
use crate::shard::{write_pack_meta, write_shard};
use crate::vocab::load_vocab;
use crate::{EncodedGame, EncodedRow};
use anyhow::{Context, Result};
use rayon::prelude::*;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// `--jobs 0` → 使用机器默认并行度；否则至少 1 个工作线程。
fn build_rayon_pool(jobs: usize) -> Result<(ThreadPool, usize)> {
    let n = if jobs == 0 {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
    } else {
        jobs.max(1)
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .context("创建 Rayon 线程池")?;
    Ok((pool, n))
}

/// 写出前固定局顺序，保证 **`jobs` 不同也能得到相同分片字节**（可复现）。
fn sort_games_for_deterministic_shards(encoded: &mut [EncodedGame]) {
    encoded.sort_by(|a, b| a.game_id.cmp(&b.game_id));
}

/// 与 `main` 中 `jsonl-shards` 一致。
pub fn run_jsonl_shards(
    jsonl_path: &Path,
    vocab_path: &Path,
    out_dir: &Path,
    jobs: usize,
    games_per_shard: usize,
) -> Result<usize> {
    eprintln!("jsonl-shards: 读取 {} ...", jsonl_path.display());
    let byte_len = fs::metadata(jsonl_path)
        .with_context(|| format!("stat {}", jsonl_path.display()))?
        .len();
    eprintln!(
        "  文件约 {:.2} GiB，将整文件载入内存后按行并行编码（超大文件请先切片或保证内存）",
        byte_len as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let (vocab, vocab_hash) = load_vocab(vocab_path)?;
    let raw = fs::read_to_string(jsonl_path)
        .with_context(|| format!("读取 {}", jsonl_path.display()))?;
    let lines: Vec<String> = raw.lines().map(|s| s.to_string()).collect();
    eprintln!("  非空行数: {}，开始 Rayon 编码 …", lines.len());

    let (pool, n_threads) = build_rayon_pool(jobs)?;
    let min_len = if lines.is_empty() {
        1usize
    } else {
        (lines.len() / n_threads.saturating_mul(4)).max(1)
    };

    let rows: Vec<(String, u16, EncodedRow)> = pool.install(|| {
        lines
            .par_iter()
            .with_min_len(min_len)
            .filter_map(|line| {
                let v: serde_json::Value = serde_json::from_str(line).ok()?;
                let gid = v["game_id"].as_str()?.to_string();
                let ply = v["ply"].as_u64()? as u16;
                let row = encode_jsonl_line(line, &vocab).ok()??;
                Some((gid, ply, row))
            })
            .collect()
    });

    eprintln!("  编码完成，有效样本 {} 条（按 game_id 聚合）…", rows.len());

    let mut by_game: HashMap<String, Vec<(u16, EncodedRow)>> = HashMap::new();
    for (gid, ply, row) in rows {
        by_game.entry(gid).or_default().push((ply, row));
    }
    let mut encoded: Vec<EncodedGame> = Vec::new();
    for (game_id, mut v) in by_game {
        v.sort_by_key(|x| x.0);
        encoded.push(EncodedGame {
            game_id,
            rows: v.into_iter().map(|x| x.1).collect(),
        });
    }
    sort_games_for_deterministic_shards(&mut encoded);

    eprintln!("  聚合局数: {}，写入分片 …", encoded.len());

    let stem = jsonl_path.file_stem().and_then(|s| s.to_str()).unwrap_or("jsonl");
    write_shards_to_dir(
        out_dir,
        &vocab_hash,
        &encoded,
        games_per_shard,
        &format!("jsonl:{stem}"),
    )
}

/// 与 `main` 中 `pgn-shards` 一致。
pub fn run_pgn_shards(
    pgn_path: &Path,
    vocab_path: &Path,
    out_dir: &Path,
    jobs: usize,
    games_per_shard: usize,
    max_games: usize,
) -> Result<usize> {
    let (vocab, vocab_hash) = load_vocab(vocab_path)?;
    let stem = pgn_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("pgn");
    let raw = fs::read_to_string(pgn_path)
        .with_context(|| format!("读取 {}", pgn_path.display()))?;
    let mut games = read_pgn_games(&raw);
    if max_games > 0 {
        games.truncate(max_games);
    }

    let (pool, _) = build_rayon_pool(jobs)?;

    let mut encoded: Vec<EncodedGame> = pool.install(|| {
        games
            .par_iter()
            .enumerate()
            .filter_map(|(gi, g)| {
                let game_id = format!("{stem}_{gi:06}");
                encode_game(g, &game_id, &vocab).ok().flatten()
            })
            .collect()
    });
    sort_games_for_deterministic_shards(&mut encoded);

    write_shards_to_dir(
        out_dir,
        &vocab_hash,
        &encoded,
        games_per_shard,
        &format!("pgn:{stem}"),
    )
}

/// 写入 `shard_*.xrsh` 与 `pack_meta.json`，返回 **分片文件个数**。
pub fn write_shards_to_dir(
    out_dir: &Path,
    vocab_hash: &[u8; 32],
    games: &[EncodedGame],
    games_per_shard: usize,
    source_note: &str,
) -> Result<usize> {
    fs::create_dir_all(out_dir).with_context(|| out_dir.display().to_string())?;
    if games.is_empty() {
        write_pack_meta(out_dir, vocab_hash, 0, source_note)?;
        return Ok(0);
    }
    let mut shard_idx = 0usize;
    for chunk in games.chunks(games_per_shard.max(1)) {
        let path = out_dir.join(format!("shard_{shard_idx:05}.xrsh"));
        write_shard(&path, vocab_hash, chunk)?;
        shard_idx += 1;
    }
    write_pack_meta(out_dir, vocab_hash, shard_idx, source_note)?;
    Ok(shard_idx)
}
