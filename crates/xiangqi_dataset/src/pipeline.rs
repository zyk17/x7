//! 从 PGN 生成分片（供 CLI 与测试复用）。`--jobs 0` 表示使用
//! [`std::thread::available_parallelism`] 作为线程数。

use crate::encode::encode_game;
use crate::pgn::read_pgn_games;
use crate::shard::{write_pack_meta, write_shard};
use crate::vocab::load_vocab;
use crate::EncodedGame;
use anyhow::{Context, Result};
use rayon::prelude::*;
use rayon::ThreadPool;
use std::fs;
use std::path::Path;

/// `--jobs 0` → 使用机器默认并行度；否则至少 1 个工作线程。
fn build_rayon_pool(jobs: usize) -> Result<(ThreadPool, usize)> {
    let n = if jobs == 0 {
        std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1)
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

pub fn run_pgn_shards(
    pgn_path: &Path,
    vocab_path: &Path,
    out_dir: &Path,
    jobs: usize,
    games_per_shard: usize,
    max_games: usize,
) -> Result<usize> {
    let (vocab, vocab_hash) = load_vocab(vocab_path)?;
    let stem = pgn_path.file_stem().and_then(|s| s.to_str()).unwrap_or("pgn");
    let raw = fs::read_to_string(pgn_path).with_context(|| format!("读取 {}", pgn_path.display()))?;
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

    write_shards_to_dir(out_dir, &vocab_hash, &encoded, games_per_shard, &format!("pgn:{stem}"))
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
