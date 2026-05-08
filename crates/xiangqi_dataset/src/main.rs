//! 维护者 CLI：`pgn-shards` / `jsonl-shards` → `XQB` 二进制分片。

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use xiangqi_dataset::{
    encode_game, encode_jsonl_line, load_vocab, read_pgn_games, write_pack_meta, write_shard,
    EncodedGame, EncodedRow,
};

#[derive(Parser)]
#[command(name = "xiangqi_dataset")]
#[command(about = "PGN / JSONL → 二进制 shards（维护者工具）")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 从 PGN / `.pgns` 生成 shards（按局并行）
    PgnShards {
        #[arg(long, required = true)]
        pgn: PathBuf,
        #[arg(long, required = true)]
        vocab: PathBuf,
        #[arg(long, required = true)]
        out_dir: PathBuf,
        /// Rayon 线程数，0 = 机器默认
        #[arg(long, default_value_t = 0usize)]
        jobs: usize,
        /// 每个 `.xqb` 文件包含的最大对局数
        #[arg(long, default_value_t = 500usize)]
        games_per_shard: usize,
        #[arg(long, default_value_t = 0usize)]
        max_games: usize,
    },
    /// 从已有 JSONL（extract_rows 格式）生成 shards
    JsonlShards {
        #[arg(long, required = true)]
        jsonl: PathBuf,
        #[arg(long, required = true)]
        vocab: PathBuf,
        #[arg(long, required = true)]
        out_dir: PathBuf,
        #[arg(long, default_value_t = 0usize)]
        jobs: usize,
        #[arg(long, default_value_t = 500usize)]
        games_per_shard: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::PgnShards {
            pgn,
            vocab,
            out_dir,
            jobs,
            games_per_shard,
            max_games,
        } => run_pgn_shards(&pgn, &vocab, &out_dir, jobs, games_per_shard, max_games),
        Cmd::JsonlShards {
            jsonl,
            vocab,
            out_dir,
            jobs,
            games_per_shard,
        } => run_jsonl_shards(&jsonl, &vocab, &out_dir, jobs, games_per_shard),
    }
}

fn run_pgn_shards(
    pgn_path: &Path,
    vocab_path: &Path,
    out_dir: &Path,
    jobs: usize,
    games_per_shard: usize,
    max_games: usize,
) -> Result<()> {
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

    let pool = if jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .context("创建线程池")?
    } else {
        rayon::ThreadPoolBuilder::new().build().context("创建线程池")?
    };

    let encoded: Vec<EncodedGame> = pool.install(|| {
        games
            .par_iter()
            .enumerate()
            .filter_map(|(gi, g)| {
                let game_id = format!("{stem}_{gi:06}");
                encode_game(g, &game_id, &vocab).ok().flatten()
            })
            .collect()
    });

    write_shards(out_dir, &vocab_hash, &encoded, games_per_shard, &format!("pgn:{stem}"))
}

fn run_jsonl_shards(
    jsonl_path: &Path,
    vocab_path: &Path,
    out_dir: &Path,
    jobs: usize,
    games_per_shard: usize,
) -> Result<()> {
    let (vocab, vocab_hash) = load_vocab(vocab_path)?;
    let raw = fs::read_to_string(jsonl_path)
        .with_context(|| format!("读取 {}", jsonl_path.display()))?;
    let lines: Vec<String> = raw.lines().map(|s| s.to_string()).collect();

    let pool = if jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .context("创建线程池")?
    } else {
        rayon::ThreadPoolBuilder::new().build().context("创建线程池")?
    };

    // 并行编码每一行，再按 game_id 聚合
    let rows: Vec<(String, u16, EncodedRow)> = pool.install(|| {
        lines
            .par_iter()
            .filter_map(|line| {
                let v: serde_json::Value = serde_json::from_str(line).ok()?;
                let gid = v["game_id"].as_str()?.to_string();
                let ply = v["ply"].as_u64()? as u16;
                let row = encode_jsonl_line(line, &vocab).ok()??;
                Some((gid, ply, row))
            })
            .collect()
    });

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

    let stem = jsonl_path.file_stem().and_then(|s| s.to_str()).unwrap_or("jsonl");
    write_shards(
        out_dir,
        &vocab_hash,
        &encoded,
        games_per_shard,
        &format!("jsonl:{stem}"),
    )
}

fn write_shards(
    out_dir: &Path,
    vocab_hash: &[u8; 32],
    games: &[EncodedGame],
    games_per_shard: usize,
    source_note: &str,
) -> Result<()> {
    fs::create_dir_all(out_dir).with_context(|| out_dir.display().to_string())?;
    if games.is_empty() {
        eprintln!("警告：无有效对局，未写入 shard");
        write_pack_meta(out_dir, vocab_hash, 0, source_note)?;
        return Ok(());
    }
    let mut shard_idx = 0usize;
    for chunk in games.chunks(games_per_shard.max(1)) {
        let path = out_dir.join(format!("shard_{shard_idx:05}.xqb"));
        write_shard(&path, vocab_hash, chunk)?;
        shard_idx += 1;
    }
    write_pack_meta(out_dir, vocab_hash, shard_idx, source_note)?;
    eprintln!("写入 {shard_idx} 个分片 -> {}", out_dir.display());
    Ok(())
}
