//! 维护者 CLI：`vocab-enum` / `pgn-shards` → XRSH（`.xrsh`）二进制分片。

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use xiangqi_dataset::{
    collect_vocab_moves_from_pgn_with_jobs,
    enumerate_canonical_vocab_moves,
    run_pgn_shards,
    write_vocab_json,
};

#[derive(Parser)]
#[command(name = "xiangqi_dataset")]
#[command(about = "PGN → 词表 / 二进制 shards（维护者工具）")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 直接按象棋几何规则枚举 canonical `move_vocab.json`（不依赖语料）
    VocabEnum {
        #[arg(long, required = true)]
        out: PathBuf,
    },
    /// 调试/校验：从 PGN / `.pgns` 扫描棋谱路径上的合法着并集，写出 `move_vocab.json`
    VocabFromPgn {
        #[arg(long, required = true)]
        pgn: PathBuf,
        #[arg(long, required = true)]
        out: PathBuf,
        /// Rayon 并行线程数：0 = 本机可用并行度（按**局**并行扫描）
        #[arg(long, default_value_t = 0usize)]
        jobs: usize,
        /// 最多处理前 N 局；0 = 全部
        #[arg(long, default_value_t = 0usize)]
        max_games: usize,
    },
    /// 从 PGN / `.pgns` 生成 shards（按局并行）
    PgnShards {
        #[arg(long, required = true)]
        pgn: PathBuf,
        #[arg(long, required = true)]
        vocab: PathBuf,
        #[arg(long, required = true)]
        out_dir: PathBuf,
        /// Rayon 并行线程数：0 = 本机可用并行度（按**局**并行编码）
        #[arg(long, default_value_t = 0usize)]
        jobs: usize,
        #[arg(long, default_value_t = 500usize)]
        games_per_shard: usize,
        #[arg(long, default_value_t = 0usize)]
        max_games: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::VocabEnum { out } => {
            let moves = enumerate_canonical_vocab_moves();
            write_vocab_json(&out, &moves)?;
            eprintln!("wrote {} canonical moves -> {}", moves.len(), out.display());
        }
        Cmd::VocabFromPgn {
            pgn,
            out,
            jobs,
            max_games,
        } => {
            let raw = std::fs::read_to_string(&pgn).with_context(|| format!("读取 {}", pgn.display()))?;
            let moves = collect_vocab_moves_from_pgn_with_jobs(&raw, max_games, jobs)?;
            write_vocab_json(&out, &moves)?;
            eprintln!("wrote {} moves -> {}", moves.len(), out.display());
        }
        Cmd::PgnShards {
            pgn,
            vocab,
            out_dir,
            jobs,
            games_per_shard,
            max_games,
        } => {
            let n = run_pgn_shards(&pgn, &vocab, &out_dir, jobs, games_per_shard, max_games)?;
            eprintln!("写入 {n} 个分片 -> {}", out_dir.display());
        }
    }
    Ok(())
}
