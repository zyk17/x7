//! 维护者 CLI：`pgn-shards` / `jsonl-shards` → XRSH（`.xrsh`）二进制分片。

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use xiangqi_dataset::{run_jsonl_shards, run_pgn_shards};

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
        /// Rayon 并行线程数：0 = 本机可用并行度（按**局**并行编码）
        #[arg(long, default_value_t = 0usize)]
        jobs: usize,
        #[arg(long, default_value_t = 500usize)]
        games_per_shard: usize,
        #[arg(long, default_value_t = 0usize)]
        max_games: usize,
    },
    /// 从已有训练 JSONL（标准字段见 `encode_jsonl_line`）生成 shards
    JsonlShards {
        #[arg(long, required = true)]
        jsonl: PathBuf,
        #[arg(long, required = true)]
        vocab: PathBuf,
        #[arg(long, required = true)]
        out_dir: PathBuf,
        /// Rayon 并行线程数：0 = 本机可用并行度（按**行**并行解析 / 编码，适合超大 JSONL）
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
        } => {
            let n = run_pgn_shards(&pgn, &vocab, &out_dir, jobs, games_per_shard, max_games)?;
            eprintln!("写入 {n} 个分片 -> {}", out_dir.display());
        }
        Cmd::JsonlShards {
            jsonl,
            vocab,
            out_dir,
            jobs,
            games_per_shard,
        } => {
            let n = run_jsonl_shards(&jsonl, &vocab, &out_dir, jobs, games_per_shard)?;
            eprintln!("写入 {n} 个分片 -> {}", out_dir.display());
        }
    }
    Ok(())
}
