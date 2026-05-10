//! PGN：`jobs=1` 与多线程下分片输出字节一致（写出前按 `game_id` 排序）。

use std::fs;
use std::path::Path;

use tempfile::tempdir;

use xiangqi_core::{legal_moves_uci, Position, START_FEN};
use xiangqi_dataset::{collect_vocab_moves_from_pgn, run_pgn_shards, write_vocab_json};

fn list_shard_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = fs::read_dir(dir)
        .expect("read_dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with("shard_") && n.ends_with(".xrsh"))
                .unwrap_or(false)
        })
        .collect();
    v.sort();
    v
}

#[test]
fn pgn_jobs_serial_equals_parallel_shards() {
    let dir = tempdir().expect("tempdir");
    let pos = Position::from_fen(START_FEN).expect("startpos");
    let mut legals = legal_moves_uci(&pos);
    legals.sort();
    let first = legals[0].clone();

    let pgn_body =
        format!("[Event \"a\"]\n\n1. {first}\n\n[Event \"b\"]\n\n1. {first}\n\n[Event \"c\"]\n\n1. {first}\n");
    let pgn_path = dir.path().join("many.pgn");
    fs::write(&pgn_path, &pgn_body).expect("pgn");

    let moves = collect_vocab_moves_from_pgn(&pgn_body, 0).expect("vocab");
    let vocab_path = dir.path().join("vocab.json");
    write_vocab_json(&vocab_path, &moves).expect("write vocab");

    let out1 = dir.path().join("out_j1");
    let out8 = dir.path().join("out_j8");
    let n1 = run_pgn_shards(&pgn_path, &vocab_path, &out1, 1, 500, 0).expect("run 1");
    let n8 = run_pgn_shards(&pgn_path, &vocab_path, &out8, 8, 500, 0).expect("run 8");
    assert_eq!(n1, n8);
    assert_eq!(n1, 1, "3 局 games_per_shard=500 → 1 个 shard");

    let s1 = list_shard_files(&out1);
    let s8 = list_shard_files(&out8);
    assert_eq!(s1.len(), s8.len());
    for (pa, pb) in s1.iter().zip(s8.iter()) {
        let ba = fs::read(pa).expect("read");
        let bb = fs::read(pb).expect("read");
        assert_eq!(ba, bb, "shard 字节须完全一致（并行 vs 串行池）");
    }

    let meta1 = fs::read_to_string(out1.join("pack_meta.json")).expect("meta");
    let meta2 = fs::read_to_string(out8.join("pack_meta.json")).expect("meta");
    assert_eq!(meta1, meta2);
}
