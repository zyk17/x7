//! JSONL：`jobs=1` 与多线程下分片输出字节一致（写出前按 `game_id` 排序）。

use serde_json::json;
use std::fs;
use std::path::Path;

use tempfile::tempdir;

use xiangqi_core::{legal_moves_uci, Position, START_FEN};
use xiangqi_dataset::run_jsonl_shards;

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
fn jsonl_jobs_serial_equals_parallel_shards() {
    let dir = tempdir().expect("tempdir");
    let pos = Position::from_fen(START_FEN).expect("startpos");
    let mut vmoves = legal_moves_uci(&pos);
    vmoves.sort();
    let first = vmoves[0].clone();

    let vocab_path = dir.path().join("vocab.json");
    fs::write(
        &vocab_path,
        serde_json::to_string(&json!({ "moves": vmoves })).expect("json"),
    )
    .expect("write vocab");

    let line = |game_id: &str| {
        serde_json::to_string(&json!({
            "fen": pos.fen(),
            "root_fen": START_FEN,
            "uci_prefix": [],
            "human_move_pyffish": first,
            "ply": 0,
            "game_id": game_id,
        }))
        .expect("line")
    };

    // 文件中局顺序故意打乱；排序后应为 game_a, game_m, game_z
    let jl = dir.path().join("many.jsonl");
    fs::write(
        &jl,
        format!("{}\n{}\n{}\n", line("game_z"), line("game_a"), line("game_m")),
    )
    .expect("jsonl");

    let out1 = dir.path().join("out_j1");
    let out8 = dir.path().join("out_j8");
    let n1 = run_jsonl_shards(&jl, &vocab_path, &out1, 1, 500).expect("run 1");
    let n8 = run_jsonl_shards(&jl, &vocab_path, &out8, 8, 500).expect("run 8");
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
