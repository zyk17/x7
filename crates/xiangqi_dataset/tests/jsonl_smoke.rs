//! 冒烟：由 JSONL 生成 XRSH 分片，校验头与 pack_meta。

use serde_json::json;
use std::fs;
use tempfile::tempdir;

use xiangqi_core::{legal_moves_uci, Position, START_FEN};
use xiangqi_dataset::{
    load_vocab_json_str, read_shard_header, run_jsonl_shards, vocab_sha256_hex,
};

#[test]
fn smoke_jsonl_generates_one_shard() {
    let dir = tempdir().expect("tempdir");
    let pos = Position::from_fen(START_FEN).expect("startpos");
    let mut vmoves = legal_moves_uci(&pos);
    vmoves.sort();

    let vocab_path = dir.path().join("vocab.json");
    fs::write(
        &vocab_path,
        serde_json::to_string(&json!({ "moves": vmoves })).expect("json"),
    )
    .expect("write vocab");

    let first = vmoves[0].clone();
    let line = serde_json::to_string(&json!({
        "fen": pos.fen(),
        "root_fen": START_FEN,
        "uci_prefix": [],
        "human_move_pyffish": first,
        "ply": 0,
        "game_id": "smoke_game_0",
    }))
    .expect("line");

    let jl = dir.path().join("smoke.jsonl");
    fs::write(&jl, format!("{line}\n")).expect("jsonl");

    let out = dir.path().join("out_xrsh");
    let n = run_jsonl_shards(&jl, &vocab_path, &out, 2, 500).expect("pipeline");
    assert_eq!(n, 1, "应写出 1 个 shard");

    let shard = out.join("shard_00000.xrsh");
    assert!(shard.is_file(), "缺少 shard 文件");

    let (ver, file_hash, n_games) = read_shard_header(&shard).expect("header");
    assert_eq!(ver, 2);
    assert_eq!(n_games, 1);

    let vocab_txt = fs::read_to_string(&vocab_path).expect("read vocab");
    let (_, exp_hash) = load_vocab_json_str(&vocab_txt).expect("vocab hash");
    assert_eq!(file_hash, exp_hash);

    let meta = fs::read_to_string(out.join("pack_meta.json")).expect("pack_meta");
    let hex = vocab_sha256_hex(&exp_hash);
    assert!(meta.contains(&hex), "pack_meta 应含完整 vocab_sha256");
}
