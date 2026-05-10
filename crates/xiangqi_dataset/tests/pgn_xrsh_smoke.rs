//! 冒烟：PGN → 词表 → XRSH，校验头与 pack_meta。

use std::fs;
use tempfile::tempdir;

use xiangqi_core::{legal_moves_uci, parse_move_uci, Position, START_FEN};
use xiangqi_dataset::{
    collect_vocab_moves_from_pgn, load_vocab_json_str, read_shard_header, run_pgn_shards, vocab_sha256_hex,
    write_vocab_json,
};

#[test]
fn smoke_pgn_generates_one_shard() {
    let dir = tempdir().expect("tempdir");
    let mut pos = Position::from_fen(START_FEN).expect("startpos");
    let mut legals0 = legal_moves_uci(&pos);
    legals0.sort();
    let m0 = legals0[0].clone();
    pos.do_move(parse_move_uci(&m0).expect("m0"));
    let mut legals1 = legal_moves_uci(&pos);
    legals1.sort();
    let m1 = legals1[0].clone();

    let pgn_text = format!("[Event \"smoke\"]\n[Result \"1/2-1/2\"]\n\n1. {m0} {m1}\n");
    let pgn_path = dir.path().join("smoke.pgn");
    fs::write(&pgn_path, &pgn_text).expect("pgn");

    let moves = collect_vocab_moves_from_pgn(&pgn_text, 0).expect("vocab");
    let vocab_path = dir.path().join("vocab.json");
    write_vocab_json(&vocab_path, &moves).expect("write vocab");

    let out = dir.path().join("out_xrsh");
    let n = run_pgn_shards(&pgn_path, &vocab_path, &out, 2, 500, 0).expect("pipeline");
    assert_eq!(n, 1, "应写出 1 个 shard");

    let shard = out.join("shard_00000.xrsh");
    assert!(shard.is_file(), "缺少 shard 文件");

    let (ver, file_hash, n_games) = read_shard_header(&shard).expect("header");
    assert_eq!(ver, 3);
    assert_eq!(n_games, 1);

    let vocab_txt = fs::read_to_string(&vocab_path).expect("read vocab");
    let (_, exp_hash) = load_vocab_json_str(&vocab_txt).expect("vocab hash");
    assert_eq!(file_hash, exp_hash);

    let meta = fs::read_to_string(out.join("pack_meta.json")).expect("pack_meta");
    let hex = vocab_sha256_hex(&exp_hash);
    assert!(meta.contains(&hex), "pack_meta 应含完整 vocab_sha256");
    assert!(meta.contains("xrsh_v3"), "pack_meta 应为 xrsh_v3");
}
