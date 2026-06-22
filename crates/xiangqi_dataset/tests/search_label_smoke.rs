use std::fs;

use tempfile::tempdir;

use xiangqi_core::{legal_moves_uci, parse_move_uci, Position, START_FEN};
use xiangqi_dataset::{
    collect_vocab_moves_from_pgn, export_search_label_shard_from_pgn, read_shard_header, write_vocab_json,
    SearchLabelExportConfig,
};

#[test]
fn smoke_search_labels_into_xrsh() {
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

    let out_dir = dir.path().join("search_xrsh");
    let rows = export_search_label_shard_from_pgn(
        &pgn_path,
        &vocab_path,
        &out_dir,
        None,
        &SearchLabelExportConfig {
            max_playouts: 32,
            max_games: 0,
            max_rows_per_game: 0,
            cpuct: 1.25,
        },
        None,
    )
    .expect("export");

    assert!(rows >= 1);
    let shard = out_dir.join("shard_00000.xrsh");
    assert!(shard.is_file());
    let (ver, _, _) = read_shard_header(&shard).expect("header");
    assert_eq!(ver, 5);
}

#[test]
fn search_label_rejects_partial_vocab_early() {
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
    let pgn_path = dir.path().join("smoke_bad_vocab.pgn");
    fs::write(&pgn_path, &pgn_text).expect("pgn");

    let vocab_path = dir.path().join("vocab_bad.json");
    write_vocab_json(&vocab_path, &[m0.clone()]).expect("write vocab");

    let out_dir = dir.path().join("search_xrsh_bad_vocab");
    let err = export_search_label_shard_from_pgn(
        &pgn_path,
        &vocab_path,
        &out_dir,
        None,
        &SearchLabelExportConfig::default(),
        None,
    )
    .expect_err("partial vocab should fail");

    assert!(err.to_string().contains("完整 canonical vocab"));
}
