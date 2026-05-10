//! 训练侧数据：**PGN → 二进制分片**（**XRSH** v2，`.xrsh`；兼容读取旧 v1 由 Python 侧实现）。
//! 按局并行（`pipeline::run_pgn_shards`）。

pub mod aux_labels;
pub use aux_labels::pseudo_aux_labels;
pub mod encode;
pub mod iccs;
pub mod pgn;
pub mod pipeline;
pub mod shard;
pub mod vocab;

pub use encode::{encode_game, game_result_red, moves_for_game, starting_fen};
pub use pgn::{read_pgn_games, ParsedGame};
pub use pipeline::{run_pgn_shards, write_shards_to_dir};
pub use shard::{read_shard_header, write_pack_meta, write_shard, EncodedGame, EncodedRow};
pub use vocab::{
    collect_vocab_moves_from_pgn,
    collect_vocab_moves_from_pgn_with_jobs,
    enumerate_canonical_vocab_moves,
    load_vocab,
    load_vocab_json_str,
    vocab_sha256_hex,
    write_vocab_json,
};
