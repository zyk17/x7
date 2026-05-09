//! 训练侧数据：PGN / JSONL → 二进制分片（**XRSH** v2，`.xrsh`；兼容读取旧 v1 由 Python 侧实现）。
//! PGN 按局并行；JSONL **按行并行**（Rayon，`pipeline::run_jsonl_shards`）。

pub mod aux_labels;
pub mod encode;
pub mod iccs;
pub mod pgn;
pub mod pipeline;
pub mod shard;
pub mod vocab;

pub use encode::{encode_game, encode_jsonl_line, moves_for_game, starting_fen};
pub use pgn::{read_pgn_games, ParsedGame};
pub use pipeline::{run_jsonl_shards, run_pgn_shards, write_shards_to_dir};
pub use shard::{
    read_shard_header, write_pack_meta, write_shard, EncodedGame, EncodedRow,
};
pub use vocab::{load_vocab, load_vocab_json_str, vocab_sha256_hex};
