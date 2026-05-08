//! 训练侧数据：PGN / JSONL → 二进制分片（`XQB` v1），按局并行（PGN）。

pub mod encode;
pub mod iccs;
pub mod pgn;
pub mod shard;
pub mod vocab;

pub use encode::{encode_game, encode_jsonl_line, moves_for_game, starting_fen};
pub use pgn::{read_pgn_games, ParsedGame};
pub use shard::{write_pack_meta, write_shard, EncodedGame, EncodedRow};
pub use vocab::{load_vocab, vocab_sha256_hex};
