//! 训练侧数据基础设施。
//!
//! 当前 crate 负责：
//!
//! - PGN → XRSH
//! - 搜索标注字段定义
//! - 维护者侧 vocab / shard CLI

pub mod encode;
pub mod iccs;
pub mod pgn;
pub mod pipeline;
pub mod search_label;
pub mod shard;
pub mod vocab;

pub use encode::{encode_game, game_result_red, moves_for_game, starting_fen};
pub use pgn::{read_pgn_games, ParsedGame};
pub use pipeline::{run_pgn_shards, write_shards_to_dir};
pub use search_label::{
    export_search_label_shard_from_manifest, export_search_label_shard_from_pgn, SearchLabelExportConfig,
};
pub use shard::{read_shard_header, write_pack_meta, write_shard, EncodedGame, EncodedRow};
pub use vocab::{
    collect_vocab_moves_from_pgn, collect_vocab_moves_from_pgn_with_jobs, enumerate_canonical_vocab_moves, load_vocab,
    load_vocab_json_str, vocab_sha256_hex, write_vocab_json,
};
