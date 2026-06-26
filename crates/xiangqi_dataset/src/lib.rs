//! 训练侧数据基础设施。
//!
//! 当前 crate 负责：
//!
//! - PGN 读取
//! - ICCS 记谱清理
//! - 未来自对弈数据预处理的最小地基

pub mod iccs;
pub mod pgn;

pub use iccs::{iccs_half_to_uci, iccs_move_to_uci};
pub use pgn::{
    movetext_iccs_pairs, movetext_uci_tokens, pgn_format, read_pgn_games, strip_comments_and_variations, ParsedGame,
};
