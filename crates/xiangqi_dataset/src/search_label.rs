use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use engin::{MctsBudget, MctsConfig, MctsEngine, OnnxPolicyValueEval, PolicyOnnx};
use xiangqi_core::{move_to_uci, parse_move_uci, Position};

use crate::encode::{moves_for_game, starting_fen};
use crate::pgn::read_pgn_games;
use crate::shard::{write_pack_meta, write_shard, EncodedGame, EncodedRow};
use crate::vocab::load_vocab;

#[derive(Clone, Debug)]
pub struct SearchLabelExportConfig {
    pub max_visits: u32,
    pub max_games: usize,
    pub max_rows_per_game: usize,
    pub cpuct: f32,
}

impl Default for SearchLabelExportConfig {
    fn default() -> Self {
        Self {
            max_visits: 256,
            max_games: 0,
            max_rows_per_game: 0,
            cpuct: 1.25,
        }
    }
}

pub fn export_search_label_shard_from_pgn(
    pgn_path: &Path,
    vocab_path: &Path,
    out_dir: &Path,
    onnx_path: Option<&Path>,
    config: &SearchLabelExportConfig,
) -> Result<u64> {
    let raw = std::fs::read_to_string(pgn_path).with_context(|| format!("读取 PGN {}", pgn_path.display()))?;
    let mut games = read_pgn_games(&raw);
    if config.max_games > 0 {
        games.truncate(config.max_games);
    }

    let (vocab_i32, vocab_hash) = load_vocab(vocab_path)?;
    let vocab_usize: HashMap<String, usize> = vocab_i32.iter().map(|(mv, idx)| (mv.clone(), *idx as usize)).collect();

    let policy = Arc::new(Mutex::new(None));
    if let Some(path) = onnx_path {
        let net = PolicyOnnx::from_file(path).map_err(anyhow::Error::msg)?;
        *policy.lock().expect("policy lock") = Some(net);
    }

    let budget = MctsBudget {
        max_visits: Some(config.max_visits.max(1)),
        max_nodes: None,
        deadline: None,
        stop: None,
    };
    let mcts_config = MctsConfig {
        cpuct: config.cpuct,
        ..MctsConfig::default()
    };

    let mut encoded_games = Vec::new();
    let mut total_rows = 0u64;

    for (game_idx, game) in games.iter().enumerate() {
        let (uci_moves, _) = match moves_for_game(game) {
            Ok(v) if !v.0.is_empty() => v,
            _ => continue,
        };
        let root_fen = starting_fen(game)?;
        let mut pos = Position::from_fen(&root_fen).map_err(anyhow::Error::msg)?;
        let mut prefix = Vec::new();
        let mut rows = Vec::new();
        let ply_total = uci_moves.len().min(u16::MAX as usize) as u16;
        let result_r = crate::encode::game_result_red(game);
        let mut engine = MctsEngine::new(
            mcts_config,
            OnnxPolicyValueEval {
                policy: &policy,
                vocab: &vocab_usize,
            },
        );

        for (ply, human_uci) in uci_moves.iter().enumerate() {
            if config.max_rows_per_game > 0 && ply >= config.max_rows_per_game {
                break;
            }

            let result = engine.search_root(&pos, budget.clone()).map_err(anyhow::Error::msg)?;
            if result.moves.is_empty() {
                break;
            }

            let mut legal_idx = Vec::with_capacity(result.moves.len());
            let mut search_counts = Vec::with_capacity(result.moves.len());
            for stat in &result.moves {
                let uci = move_to_uci(stat.mv);
                let Some(&idx) = vocab_i32.get(&uci) else {
                    continue;
                };
                legal_idx.push(idx);
                search_counts.push(stat.visits.min(u16::MAX as u32) as u16);
            }
            if legal_idx.is_empty() {
                break;
            }
            rows.push(EncodedRow {
                fen: pos.fen(),
                root_fen: root_fen.clone(),
                uci_prefix: prefix.clone(),
                target_idx: vocab_i32.get(human_uci).copied().unwrap_or(-1),
                legal_idx,
                ply: ply as u16,
                game_result_red: result_r,
                ply_total,
                search_q: result.root_value,
                search_visits: result.visits,
                search_counts,
            });
            total_rows += 1;

            let Some(mv) = parse_move_uci(human_uci) else {
                break;
            };
            if !pos.legal(mv) {
                break;
            }
            pos.do_move(mv);
            prefix.push(human_uci.clone());
        }

        if !rows.is_empty() {
            encoded_games.push(EncodedGame {
                game_id: format!("search_game_{game_idx:08}"),
                rows,
            });
        }
    }

    std::fs::create_dir_all(out_dir).with_context(|| out_dir.display().to_string())?;
    if encoded_games.is_empty() {
        write_pack_meta(out_dir, &vocab_hash, 0, "search_label_pgn")?;
        return Ok(0);
    }
    write_shard(&out_dir.join("shard_00000.xrsh"), &vocab_hash, &encoded_games)?;
    write_pack_meta(out_dir, &vocab_hash, 1, "search_label_pgn")?;
    Ok(total_rows)
}
