use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::Deserialize;

use engin::{MctsBudget, MctsConfig, MctsEngine, OnnxPolicyValueEval, PolicyOnnx, SharedPolicy};
use xiangqi_core::{move_to_uci, parse_move_uci, Position};

use crate::encode::{moves_for_game, starting_fen};
use crate::pgn::read_pgn_games;
use crate::shard::{write_pack_meta, write_shard, EncodedGame, EncodedRow};
use crate::vocab::load_vocab;

#[derive(Clone, Debug)]
pub struct SearchLabelExportConfig {
    pub max_playouts: u32,
    pub max_games: usize,
    pub max_rows: usize,
    pub max_rows_per_game: usize,
    pub cpuct: f32,
    pub jobs: usize,
}

impl Default for SearchLabelExportConfig {
    fn default() -> Self {
        Self {
            max_playouts: 256,
            max_games: 0,
            max_rows: 0,
            max_rows_per_game: 0,
            cpuct: 1.25,
            jobs: 0,
        }
    }
}

const SEARCH_LABEL_GAMES_PER_SHARD: usize = 256;

#[derive(Debug, Deserialize)]
struct SearchManifest {
    rows: Vec<SearchManifestRow>,
}

#[derive(Clone, Debug, Deserialize)]
struct SearchManifestRow {
    game_id: Option<String>,
    fen: String,
    root_fen: Option<String>,
    uci_prefix: Option<Vec<String>>,
    target_idx: i32,
    legal_idx: Option<Vec<i32>>,
    ply: u16,
    game_result_red: Option<i8>,
    ply_total: Option<u16>,
}

fn flush_search_label_shard(
    out_dir: &Path,
    vocab_hash: &[u8; 32],
    shard_index: usize,
    encoded_games: &mut Vec<EncodedGame>,
) -> Result<()> {
    if encoded_games.is_empty() {
        return Ok(());
    }
    let shard_path = out_dir.join(format!("shard_{shard_index:05}.xrsh"));
    write_shard(&shard_path, vocab_hash, encoded_games)?;
    encoded_games.clear();
    Ok(())
}

fn make_shared_policy(onnx_path: Option<&Path>) -> Result<SharedPolicy> {
    let mut policy: SharedPolicy = None;
    if let Some(path) = onnx_path {
        let net = PolicyOnnx::from_file(path).map_err(anyhow::Error::msg)?;
        policy = Some(Arc::new(Mutex::new(net)));
    }
    Ok(policy)
}

fn make_budget(config: &SearchLabelExportConfig) -> MctsBudget {
    MctsBudget {
        max_playouts: Some(config.max_playouts.max(1)),
        max_nodes: None,
        deadline: None,
        stop: None,
    }
}

fn make_mcts_config(config: &SearchLabelExportConfig) -> MctsConfig {
    MctsConfig {
        cpuct: config.cpuct,
        ..MctsConfig::default()
    }
}

fn resolved_jobs(jobs: usize) -> usize {
    if jobs > 0 {
        return jobs;
    }
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn label_manifest_row(
    row_idx: usize,
    row: &SearchManifestRow,
    engine: &mut MctsEngine<OnnxPolicyValueEval<'_>>,
    budget: &MctsBudget,
    vocab_i32: &HashMap<String, i32>,
) -> Result<Option<(usize, String, EncodedRow)>> {
    let legal_idx = row.legal_idx.as_deref().unwrap_or(&[]);
    if legal_idx.is_empty() {
        return Ok(None);
    }
    let pos = Position::from_fen(&row.fen).map_err(anyhow::Error::msg)?;
    let result = engine.search_root(&pos, budget.clone()).map_err(anyhow::Error::msg)?;
    if result.moves.is_empty() {
        return Ok(None);
    }

    let mut result_legal_idx = Vec::with_capacity(result.moves.len());
    let mut search_counts = Vec::with_capacity(result.moves.len());
    for stat in &result.moves {
        let uci = move_to_uci(stat.mv);
        let Some(&idx) = vocab_i32.get(&uci) else {
            anyhow::bail!("legal move {uci} 不在词表中；search_label 只能使用完整 canonical vocab");
        };
        result_legal_idx.push(idx);
        search_counts.push(stat.visits.min(u16::MAX as u32) as u16);
    }
    if !result_legal_idx.contains(&row.target_idx) {
        return Ok(None);
    }

    let game_id = row
        .game_id
        .clone()
        .unwrap_or_else(|| format!("search_manifest_{row_idx:08}"));
    Ok(Some((
        row_idx,
        game_id,
        EncodedRow {
            fen: row.fen.clone(),
            root_fen: row.root_fen.clone().unwrap_or_else(|| row.fen.clone()),
            uci_prefix: row.uci_prefix.clone().unwrap_or_default(),
            target_idx: row.target_idx,
            legal_idx: result_legal_idx,
            ply: row.ply,
            game_result_red: row.game_result_red.unwrap_or(2),
            ply_total: row.ply_total.unwrap_or(row.ply.saturating_add(1)),
            search_q: result.root_value,
            search_visits: result.playouts,
            search_counts,
        },
    )))
}

pub fn export_search_label_shard_from_pgn(
    pgn_path: &Path,
    vocab_path: &Path,
    out_dir: &Path,
    onnx_path: Option<&Path>,
    config: &SearchLabelExportConfig,
    round: Option<u32>,
) -> Result<u64> {
    let raw = std::fs::read_to_string(pgn_path).with_context(|| format!("读取 PGN {}", pgn_path.display()))?;
    let mut games = read_pgn_games(&raw);
    if config.max_games > 0 {
        games.truncate(config.max_games);
    }

    let (vocab_i32, vocab_hash) = load_vocab(vocab_path)?;
    let vocab_usize: HashMap<String, usize> = vocab_i32.iter().map(|(mv, idx)| (mv.clone(), *idx as usize)).collect();

    let policy = make_shared_policy(onnx_path)?;
    let budget = make_budget(config);
    let mcts_config = make_mcts_config(config);

    let mut encoded_games = Vec::new();
    let mut total_rows = 0u64;
    let mut shard_count = 0usize;

    std::fs::create_dir_all(out_dir).with_context(|| out_dir.display().to_string())?;

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
                    anyhow::bail!("legal move {uci} 不在词表中；search_label 只能使用完整 canonical vocab");
                };
                legal_idx.push(idx);
                search_counts.push(stat.visits.min(u16::MAX as u32) as u16);
            }
            if legal_idx.is_empty() {
                break;
            }
            let Some(&target_idx) = vocab_i32.get(human_uci) else {
                anyhow::bail!(
                    "human move {human_uci} 不在词表中；search_label 只能使用完整 canonical vocab"
                );
            };
            rows.push(EncodedRow {
                fen: pos.fen(),
                root_fen: root_fen.clone(),
                uci_prefix: prefix.clone(),
                target_idx,
                legal_idx,
                ply: ply as u16,
                game_result_red: result_r,
                ply_total,
                search_q: result.root_value,
                search_visits: result.playouts,
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
            if encoded_games.len() >= SEARCH_LABEL_GAMES_PER_SHARD {
                flush_search_label_shard(out_dir, &vocab_hash, shard_count, &mut encoded_games)?;
                shard_count += 1;
            }
        }
    }

    let source_note = match round {
        Some(r) => format!("search_label_pgn:round_{r}"),
        None => "search_label_pgn".to_string(),
    };
    if encoded_games.is_empty() && shard_count == 0 {
        write_pack_meta(out_dir, &vocab_hash, 0, &source_note)?;
        return Ok(0);
    }
    let has_tail_shard = !encoded_games.is_empty();
    flush_search_label_shard(out_dir, &vocab_hash, shard_count, &mut encoded_games)?;
    if has_tail_shard {
        shard_count += 1;
    }
    write_pack_meta(out_dir, &vocab_hash, shard_count, &source_note)?;
    Ok(total_rows)
}

pub fn export_search_label_shard_from_manifest(
    manifest_path: &Path,
    vocab_path: &Path,
    out_dir: &Path,
    onnx_path: Option<&Path>,
    config: &SearchLabelExportConfig,
    round: Option<u32>,
) -> Result<u64> {
    let raw =
        std::fs::read_to_string(manifest_path).with_context(|| format!("读取 manifest {}", manifest_path.display()))?;
    let mut manifest: SearchManifest =
        serde_json::from_str(&raw).with_context(|| format!("解析 manifest {}", manifest_path.display()))?;
    let manifest_rows_total = manifest.rows.len();
    if config.max_rows > 0 {
        if config.max_rows > manifest_rows_total {
            eprintln!(
                "warning: manifest 仅有 {} rows，但请求了 --max-rows {}；将只处理可用 rows",
                manifest_rows_total, config.max_rows
            );
        }
        manifest.rows.truncate(config.max_rows);
    }

    let (vocab_i32, vocab_hash) = load_vocab(vocab_path)?;
    let vocab_usize: HashMap<String, usize> = vocab_i32.iter().map(|(mv, idx)| (mv.clone(), *idx as usize)).collect();

    let budget = make_budget(config);
    let mcts_config = make_mcts_config(config);

    let mut encoded_games = Vec::new();
    let mut rows_by_game: HashMap<String, Vec<EncodedRow>> = HashMap::new();
    let mut shard_count = 0usize;

    std::fs::create_dir_all(out_dir).with_context(|| out_dir.display().to_string())?;

    let job_count = resolved_jobs(config.jobs);
    let labeled_rows = if job_count > 1 && manifest.rows.len() > 1 {
        let chunk_size = manifest.rows.len().div_ceil(job_count).max(1);
        let run = || -> Result<Vec<(usize, String, EncodedRow)>> {
            let chunks: Vec<(usize, &[SearchManifestRow])> = manifest.rows.chunks(chunk_size).enumerate().collect();
            let partials: Vec<Result<Vec<(usize, String, EncodedRow)>>> = chunks
                .into_par_iter()
                .map(|(chunk_idx, rows)| {
                    let policy = make_shared_policy(onnx_path)?;
                    let mut engine = MctsEngine::new(
                        mcts_config,
                        OnnxPolicyValueEval {
                            policy: &policy,
                            vocab: &vocab_usize,
                        },
                    );
                    let mut out = Vec::new();
                    let start = chunk_idx * chunk_size;
                    for (offset, row) in rows.iter().enumerate() {
                        if let Some(labeled) =
                            label_manifest_row(start + offset, row, &mut engine, &budget, &vocab_i32)?
                        {
                            out.push(labeled);
                        }
                    }
                    Ok(out)
                })
                .collect();
            let mut out = Vec::new();
            for part in partials {
                out.extend(part?);
            }
            Ok(out)
        };
        if config.jobs > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(job_count)
                .build()
                .map_err(anyhow::Error::msg)?
                .install(run)?
        } else {
            run()?
        }
    } else {
        let policy = make_shared_policy(onnx_path)?;
        let mut engine = MctsEngine::new(
            mcts_config,
            OnnxPolicyValueEval {
                policy: &policy,
                vocab: &vocab_usize,
            },
        );
        let mut out = Vec::new();
        for (row_idx, row) in manifest.rows.iter().enumerate() {
            if let Some(labeled) = label_manifest_row(row_idx, row, &mut engine, &budget, &vocab_i32)? {
                out.push(labeled);
            }
        }
        out
    };

    let total_rows = labeled_rows.len() as u64;
    for (_row_idx, game_id, row) in labeled_rows {
        rows_by_game.entry(game_id).or_default().push(row);
    }

    let mut ordered: Vec<(String, Vec<EncodedRow>)> = rows_by_game.into_iter().collect();
    ordered.sort_by(|a, b| a.0.cmp(&b.0));
    for (game_id, rows) in ordered {
        if rows.is_empty() {
            continue;
        }
        encoded_games.push(EncodedGame { game_id, rows });
        if encoded_games.len() >= SEARCH_LABEL_GAMES_PER_SHARD {
            flush_search_label_shard(out_dir, &vocab_hash, shard_count, &mut encoded_games)?;
            shard_count += 1;
        }
    }

    let source_note = match round {
        Some(r) => format!("search_label_manifest:round_{r}"),
        None => "search_label_manifest".to_string(),
    };
    if encoded_games.is_empty() && shard_count == 0 {
        write_pack_meta(out_dir, &vocab_hash, 0, &source_note)?;
        return Ok(0);
    }
    let has_tail_shard = !encoded_games.is_empty();
    flush_search_label_shard(out_dir, &vocab_hash, shard_count, &mut encoded_games)?;
    if has_tail_shard {
        shard_count += 1;
    }
    write_pack_meta(out_dir, &vocab_hash, shard_count, &source_note)?;
    Ok(total_rows)
}
