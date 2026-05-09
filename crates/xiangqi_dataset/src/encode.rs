//! 从对局或 JSONL 行编码为 [`shard::EncodedRow`]（用 `xiangqi_core` 生成合法着下标）。

use crate::aux_labels::pseudo_aux_labels;
use crate::iccs::iccs_move_to_pyffish;
use crate::pgn::{movetext_iccs_pairs, movetext_uci_tokens, pgn_format, ParsedGame};
use crate::shard::{EncodedGame, EncodedRow};
use anyhow::Result;
use std::collections::HashMap;

use xiangqi_core::{legal_moves_uci, parse_pyffish_uci, Position, START_FEN};

/// 棋谱头中的起始 FEN；缺省为 [`START_FEN`]。
pub fn starting_fen(game: &ParsedGame) -> Result<String> {
    if let Some(f) = game.headers.get("FEN") {
        let f = f.trim();
        if !f.is_empty() {
            Position::from_fen(f).map_err(|e| anyhow::anyhow!("无效 FEN: {e}"))?;
            return Ok(f.to_string());
        }
    }
    Ok(START_FEN.to_string())
}

/// PGN 记谱 → pyffish UCI 列表（Python 侧同名逻辑已移除；仅此 Rust 实现）。
pub fn moves_for_game(game: &ParsedGame) -> Result<(Vec<String>, String)> {
    let fmt = pgn_format(&game.headers);
    let text = &game.movetext_raw;

    let iccs_pairs = movetext_iccs_pairs(text);
    if !iccs_pairs.is_empty() && fmt.as_str() != "UCI" {
        let mut py = Vec::new();
        for p in &iccs_pairs {
            py.push(iccs_move_to_pyffish(p)?);
        }
        return Ok((py, if fmt.is_empty() { "ICCS".into() } else { fmt }));
    }

    let uci_toks = movetext_uci_tokens(text);
    if !uci_toks.is_empty() {
        return Ok((uci_toks, if fmt.is_empty() { "UCI".into() } else { fmt }));
    }

    if !iccs_pairs.is_empty() {
        let mut py = Vec::new();
        for p in &iccs_pairs {
            py.push(iccs_move_to_pyffish(p)?);
        }
        return Ok((py, "ICCS".into()));
    }

    Ok((Vec::new(), fmt))
}

/// 一局 → 编码样本（跳过无法映射或非法的尾段，与 Python 类似）。
pub fn encode_game(game: &ParsedGame, game_id: &str, vocab: &HashMap<String, i32>) -> Result<Option<EncodedGame>> {
    let (py_moves, _fmt) = moves_for_game(game)?;
    if py_moves.is_empty() {
        return Ok(None);
    }
    let root_fen = starting_fen(game)?;
    let mut pos = Position::from_fen(&root_fen).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut prefix: Vec<String> = Vec::new();
    let mut rows = Vec::new();

    for (ply, py_uci) in py_moves.iter().enumerate() {
        let legal_str = legal_moves_uci(&pos);
        if !legal_str.iter().any(|x| x == py_uci) {
            break;
        }
        let target_idx = match vocab.get(py_uci) {
            Some(&i) => i,
            None => {
                break;
            }
        };
        let mut legal_idx: Vec<i32> = Vec::with_capacity(legal_str.len());
        for u in &legal_str {
            if let Some(&j) = vocab.get(u) {
                legal_idx.push(j);
            }
        }
        if legal_idx.len() != legal_str.len() {
            // 词表未覆盖全部合法着，无法做稀疏掩码训练
            break;
        }

        let (aux_attack, aux_danger, aux_tactical) = pseudo_aux_labels(&pos);

        rows.push(EncodedRow {
            fen: pos.fen(),
            root_fen: root_fen.clone(),
            uci_prefix: prefix.clone(),
            target_idx,
            legal_idx,
            ply: ply as u16,
            aux_attack,
            aux_danger,
            aux_tactical,
        });

        let Some(mv) = parse_pyffish_uci(py_uci) else {
            break;
        };
        if !pos.legal(mv) {
            break;
        }
        pos.do_move(mv);
        prefix.push(py_uci.clone());
    }

    if rows.is_empty() {
        return Ok(None);
    }
    Ok(Some(EncodedGame {
        game_id: game_id.to_string(),
        rows,
    }))
}

/// 从 JSONL 的一行编码（字段：`fen` / `root_fen` / `uci_prefix` / `human_move_pyffish` / …）。
pub fn encode_jsonl_line(line: &str, vocab: &HashMap<String, i32>) -> Result<Option<EncodedRow>> {
    let v: serde_json::Value = serde_json::from_str(line)?;
    let Some(fen) = v["fen"].as_str().map(str::to_string) else {
        return Ok(None);
    };
    let Some(root_fen) = v["root_fen"].as_str().map(str::to_string) else {
        return Ok(None);
    };
    let Some(human) = v["human_move_pyffish"].as_str() else {
        return Ok(None);
    };
    let ply = v["ply"].as_u64().unwrap_or(0) as u16;

    let prefix_json = v["uci_prefix"].as_array().cloned().unwrap_or_default();
    let mut prefix: Vec<String> = Vec::new();
    for p in prefix_json {
        if let Some(s) = p.as_str() {
            prefix.push(s.to_string());
        }
    }

    let mut pos = match Position::from_fen(&root_fen) {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    for u in &prefix {
        let Some(mv) = parse_pyffish_uci(u) else {
            return Ok(None);
        };
        if !pos.legal(mv) {
            return Ok(None);
        }
        pos.do_move(mv);
    }
    if pos.fen() != fen {
        return Ok(None);
    }

    let legal_str = legal_moves_uci(&pos);
    if !legal_str.iter().any(|x| x == human) {
        return Ok(None);
    }
    let Some(&target_idx) = vocab.get(human) else {
        return Ok(None);
    };
    let mut legal_idx: Vec<i32> = Vec::with_capacity(legal_str.len());
    for u in &legal_str {
        let Some(&j) = vocab.get(u) else {
            return Ok(None);
        };
        legal_idx.push(j);
    }

    let (aux_attack, aux_danger, aux_tactical) = pseudo_aux_labels(&pos);

    Ok(Some(EncodedRow {
        fen,
        root_fen,
        uci_prefix: prefix,
        target_idx,
        legal_idx,
        ply,
        aux_attack,
        aux_danger,
        aux_tactical,
    }))
}
