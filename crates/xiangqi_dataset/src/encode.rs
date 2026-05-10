//! 从 **PGN 对局**编码为 [`shard::EncodedRow`]（`xiangqi_core` 生成合法着下标）。

use crate::aux_labels::pseudo_aux_labels;
use crate::iccs::iccs_move_to_uci;
use crate::pgn::{movetext_iccs_pairs, movetext_uci_tokens, pgn_format, ParsedGame};
use crate::shard::{EncodedGame, EncodedRow};
use anyhow::Result;
use std::collections::HashMap;

use xiangqi_core::{legal_moves_uci, parse_move_uci, Position, START_FEN};

/// PGN `[Result "..."]` → 红方视角：`1` 红胜、`-1` 黑胜、`0` 和、`2` 未知。
///
/// 识别常见写法（与多数棋库一致）：`1-0`、`0-1`、`1/2-1/2`；另接受 `1/2` 表示和棋。
/// 标签值会先 **trim** 并去掉内部空白再匹配，故 `1 - 0` 等同 `1-0`。
pub fn game_result_red(game: &ParsedGame) -> i8 {
    let Some(raw) = game.headers.get("Result") else {
        return 2;
    };
    let compact: String = raw.trim().chars().filter(|c| !c.is_whitespace()).collect();
    match compact.as_str() {
        "1-0" => 1,
        "0-1" => -1,
        "1/2-1/2" | "1/2" => 0,
        "*" => 2,
        _ => 2,
    }
}

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

/// PGN 记谱 → 着法 UCI 字符串列表（仅此 Rust 实现）。
pub fn moves_for_game(game: &ParsedGame) -> Result<(Vec<String>, String)> {
    let fmt = pgn_format(&game.headers);
    let text = &game.movetext_raw;

    let iccs_pairs = movetext_iccs_pairs(text);
    if !iccs_pairs.is_empty() && fmt.as_str() != "UCI" {
        let mut uci_moves = Vec::new();
        for p in &iccs_pairs {
            uci_moves.push(iccs_move_to_uci(p)?);
        }
        return Ok((uci_moves, if fmt.is_empty() { "ICCS".into() } else { fmt }));
    }

    let uci_toks = movetext_uci_tokens(text);
    if !uci_toks.is_empty() {
        return Ok((uci_toks, if fmt.is_empty() { "UCI".into() } else { fmt }));
    }

    if !iccs_pairs.is_empty() {
        let mut uci_moves = Vec::new();
        for p in &iccs_pairs {
            uci_moves.push(iccs_move_to_uci(p)?);
        }
        return Ok((uci_moves, "ICCS".into()));
    }

    Ok((Vec::new(), fmt))
}

/// 一局 → 编码样本（跳过无法映射或非法的尾段，与 Python 类似）。
pub fn encode_game(game: &ParsedGame, game_id: &str, vocab: &HashMap<String, i32>) -> Result<Option<EncodedGame>> {
    let (uci_moves, _fmt) = moves_for_game(game)?;
    if uci_moves.is_empty() {
        return Ok(None);
    }
    let result_r = game_result_red(game);
    let ply_total: u16 = uci_moves.len().min(u16::MAX as usize) as u16;
    let root_fen = starting_fen(game)?;
    let mut pos = Position::from_fen(&root_fen).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut prefix: Vec<String> = Vec::new();
    let mut rows = Vec::new();

    for (ply, uci_tok) in uci_moves.iter().enumerate() {
        let legal_str = legal_moves_uci(&pos);
        if !legal_str.iter().any(|x| x == uci_tok) {
            break;
        }
        let target_idx = match vocab.get(uci_tok) {
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
            game_result_red: result_r,
            ply_total,
        });

        let Some(mv) = parse_move_uci(uci_tok) else {
            break;
        };
        if !pos.legal(mv) {
            break;
        }
        pos.do_move(mv);
        prefix.push(uci_tok.clone());
    }

    if rows.is_empty() {
        return Ok(None);
    }
    Ok(Some(EncodedGame {
        game_id: game_id.to_string(),
        rows,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pgn::ParsedGame;
    use std::collections::HashMap;

    fn game_with_result(result: &str) -> ParsedGame {
        let mut headers = HashMap::new();
        headers.insert("Result".to_string(), result.to_string());
        ParsedGame {
            headers,
            movetext_raw: String::new(),
        }
    }

    #[test]
    fn game_result_red_standard_headers() {
        assert_eq!(game_result_red(&game_with_result("1-0")), 1);
        assert_eq!(game_result_red(&game_with_result("0-1")), -1);
        assert_eq!(game_result_red(&game_with_result("1/2-1/2")), 0);
    }

    #[test]
    fn game_result_red_whitespace_inside_score() {
        assert_eq!(game_result_red(&game_with_result("  1 - 0  ")), 1);
        assert_eq!(game_result_red(&game_with_result("0 -\t1")), -1);
    }
}
