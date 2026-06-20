//! 词表 JSON（`{ "moves": [...] }`，可选 `size`）与 SHA256 指纹。
//!
//! 从 PGN 建词表时，对棋谱中每一局沿真实着法重放，在**每一步落子前**把当前局面的全部合法 UCI 并入集合，保证与 [`crate::encode::encode_game`] 所需的稀疏掩码一致。

use crate::encode::{moves_for_game, starting_fen};
use crate::pgn::{read_pgn_games, ParsedGame};
use anyhow::{Context, Result};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use xiangqi_core::{legal_moves_uci, parse_move_uci, Position};

#[derive(Debug, Deserialize)]
pub struct VocabFile {
    pub moves: Vec<String>,
}

/// 解析 JSON 字符串（测试 / 内存调用）。
pub fn load_vocab_json_str(text: &str) -> Result<(HashMap<String, i32>, [u8; 32])> {
    let v: VocabFile = serde_json::from_str(text).context("词表 JSON 解析失败（需要 moves 数组）")?;
    Ok(hash_vocab_moves(v.moves))
}

/// 加载词表并计算与 Python `vocab_fingerprint_ordered_moves` 一致的 SHA-256（32 字节）。
pub fn load_vocab(path: &Path) -> Result<(HashMap<String, i32>, [u8; 32])> {
    let text = std::fs::read_to_string(path).with_context(|| format!("读取词表 {}", path.display()))?;
    load_vocab_json_str(&text)
}

fn hash_vocab_moves(moves: Vec<String>) -> (HashMap<String, i32>, [u8; 32]) {
    let mut hasher = Sha256::new();
    for m in &moves {
        hasher.update(m.as_bytes());
        hasher.update(b"\0");
    }
    let hash: [u8; 32] = hasher.finalize().into();
    let map: HashMap<String, i32> = moves.into_iter().enumerate().map(|(i, m)| (m, i as i32)).collect();
    (map, hash)
}

pub fn vocab_sha256_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// 直接按象棋几何规则枚举**所有可能出现的合法 UCI 着法串**（标准 `a0`～`i9`）。
///
/// 该集合不依赖语料，顺序固定为字典序，可作为仓库级 canonical `move_vocab.json`。
pub fn enumerate_canonical_vocab_moves() -> Vec<String> {
    let mut seen = BTreeSet::new();
    enumerate_rook_like(&mut seen);
    enumerate_horse(&mut seen);
    enumerate_elephant(&mut seen);
    enumerate_advisor(&mut seen);
    enumerate_king(&mut seen);
    enumerate_pawn(&mut seen);
    seen.into_iter().collect()
}

/// 从 PGN 文本收集词表：各局重放路径上，每一步的「当前局面全部合法 UCI」之并集（有序、去重）。
pub fn collect_vocab_moves_from_pgn(raw: &str, max_games: usize) -> Result<Vec<String>> {
    collect_vocab_moves_from_pgn_with_jobs(raw, max_games, 0)
}

/// 从 PGN 文本收集词表，支持显式控制 Rayon 并行线程数（0 = 全局默认并行度）。
pub fn collect_vocab_moves_from_pgn_with_jobs(raw: &str, max_games: usize, jobs: usize) -> Result<Vec<String>> {
    let mut games = read_pgn_games(raw);
    if max_games > 0 {
        games.truncate(max_games);
    }
    let run = || collect_vocab_moves_from_games(&games);
    if jobs == 0 {
        run()
    } else {
        ThreadPoolBuilder::new().num_threads(jobs).build()?.install(run)
    }
}

fn collect_vocab_moves_from_games(games: &[ParsedGame]) -> Result<Vec<String>> {
    let per_game_sets: Vec<Result<BTreeSet<String>>> = games.par_iter().map(collect_vocab_moves_for_game).collect();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for game_set in per_game_sets {
        seen.extend(game_set?);
    }
    Ok(seen.into_iter().collect())
}

fn collect_vocab_moves_for_game(g: &ParsedGame) -> Result<BTreeSet<String>> {
    let (uci_moves, _) = match moves_for_game(g) {
        Ok(x) => x,
        Err(_) => return Ok(BTreeSet::new()),
    };
    if uci_moves.is_empty() {
        return Ok(BTreeSet::new());
    }
    let root_fen = starting_fen(g)?;
    let mut pos = Position::from_fen(&root_fen).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut seen = BTreeSet::new();
    for uci_tok in &uci_moves {
        let legal = legal_moves_uci(&pos);
        seen.extend(legal.iter().cloned());
        let Some(mv) = parse_move_uci(uci_tok) else {
            break;
        };
        if !legal.iter().any(|x| x == uci_tok) || !pos.legal(mv) {
            break;
        }
        pos.do_move(mv);
    }
    Ok(seen)
}

fn push_move(seen: &mut BTreeSet<String>, ff: i32, fr: i32, tf: i32, tr: i32) {
    if !(0..=8).contains(&ff) || !(0..=8).contains(&tf) || !(0..=9).contains(&fr) || !(0..=9).contains(&tr) {
        return;
    }
    if ff == tf && fr == tr {
        return;
    }
    seen.insert(format!(
        "{}{}{}{}",
        (b'a' + ff as u8) as char,
        fr,
        (b'a' + tf as u8) as char,
        tr
    ));
}

fn enumerate_rook_like(seen: &mut BTreeSet<String>) {
    for ff in 0..=8 {
        for fr in 0..=9 {
            for tf in 0..=8 {
                if tf != ff {
                    push_move(seen, ff, fr, tf, fr);
                }
            }
            for tr in 0..=9 {
                if tr != fr {
                    push_move(seen, ff, fr, ff, tr);
                }
            }
        }
    }
}

fn enumerate_horse(seen: &mut BTreeSet<String>) {
    const DELTAS: &[(i32, i32)] = &[(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
    for ff in 0..=8 {
        for fr in 0..=9 {
            for &(df, dr) in DELTAS {
                push_move(seen, ff, fr, ff + df, fr + dr);
            }
        }
    }
}

fn enumerate_elephant(seen: &mut BTreeSet<String>) {
    const DELTAS: &[(i32, i32)] = &[(2, 2), (2, -2), (-2, -2), (-2, 2)];
    for ff in 0..=8 {
        for fr in 0..=9 {
            for &(df, dr) in DELTAS {
                let tr = fr + dr;
                if (0..=4).contains(&fr) && (0..=4).contains(&tr) {
                    push_move(seen, ff, fr, ff + df, tr);
                }
                if (5..=9).contains(&fr) && (5..=9).contains(&tr) {
                    push_move(seen, ff, fr, ff + df, tr);
                }
            }
        }
    }
}

fn enumerate_advisor(seen: &mut BTreeSet<String>) {
    const DELTAS: &[(i32, i32)] = &[(1, 1), (1, -1), (-1, -1), (-1, 1)];
    for &(r0, r1) in &[(0, 2), (7, 9)] {
        for ff in 3..=5 {
            for fr in r0..=r1 {
                for &(df, dr) in DELTAS {
                    let tf = ff + df;
                    let tr = fr + dr;
                    if (3..=5).contains(&tf) && (r0..=r1).contains(&tr) {
                        push_move(seen, ff, fr, tf, tr);
                    }
                }
            }
        }
    }
}

fn enumerate_king(seen: &mut BTreeSet<String>) {
    const DELTAS: &[(i32, i32)] = &[(1, 0), (0, 1), (-1, 0), (0, -1)];
    for &(r0, r1) in &[(0, 2), (7, 9)] {
        for ff in 3..=5 {
            for fr in r0..=r1 {
                for &(df, dr) in DELTAS {
                    let tf = ff + df;
                    let tr = fr + dr;
                    if (3..=5).contains(&tf) && (r0..=r1).contains(&tr) {
                        push_move(seen, ff, fr, tf, tr);
                    }
                }
            }
        }
    }
    for ff in 3..=5 {
        for fr in 0..=2 {
            for tr in 7..=9 {
                push_move(seen, ff, fr, ff, tr);
                push_move(seen, ff, tr, ff, fr);
            }
        }
    }
}

fn enumerate_pawn(seen: &mut BTreeSet<String>) {
    for ff in 0..=8 {
        for fr in 0..=9 {
            // Red pawn: forward is +1, after crossing river (rank >= 5) can also move sideways.
            push_move(seen, ff, fr, ff, fr + 1);
            if fr >= 5 {
                push_move(seen, ff, fr, ff - 1, fr);
                push_move(seen, ff, fr, ff + 1, fr);
            }
            // Black pawn: forward is -1, after crossing river (rank <= 4) can also move sideways.
            push_move(seen, ff, fr, ff, fr - 1);
            if fr <= 4 {
                push_move(seen, ff, fr, ff - 1, fr);
                push_move(seen, ff, fr, ff + 1, fr);
            }
        }
    }
}

/// 写出与 [`load_vocab`] 兼容的 JSON（`moves` + `size`）。
pub fn write_vocab_json(path: &Path, moves: &[String]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| parent.display().to_string())?;
    }
    let text =
        serde_json::to_string_pretty(&json!({ "moves": moves, "size": moves.len() })).context("序列化词表 JSON")?;
    fs::write(path, text).with_context(|| format!("写入 {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_hash_stable() {
        let j = r#"{"moves":["a1b2","h2h3"]}"#;
        let (_, h1) = load_vocab_json_str(j).unwrap();
        let (_, h2) = load_vocab_json_str(j).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(vocab_sha256_hex(&h1).len(), 64);
    }

    #[test]
    fn canonical_vocab_contains_core_move_shapes() {
        let moves = enumerate_canonical_vocab_moves();
        let set: BTreeSet<_> = moves.iter().cloned().collect();
        assert!(set.contains("a0a9"));
        assert!(set.contains("b0c2"));
        assert!(set.contains("c0e2"));
        assert!(set.contains("d0e1"));
        assert!(set.contains("e0e9"));
        assert!(set.contains("a3a4"));
        assert!(set.contains("a5b5"));
    }
}
