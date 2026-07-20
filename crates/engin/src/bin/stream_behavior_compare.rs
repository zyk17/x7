//! Correctness probes for stream vs classic behavior.
//!
//! Cases:
//! 1. startpos uniform — settled, legal set, budget
//! 2. checkmated root — stream terminal, bestmove null (`Move::NULL`)
//! 3. black-to-move — UCI-oriented bestmove matches classic
//!
//! Ranking (terminal win > loss, then N/Q/P) is covered by
//! `stats::tests::terminal_win_outranks_higher_n_terminal_loss`.
//! Not a UCI path.

use std::collections::BTreeSet;
use std::process::ExitCode;
use std::sync::Arc;

use engin::neural::backend::{Backend, UniformBackend};
use engin::search::classic::ClassicSearch;
use engin::search::stream::{
    best_move, principal_variation, root_settled, root_stats, ExpansionState, Search, SearchConfig,
    SearchGeneration,
};
use engin::SearchBase;
use xiangqi_core::{GameState, Move, PositionHistory, STARTPOS_FEN};

const CHECKMATE_BLACK_TO_MOVE: &str = "4k4/3RPR3/4C4/9/9/9/9/9/9/4K4 b - - 0 1";

/// Board-absolute move from a UCI-oriented bestmove (px0 flip is involutive).
fn board_absolute(mv: Move, root_is_black: bool) -> Move {
    if root_is_black && !mv.is_null() {
        mv.flip()
    } else {
        mv
    }
}

fn case_startpos_uniform() -> Result<(), String> {
    println!("== case startpos_uniform ==");
    let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).map_err(|e| e.to_string())?;
    let history = Arc::new(PositionHistory::from_positions(state.positions()));
    let root_is_black = history.is_black_to_move();
    let board_legal: BTreeSet<_> = history
        .last()
        .board()
        .generate_legal_moves()
        .iter()
        .map(|m| m.to_uci())
        .collect();
    let playouts = 64u64;

    let mut classic = ClassicSearch::new(Box::new(UniformBackend::default()));
    classic.set_position(&state).map_err(|e| e.to_string())?;
    let (classic_best, _) = classic.run_blocking_nodes(playouts as u32);

    let mut search = Search::new(
        Arc::new(UniformBackend::default()) as Arc<dyn Backend>,
        SearchGeneration(1),
        history,
        SearchConfig::default(),
    );
    let stream_stats = search.run_playouts(playouts).map_err(|e| e.to_string())?;
    let stream_best =
        best_move(search.repository(), search.root_key(), root_is_black).ok_or("stream missing bestmove")?;
    let stream_legal: BTreeSet<_> = root_stats(search.repository(), search.root_key())
        .ok_or("stream root")?
        .edges
        .iter()
        .map(|e| e.mv.to_uci())
        .collect();
    if stream_stats.completed_playouts != playouts {
        return Err(format!(
            "budget mismatch stream={} want={playouts}",
            stream_stats.completed_playouts
        ));
    }
    if !root_settled(search.repository(), search.root_key()) {
        return Err("root not settled".into());
    }
    if stream_legal != board_legal {
        return Err("stream legal set mismatch".into());
    }
    println!(
        "classic={} stream={} equal={}",
        classic_best.to_uci(),
        stream_best.to_uci(),
        classic_best == stream_best,
    );
    if !board_legal.contains(&board_absolute(stream_best, root_is_black).to_uci()) {
        return Err("bestmove not in legal set".into());
    }
    let pv = principal_variation(search.repository(), search.root_key(), root_is_black);
    if pv.first().copied() != Some(stream_best) {
        return Err(format!("pv[0]={:?} != best={}", pv.first(), stream_best.to_uci()));
    }
    search.stop_and_join();
    println!("startpos_uniform PASS");
    Ok(())
}

fn case_checkmate_root() -> Result<(), String> {
    println!("== case checkmate_root ==");
    let state =
        GameState::from_fen_moves(CHECKMATE_BLACK_TO_MOVE, &[] as &[&str]).map_err(|e| e.to_string())?;
    let history = Arc::new(PositionHistory::from_positions(state.positions()));
    let root_is_black = history.is_black_to_move();

    let mut classic = ClassicSearch::new(Box::new(UniformBackend::default()));
    classic.set_position(&state).map_err(|e| e.to_string())?;
    let (classic_best, _) = classic.run_blocking_nodes(8);

    let mut search = Search::new(
        Arc::new(UniformBackend::default()) as Arc<dyn Backend>,
        SearchGeneration(2),
        history,
        SearchConfig::default(),
    );
    search.run_playouts(1).map_err(|e| e.to_string())?;
    let root = search
        .repository()
        .get(search.root_key())
        .ok_or("missing terminal root")?;
    if root.expansion_state() != ExpansionState::Terminal {
        return Err(format!("expected Terminal, got {:?}", root.expansion_state()));
    }
    let stream_best = best_move(search.repository(), search.root_key(), root_is_black);
    search.stop_and_join();
    if stream_best != Some(Move::NULL) {
        return Err(format!(
            "terminal root must report Move::NULL, got {:?}",
            stream_best.map(|m| m.to_uci())
        ));
    }
    if !classic_best.is_null() {
        return Err(format!(
            "classic checkmate bestmove expected null, got {}",
            classic_best.to_uci()
        ));
    }
    println!(
        "checkmate_root PASS classic={} stream_bestmove=null terminal=true",
        classic_best.to_uci()
    );
    Ok(())
}

fn case_black_to_move_orient() -> Result<(), String> {
    println!("== case black_to_move_orient ==");
    let state = GameState::from_fen_moves(STARTPOS_FEN, &["h2e2"]).map_err(|e| e.to_string())?;
    let history = Arc::new(PositionHistory::from_positions(state.positions()));
    if !history.is_black_to_move() {
        return Err("expected black to move".into());
    }
    let root_is_black = true;
    let playouts = 48u64;

    let mut classic = ClassicSearch::new(Box::new(UniformBackend::default()));
    classic.set_position(&state).map_err(|e| e.to_string())?;
    let (classic_best, _) = classic.run_blocking_nodes(playouts as u32);

    let mut search = Search::new(
        Arc::new(UniformBackend::default()) as Arc<dyn Backend>,
        SearchGeneration(3),
        history,
        SearchConfig::default(),
    );
    search.run_playouts(playouts).map_err(|e| e.to_string())?;
    let stream_best =
        best_move(search.repository(), search.root_key(), root_is_black).ok_or("stream best")?;

    let legal: BTreeSet<_> = search
        .repository()
        .get(search.root_key())
        .ok_or("root")?
        .edges()
        .iter()
        .map(|e| e.mv().to_uci())
        .collect();
    search.stop_and_join();
    println!(
        "classic_uci={} stream_uci={} equal={}",
        classic_best.to_uci(),
        stream_best.to_uci(),
        classic_best == stream_best,
    );
    if !legal.contains(&board_absolute(stream_best, root_is_black).to_uci())
        || !legal.contains(&board_absolute(classic_best, root_is_black).to_uci())
    {
        return Err("UCI-oriented bestmove is not a legal board move".into());
    }
    println!("black_to_move_orient PASS");
    Ok(())
}

fn main() -> ExitCode {
    let cases = [
        case_startpos_uniform,
        case_checkmate_root,
        case_black_to_move_orient,
    ];
    for case in cases {
        if let Err(error) = case() {
            eprintln!("FAIL: {error}");
            return ExitCode::from(2);
        }
    }
    println!("ALL PASS");
    ExitCode::SUCCESS
}
