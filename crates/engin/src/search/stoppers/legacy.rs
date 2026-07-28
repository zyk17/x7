//! px0 `stoppers/legacy.h:35-42`、`legacy.cc:37-174`。

use std::sync::{Arc, Mutex};

use xiangqi_core::Position;

use crate::uci_loop::GoParams;

use super::stopper::{SearchStopper, TimeLimitStopper};
use super::timemgr::{IterationStats, StoppersHints};

const DEFAULT_MIDPOINT: f32 = 51.5;
const DEFAULT_STEEPNESS: f32 = 7.0;
const DEFAULT_FIRST_MOVE_BONUS: f32 = 1.8;
const DEFAULT_BOOK_PLY_BONUS: f32 = 0.25;
const SMART_PRUNING_TOLERANCE_MS: f32 = 200.0;

/// px0 `ComputeEstimatedMovesToGo` (`legacy.cc:43-61`).
pub fn compute_estimated_moves_to_go(ply: u32, midpoint: f32, steepness: f32) -> f32 {
    let current_move = ply as f32 / 2.0;
    midpoint * (1.0 + 2.0 * (current_move / midpoint).powf(steepness)).powf(1.0 / steepness) - current_move
}

#[derive(Debug)]
struct LegacyState {
    first_move_of_game: bool,
    time_spared_ms: i64,
}

/// px0 `LegacyTimeManager` (`legacy.cc:64-174`), the factory default.
pub struct LegacyTimeManager {
    move_overhead_ms: i64,
    slowmover: f32,
    state: Arc<Mutex<LegacyState>>,
}

impl LegacyTimeManager {
    /// px0 `MakeLegacyTimeManager` (`legacy.cc:170-174`) with factory defaults
    /// from `factory.cc:73-114`.
    pub fn new(move_overhead_ms: i64, slowmover: f32) -> Self {
        Self {
            move_overhead_ms,
            slowmover,
            state: Arc::new(Mutex::new(LegacyState {
                first_move_of_game: true,
                time_spared_ms: 0,
            })),
        }
    }

    pub const fn move_overhead_ms(&self) -> i64 {
        self.move_overhead_ms
    }

    /// px0 `LegacyTimeManager::GetStopper` (`legacy.cc:92-166`).
    pub fn get_stopper(&mut self, params: &GoParams, position: &Position) -> Option<Box<dyn SearchStopper>> {
        let time = if position.is_black_to_move() {
            params.btime
        } else {
            params.wtime
        }?;
        if params.infinite || params.ponder {
            return None;
        }
        let increment = if position.is_black_to_move() {
            params.binc
        } else {
            params.winc
        }
        .unwrap_or(0)
        .max(0);

        let mut moves_to_go = compute_estimated_moves_to_go(position.game_ply(), DEFAULT_MIDPOINT, DEFAULT_STEEPNESS);
        if let Some(value) = params.movestogo.filter(|&value| value > 0) {
            if (value as f32) < moves_to_go {
                moves_to_go = value as f32;
            }
        }

        let mut state = self.state.lock().expect("legacy time manager lock");
        let mut total_moves_time =
            (time as f32 + increment as f32 * (moves_to_go - 1.0) - self.move_overhead_ms as f32).max(0.0);
        let mut time_to_squander = 0_i64;
        if state.time_spared_ms > 0 {
            total_moves_time = (total_moves_time - state.time_spared_ms as f32).max(0.0);
            time_to_squander = (state.time_spared_ms as f32) as i64;
            state.time_spared_ms -= time_to_squander;
        }

        let mut this_move_time = total_moves_time / moves_to_go;
        if state.first_move_of_game {
            this_move_time *=
                1.0 + DEFAULT_FIRST_MOVE_BONUS + DEFAULT_BOOK_PLY_BONUS * position.game_ply().min(12) as f32;
            state.first_move_of_game = false;
        }
        if self.slowmover < 1.0 || this_move_time * self.slowmover > SMART_PRUNING_TOLERANCE_MS {
            state.time_spared_ms -= (this_move_time * (self.slowmover - 1.0)) as i64;
            this_move_time *= self.slowmover;
        }
        this_move_time += time_to_squander as f32;
        let deadline_ms = (this_move_time as i64).min(time - self.move_overhead_ms);
        drop(state);
        Some(Box::new(LegacyStopper {
            limit: TimeLimitStopper::new(deadline_ms),
            state: Arc::clone(&self.state),
        }))
    }

    #[cfg(test)]
    fn time_spared_ms(&self) -> i64 {
        self.state.lock().expect("legacy time manager lock").time_spared_ms
    }
}

/// px0 internal `LegacyStopper` (`legacy.cc:66-88`).
struct LegacyStopper {
    limit: TimeLimitStopper,
    state: Arc<Mutex<LegacyState>>,
}

impl SearchStopper for LegacyStopper {
    fn should_stop(&mut self, stats: &IterationStats, hints: &mut StoppersHints) -> bool {
        self.limit.should_stop(stats, hints)
    }

    fn on_search_done(&mut self, stats: &IterationStats) {
        let mut state = self.state.lock().expect("legacy time manager lock");
        state.time_spared_ms += self.limit.time_limit_ms() - stats.time_since_movestart;
    }
}

#[cfg(test)]
mod tests {
    use xiangqi_core::{Position, STARTPOS_FEN};

    use super::{compute_estimated_moves_to_go, LegacyTimeManager};
    use crate::search::stoppers::{IterationStats, StoppersHints};
    use crate::uci_loop::GoParams;

    #[test]
    fn estimated_moves_matches_px0_curve() {
        assert!((compute_estimated_moves_to_go(0, 51.5, 7.0) - 51.5).abs() < 0.001);
        assert!(compute_estimated_moves_to_go(100, 51.5, 7.0) > 0.0);
    }

    #[test]
    fn legacy_stopper_saves_unused_time_after_search() {
        let position = Position::from_fen(STARTPOS_FEN).expect("startpos");
        let mut manager = LegacyTimeManager::new(200, 1.0);
        let mut stopper = manager
            .get_stopper(
                &GoParams {
                    wtime: Some(10_000),
                    ..GoParams::default()
                },
                &position,
            )
            .expect("clock stopper");
        let mut hints = StoppersHints::default();
        assert!(!stopper.should_stop(&IterationStats::default(), &mut hints));
        stopper.on_search_done(&IterationStats {
            time_since_movestart: 1,
            ..IterationStats::default()
        });
        assert!(manager.time_spared_ms() > 0);
    }
}
