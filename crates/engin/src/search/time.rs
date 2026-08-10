//! UCI 时钟预算。
//!
//! 分配形状历史上参考过 px0 legacy stopper；stream 只在搜索启动时取得 deadline、
//! 在 drain 后归还未用时间，不保留通用停止链或可调进攻/保守倍率。

use std::time::{Duration, Instant};

use xiangqi_core::Position;

use crate::uci_loop::GoParams;

const MOVE_OVERHEAD_MS: i64 = 200;
const DEFAULT_MIDPOINT: f32 = 51.5;
const DEFAULT_STEEPNESS: f32 = 7.0;
const DEFAULT_FIRST_MOVE_BONUS: f32 = 1.8;
const DEFAULT_BOOK_PLY_BONUS: f32 = 0.25;

/// 跨回合保留的时钟状态。
#[derive(Debug)]
pub(crate) struct TimeManager {
    first_move_of_game: bool,
    time_spared_ms: i64,
}

impl Default for TimeManager {
    fn default() -> Self {
        Self {
            first_move_of_game: true,
            time_spared_ms: 0,
        }
    }
}

impl TimeManager {
    /// 计算当前一手的固定中性预算。
    ///
    /// 对应 px0 `LegacyTimeManager::GetStopper`（`legacy.cc:92-166`）；只保留
    /// 固定中性的分配公式，不提供激进或保守倍率。
    pub(crate) fn budget(&mut self, params: &GoParams, position: &Position) -> Option<TimeBudget> {
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

        let mut moves_to_go = estimated_moves_to_go(position.game_ply());
        if let Some(value) = params.movestogo.filter(|&value| value > 0) {
            moves_to_go = moves_to_go.min(value as f32);
        }

        let mut total_moves_time =
            (time as f32 + increment as f32 * (moves_to_go - 1.0) - MOVE_OVERHEAD_MS as f32).max(0.0);
        let time_to_squander = self.time_spared_ms.max(0);
        if time_to_squander > 0 {
            total_moves_time = (total_moves_time - time_to_squander as f32).max(0.0);
            self.time_spared_ms -= time_to_squander;
        }

        let mut this_move_time = total_moves_time / moves_to_go;
        if self.first_move_of_game {
            this_move_time *=
                1.0 + DEFAULT_FIRST_MOVE_BONUS + DEFAULT_BOOK_PLY_BONUS * position.game_ply().min(12) as f32;
            self.first_move_of_game = false;
        }
        this_move_time += time_to_squander as f32;
        Some(TimeBudget {
            limit_ms: (this_move_time as i64).min(time - MOVE_OVERHEAD_MS),
        })
    }

    /// 新对局清除 px0 legacy 预算中保留的首手与剩余时间状态。
    pub(crate) fn reset(&mut self) {
        *self = Self {
            first_move_of_game: true,
            time_spared_ms: 0,
        };
    }

    /// 搜索 drain 后归还未用预算。
    pub(crate) fn finish(&mut self, budget: TimeBudget, elapsed: Duration) {
        self.time_spared_ms += budget.limit_ms - elapsed.as_millis().min(i64::MAX as u128) as i64;
    }
}

/// 已经分配给当前搜索、不可再变的时钟预算。
#[derive(Clone, Copy, Debug)]
pub(crate) struct TimeBudget {
    limit_ms: i64,
}

impl TimeBudget {
    pub(crate) fn deadline_after(self, started: Instant) -> Instant {
        started + Duration::from_millis(self.limit_ms.max(0) as u64)
    }
}

/// 估计剩余着数，供时钟分配使用。
fn estimated_moves_to_go(ply: u32) -> f32 {
    let current_move = ply as f32 / 2.0;
    DEFAULT_MIDPOINT
        * (1.0 + 2.0 * (current_move / DEFAULT_MIDPOINT).powf(DEFAULT_STEEPNESS)).powf(1.0 / DEFAULT_STEEPNESS)
        - current_move
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use xiangqi_core::{Position, STARTPOS_FEN};

    use super::{TimeManager, estimated_moves_to_go};
    use crate::uci_loop::GoParams;

    #[test]
    fn estimated_moves_matches_px0_curve() {
        assert!((estimated_moves_to_go(0) - 51.5).abs() < 0.001);
        assert!(estimated_moves_to_go(100) > 0.0);
    }

    #[test]
    fn unused_clock_time_is_available_to_the_next_move() {
        let position = Position::from_fen(STARTPOS_FEN).expect("startpos");
        let params = GoParams {
            wtime: Some(10_000),
            ..GoParams::default()
        };
        let mut manager = TimeManager::default();
        let budget = manager.budget(&params, &position).expect("clock budget");
        manager.finish(budget, Duration::from_millis(1));
        assert!(manager.time_spared_ms > 0);
    }
}
