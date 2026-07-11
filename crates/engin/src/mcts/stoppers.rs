//! lc0 UCI time manager（默认 legacy + simple 参数内化）。

/// lc0 `MoveOverheadMs` 默认（factory.cc:70）。
pub const LC0_DEFAULT_MOVE_OVERHEAD_MS: i64 = 200;

/// lc0 `SimpleTimeManager` 默认参数（simple.cc:39-43）。
const SIMPLE_BASE_PCT: f32 = 1.4;
const SIMPLE_PLY_PCT: f32 = 0.049;
const SIMPLE_TIME_FACTOR: f32 = 1.5;

/// UCI `go` 中与时钟相关的字段。
#[derive(Debug, Clone, Default)]
pub struct UciTimeParams {
    pub infinite: bool,
    pub ponder: bool,
    pub movetime: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u32>,
}

impl UciTimeParams {
    pub fn has_clock(&self) -> bool {
        self.wtime.is_some() || self.btime.is_some()
    }
}

/// lc0 `SimpleTimeManager::GetStopper`（simple.cc:74-125）。
pub fn allocate_simple_think_time_ms(
    params: &UciTimeParams,
    is_black_to_move: bool,
    game_ply: i32,
    move_overhead_ms: i64,
) -> Option<u64> {
    if params.infinite || params.ponder {
        return None;
    }
    let time = if is_black_to_move {
        params.btime
    } else {
        params.wtime
    }?;
    let increment = if is_black_to_move {
        params.binc
    } else {
        params.winc
    }
    .unwrap_or(0);

    let time_available = (time as f32) - move_overhead_ms as f32;
    if time_available <= 0.0 {
        return Some(0);
    }

    let time_ratio = if time > 0 {
        increment as f32 / time as f32
    } else {
        0.0
    };
    let mut pct = (SIMPLE_BASE_PCT + game_ply as f32 * SIMPLE_PLY_PCT) * 0.01;
    pct += time_ratio * SIMPLE_TIME_FACTOR;
    let mut time_budgeted = time_available * pct;
    time_budgeted = time_budgeted.min(time_available);
    Some(time_budgeted.max(0.0) as u64)
}

/// lc0 `LegacyTimeManager::GetStopper` 的 movestogo 均分（legacy.cc:111-168，简化无 spared/slowmover）。
pub fn allocate_legacy_think_time_ms(
    params: &UciTimeParams,
    is_black_to_move: bool,
    game_ply: i32,
    move_overhead_ms: i64,
) -> Option<u64> {
    if params.infinite || params.ponder {
        return None;
    }
    let time = if is_black_to_move {
        params.btime
    } else {
        params.wtime
    }?;
    let increment = if is_black_to_move {
        params.binc
    } else {
        params.winc
    }
    .unwrap_or(0) as i64;

    let mut movestogo = estimate_moves_to_go(game_ply);
    if let Some(mtg) = params.movestogo {
        if mtg > 0 && (mtg as f32) < movestogo {
            movestogo = mtg as f32;
        }
    }

    let total_moves_time = (time as f32 + increment as f32 * (movestogo - 1.0)
        - move_overhead_ms as f32)
        .max(0.0);
    let this_move_time = total_moves_time / movestogo;
    let deadline = this_move_time.min((time as f32) - move_overhead_ms as f32);
    Some(deadline.max(0.0) as u64)
}

/// lc0 legacy 中 `ComputeEstimatedMovesToGo` 的简化：按局内步数估计剩余步数。
fn estimate_moves_to_go(game_ply: i32) -> f32 {
    (80.0 - game_ply as f32 * 0.5).max(10.0)
}

/// 按 lc0 默认 time manager=legacy；无 movestogo 时用 simple 分配。
pub fn allocate_think_time_ms(
    params: &UciTimeParams,
    is_black_to_move: bool,
    game_ply: i32,
) -> Option<u64> {
    let overhead = LC0_DEFAULT_MOVE_OVERHEAD_MS;
    if params.movestogo.is_some() {
        allocate_legacy_think_time_ms(params, is_black_to_move, game_ply, overhead)
    } else {
        allocate_simple_think_time_ms(params, is_black_to_move, game_ply, overhead)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_time_white_to_move_uses_wtime() {
        let params = UciTimeParams {
            wtime: Some(60_000),
            btime: Some(60_000),
            ..Default::default()
        };
        let ms = allocate_simple_think_time_ms(&params, false, 0, 200).expect("time");
        assert!(ms > 0 && ms < 60_000);
    }

    #[test]
    fn simple_time_black_to_move_uses_btime() {
        let params = UciTimeParams {
            wtime: Some(10_000),
            btime: Some(90_000),
            ..Default::default()
        };
        let ms = allocate_simple_think_time_ms(&params, true, 0, 200).expect("time");
        assert!(ms > 0 && ms < 90_000);
    }

    #[test]
    fn legacy_movestogo_splits_remaining_time() {
        let params = UciTimeParams {
            wtime: Some(60_000),
            btime: Some(60_000),
            movestogo: Some(20),
            ..Default::default()
        };
        let ms = allocate_legacy_think_time_ms(&params, false, 0, 200).expect("time");
        assert!(ms >= 2_000 && ms <= 4_000);
    }
}
