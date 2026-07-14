//! px0 `stoppers/timemgr.h:43-102`、`timemgr.cc:35-66`。

/// px0 `IterationStats` (`timemgr.h:43-61`)。
#[derive(Clone, Debug, Default)]
pub struct IterationStats {
    pub time_since_movestart: i64,
    pub time_since_first_batch: i64,
    pub total_nodes: i64,
    pub nodes_since_movestart: i64,
    pub batches_since_movestart: i64,
    pub average_depth: i32,
}

/// px0 `StoppersHints` (`timemgr.h:68-102`)。
#[derive(Clone, Debug)]
pub struct StoppersHints {
    remaining_time_ms: i64,
    remaining_playouts: i64,
}

impl Default for StoppersHints {
    fn default() -> Self {
        Self {
            // px0 `StoppersHints::Reset` (`timemgr.cc:60-66`). Keep these
            // finite so later arithmetic cannot overflow a u32 node budget.
            remaining_time_ms: 100_000_000_000,
            remaining_playouts: 4_000_000_000,
        }
    }
}

impl StoppersHints {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn update_estimated_remaining_time_ms(&mut self, value: i64) {
        self.remaining_time_ms = self.remaining_time_ms.min(value);
    }

    pub fn estimated_remaining_time_ms(&self) -> i64 {
        self.remaining_time_ms
    }

    pub fn update_estimated_remaining_playouts(&mut self, value: i64) {
        self.remaining_playouts = self.remaining_playouts.min(value);
    }

    pub fn estimated_remaining_playouts(&self) -> i64 {
        self.remaining_playouts.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::StoppersHints;

    #[test]
    fn hints_match_px0_reset_and_minimum_updates() {
        let mut hints = StoppersHints::default();
        assert_eq!(hints.estimated_remaining_time_ms(), 100_000_000_000);
        assert_eq!(hints.estimated_remaining_playouts(), 4_000_000_000);

        hints.update_estimated_remaining_playouts(64);
        hints.update_estimated_remaining_playouts(128);
        assert_eq!(hints.estimated_remaining_playouts(), 64);

        hints.update_estimated_remaining_playouts(-10);
        assert_eq!(hints.estimated_remaining_playouts(), 1);
        hints.reset();
        assert_eq!(hints.estimated_remaining_playouts(), 4_000_000_000);
    }
}
