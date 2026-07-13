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
#[derive(Clone, Debug, Default)]
pub struct StoppersHints {
    remaining_time_ms: i64,
    remaining_playouts: i64,
}

impl StoppersHints {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn update_estimated_remaining_time_ms(&mut self, value: i64) {
        self.remaining_time_ms = value;
    }

    pub fn estimated_remaining_time_ms(&self) -> i64 {
        self.remaining_time_ms
    }

    pub fn update_estimated_remaining_playouts(&mut self, value: i64) {
        self.remaining_playouts = value;
    }

    pub fn estimated_remaining_playouts(&self) -> i64 {
        self.remaining_playouts
    }
}
