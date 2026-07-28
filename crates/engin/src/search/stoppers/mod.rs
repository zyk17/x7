//! px0 `src/search/classic/stoppers/*` 的独立 stopper 子集。
//!
//! 它不依赖任何 tree/worker 实现，保留给 stream 的正式时钟管理接入。

pub mod legacy;
pub mod stopper;
pub mod timemgr;

pub use legacy::LegacyTimeManager;
pub use stopper::{
    build_search_stoppers, ChainedSearchStopper, PlayoutsStopper, SearchStopper, TimeLimitStopper, VisitsStopper,
};
pub use timemgr::{IterationStats, StoppersHints};
