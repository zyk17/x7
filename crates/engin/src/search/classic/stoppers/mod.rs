//! px0 `src/search/classic/stoppers/*` 的 P4 stopper 子集。

#[allow(clippy::module_inception)]
pub mod stoppers;
pub mod timemgr;

pub use stoppers::{
    build_search_stoppers, ChainedSearchStopper, PlayoutsStopper, SearchStopper, TimeLimitStopper, VisitsStopper,
};
pub use timemgr::{IterationStats, StoppersHints};
