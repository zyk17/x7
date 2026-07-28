//! px0 `src/search/search.h:45-99` 的 P3 搜索接口。

pub mod classic;
pub mod factory;
pub mod stream;

use xiangqi_core::GameState;

use std::sync::Arc;

use crate::{callbacks::SearchResponder, EnginError, GoParams};

pub use factory::SearchFactory;

/// px0 `SearchBase` (`src/search/search.h:45-84`)。
pub trait SearchBase {
    /// Installs the structured output sink before a search starts.  The sink
    /// is owned by the caller and may safely be used by a watchdog thread.
    fn set_responder(&mut self, responder: Option<Arc<dyn SearchResponder>>);
    fn new_game(&mut self) -> Result<(), EnginError>;
    fn set_position(&mut self, state: &GameState) -> Result<(), EnginError>;
    /// Validates implementation-specific UCI limits before Engine interrupts
    /// an active search. px0 applies its stopper configuration before worker
    /// startup (`src/search/classic/wrapper.cc:114-140`).
    fn validate_go(&self, _params: &GoParams) -> Result<(), EnginError> {
        Ok(())
    }
    fn start_search(&mut self, params: &GoParams) -> Result<(), EnginError>;
    fn start_clock(&mut self) -> Result<(), EnginError>;
    fn wait_search(&mut self) -> Result<(), EnginError>;
    fn stop_search(&mut self) -> Result<(), EnginError>;
    fn abort_search(&mut self) -> Result<(), EnginError>;

    fn best_move(&self) -> Option<xiangqi_core::Move> {
        None
    }
}
