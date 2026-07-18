//! px0 `src/search/search.h:45-99` 的 P3 搜索接口。

pub mod classic;
pub mod stream;

use xiangqi_core::GameState;

use crate::{EnginError, GoParams};

/// px0 `SearchBase` (`src/search/search.h:45-84`)。
pub trait SearchBase {
    fn new_game(&mut self) -> Result<(), EnginError>;
    fn set_position(&mut self, state: &GameState) -> Result<(), EnginError>;
    fn start_search(&mut self, params: &GoParams) -> Result<(), EnginError>;
    fn start_clock(&mut self) -> Result<(), EnginError>;
    fn wait_search(&mut self) -> Result<(), EnginError>;
    fn stop_search(&mut self) -> Result<(), EnginError>;
    fn abort_search(&mut self) -> Result<(), EnginError>;
}
