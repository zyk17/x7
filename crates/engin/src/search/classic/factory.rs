//! Classic implementation of px0's generic `SearchFactory` boundary.

use std::sync::Arc;

use crate::neural::backend::Backend;
use crate::search::{SearchBase, SearchFactory};

use super::ClassicSearch;

#[derive(Clone, Copy, Debug, Default)]
pub struct Factory;

impl SearchFactory for Factory {
    fn create(&self, backend: Arc<dyn Backend>) -> Box<dyn SearchBase> {
        Box::new(ClassicSearch::new(backend))
    }
}
