use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::worker::{cancel_collision_path, PendingKind, SearchIteration};
use super::{MctsConfig, MctsTree};

pub(crate) type SharedMctsTree = Arc<RwLock<MctsTree>>;

#[derive(Clone)]
pub(crate) struct SharedCollisionEntry {
    pub path: Vec<super::worker::PathStep>,
    pub multivisit: u32,
}

/// px0 `shared_collisions_`：跨 worker 记录 collision，在任意 batch 有实质工作时统一 cancel。
#[derive(Default)]
pub(crate) struct SharedCollisions {
    entries: Mutex<Vec<SharedCollisionEntry>>,
}

impl SharedCollisions {
    pub fn collect(&self, iteration: &SearchIteration) {
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        for pending in &iteration.pending {
            if !matches!(pending.kind, PendingKind::Collision) {
                continue;
            }
            if !pending.path.is_empty() {
                guard.push(SharedCollisionEntry {
                    path: pending.path.clone(),
                    multivisit: pending.multivisit,
                });
            }
        }
    }

    pub fn cancel_all(&self, tree: &mut MctsTree) {
        let entries = {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *guard)
        };
        for entry in entries {
            cancel_collision_path(tree, &entry.path, entry.multivisit);
        }
    }
}

pub(crate) fn calculate_collisions_left(nodes: i64, config: MctsConfig) -> i32 {
    let end = i64::from(config.max_collision_visits_scaling_end.max(1));
    let start = i64::from(config.max_collision_visits_scaling_start.max(1));
    if nodes >= end {
        return config.max_collision_visits;
    }
    if nodes <= start {
        return 1;
    }
    let ratio = ((nodes - start) as f32 / (end - start) as f32)
        .powf(config.max_collision_visits_scaling_power.max(0.01));
    mix(config.max_collision_visits, 1, ratio)
}

fn mix(high: i32, low: i32, ratio: f32) -> i32 {
    (low as f32 + (high - low) as f32 * ratio).round() as i32
}

pub(crate) fn init_pending_searchers(config: MctsConfig) -> Option<Arc<AtomicI32>> {
    if config.max_concurrent_searchers == 0 {
        return None;
    }
    Some(Arc::new(AtomicI32::new(config.max_concurrent_searchers)))
}

pub(crate) fn acquire_searcher_slot(pending: &AtomicI32) {
    loop {
        let available = pending.load(Ordering::Acquire);
        if available == 0 {
            std::thread::yield_now();
            continue;
        }
        if pending
            .compare_exchange_weak(available, available - 1, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
    }
}

pub(crate) fn release_searcher_slot(pending: &AtomicI32) {
    pending.fetch_add(1, Ordering::AcqRel);
}

pub(crate) fn ensure_tree_quiescent(tree: &MctsTree) -> Result<(), String> {
    let in_flight = super::worker::total_in_flight_in_tree(tree);
    if in_flight != 0 {
        return Err(format!("search ended with {in_flight} in-flight updates"));
    }
    Ok(())
}
