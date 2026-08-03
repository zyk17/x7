//! 可复用的 stream 搜索状态。
//!
//! 参考：LC3 Overview 的 "Search" / "WatchdogWorker"：
//! <https://lczero.org/dev/lc0/search/lc3/overview/>.
//! tree 替换对照 px0 `NodeTree::ResetToPosition`
//! （`src/search/classic/node.cc:484-520`）。

use std::sync::Arc;
use std::time::Instant;

use xiangqi_core::PositionHistory;

use crate::EnginError;
use crate::callbacks::{ThinkingInfo, Wdl};
use crate::neural::backend::Backend;

use super::{
    GcStats, NodeKey, NodeRepository, Search, SearchConfig, SearchControl, SearchGeneration, SearchLimits, Stats, Tree,
    WorkerPool, best_mate, best_move_filtered, principal_variation_filtered, root_stats, root_variations,
};

/// watchdog 持有的只读 root view，搜索 worker 不持有它。
#[derive(Clone)]
pub(crate) struct WatchdogSnapshot {
    repository: Arc<NodeRepository>,
    root_key: NodeKey,
    root_is_black: bool,
    root_move_filter: Vec<xiangqi_core::Move>,
    multi_pv: usize,
}

impl WatchdogSnapshot {
    /// 对齐 px0 `Search::SendUciInfo`：同一 root 快照输出按根边排序的多条 PV。
    pub fn thinking_infos(&self, stats: Stats, started: Instant) -> Vec<ThinkingInfo> {
        let time = started.elapsed().as_millis() as i64;
        let nodes = stats.completed_playouts as i64;
        let nps = if time == 0 { 0 } else { (nodes * 1000 / time) as i32 };
        let eps = if time == 0 {
            0
        } else {
            (stats.network_evaluations as i64 * 1000 / time) as i32
        };
        let Some(root) = root_stats(&self.repository, self.root_key) else {
            return vec![ThinkingInfo {
                depth: stats.average_depth.min(i32::MAX as u64) as i32,
                seldepth: stats.max_depth.min(i32::MAX as u64) as i32,
                time,
                nodes,
                nps,
                eps,
                ..ThinkingInfo::default()
            }];
        };

        // root node 保存 incoming-edge/走子方视角；UCI 输出当前行棋方视角，故需要翻转
        // 符号（LC3 glossary：`v = w - l`）。
        let wl = (-root.q).clamp(-1.0, 1.0);
        let draw = root.draw.clamp(0.0, 1.0);
        let win = ((1.0 - draw + wl) * 0.5).clamp(0.0, 1.0);
        let loss = ((1.0 - draw - wl) * 0.5).clamp(0.0, 1.0);
        let common = ThinkingInfo {
            depth: stats.average_depth.min(i32::MAX as u64) as i32,
            seldepth: stats.max_depth.min(i32::MAX as u64) as i32,
            time,
            nodes,
            nps,
            eps,
            ..ThinkingInfo::default()
        };
        let variations = root_variations(
            &self.repository,
            self.root_key,
            self.root_is_black,
            &self.root_move_filter,
            self.multi_pv,
        );
        if variations.is_empty() {
            let mate = best_mate(&self.repository, self.root_key, &self.root_move_filter);
            return vec![ThinkingInfo {
                mate,
                score: mate.is_none().then_some((wl * 1000.0).round() as i32),
                wdl: Some(Wdl {
                    w: (win * 1000.0).round() as i32,
                    d: (draw * 1000.0).round() as i32,
                    l: (loss * 1000.0).round() as i32,
                }),
                pv: principal_variation_filtered(
                    &self.repository,
                    self.root_key,
                    self.root_is_black,
                    &self.root_move_filter,
                ),
                ..common
            }];
        }
        let show_multipv = variations.len() > 1;
        variations
            .into_iter()
            .enumerate()
            .map(|(index, variation)| {
                let win = ((1.0 - variation.draw + variation.wl) * 0.5).clamp(0.0, 1.0);
                let loss = ((1.0 - variation.draw - variation.wl) * 0.5).clamp(0.0, 1.0);
                ThinkingInfo {
                    mate: variation.mate,
                    score: variation
                        .mate
                        .is_none()
                        .then_some((variation.wl * 1000.0).round() as i32),
                    wdl: Some(Wdl {
                        w: (win * 1000.0).round() as i32,
                        d: (variation.draw * 1000.0).round() as i32,
                        l: (loss * 1000.0).round() as i32,
                    }),
                    pv: variation.pv,
                    multipv: if show_multipv { (index + 1) as i32 } else { -1 },
                    ..common.clone()
                }
            })
            .collect()
    }
}

/// 外层 Engine 消费的已完成 stream 结果。
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub stats: Stats,
    pub best_move: Option<xiangqi_core::Move>,
    pub principal_variation: Vec<xiangqi_core::Move>,
}

/// 可复用 stream 状态：backend、保留 tree、generation 与 worker pool。
pub(crate) struct SearchState {
    backend: Arc<dyn Backend>,
    tree: Option<Tree>,
    next_generation: u64,
    multi_pv: usize,
    mini_batch_size: usize,
    worker_pool: Arc<WorkerPool>,
}

/// 一次已启动的 stream job；owner 负责运行、drain，并归还 worker。
pub(crate) struct RunningSearch {
    search: Search,
    root_is_black: bool,
    root_move_filter: Vec<xiangqi_core::Move>,
    multi_pv: usize,
}

impl RunningSearch {
    pub fn control(&self) -> SearchControl {
        self.search.control()
    }

    pub fn watchdog_snapshot(&self) -> WatchdogSnapshot {
        WatchdogSnapshot {
            repository: Arc::clone(self.search.repository()),
            root_key: self.search.root_key(),
            root_is_black: self.root_is_black,
            root_move_filter: self.root_move_filter.clone(),
            multi_pv: self.multi_pv,
        }
    }

    pub fn run(mut self, limits: SearchLimits) -> Result<SearchResult, EnginError> {
        let stats = self.search.run_with_limits(limits)?;
        let best_move = best_move_filtered(
            self.search.repository(),
            self.search.root_key(),
            self.root_is_black,
            &self.root_move_filter,
        );
        let principal_variation = principal_variation_filtered(
            self.search.repository(),
            self.search.root_key(),
            self.root_is_black,
            &self.root_move_filter,
        );
        self.search.stop_and_finish();
        Ok(SearchResult {
            stats,
            best_move,
            principal_variation,
        })
    }
}

impl SearchState {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        let config = SearchConfig::default();
        let worker_pool = Arc::new(WorkerPool::new(backend.as_ref(), &config));
        Self {
            backend,
            tree: None,
            next_generation: 0,
            multi_pv: 1,
            mini_batch_size: config.eval_batch_size,
            worker_pool,
        }
    }

    /// 更新 Engine 生命周期参数；下一次 job 若 batch 改变则重建常驻 worker。
    /// 参考 px0 `BaseSearchParams::kMiniBatchSizeId`（`params.cc:178-182,546`）。
    pub fn set_mini_batch_size(&mut self, mini_batch_size: usize) {
        self.mini_batch_size = mini_batch_size;
    }

    /// 更新 Engine 生命周期参数；当前 job 的 watchdog 快照不受影响。
    /// 参考 px0 `BaseSearchParams::GetMultiPv`（`params.h:101`）。
    pub fn set_multi_pv(&mut self, multi_pv: usize) {
        self.multi_pv = multi_pv;
    }

    /// 在 session stop/drain 后写入完整 UCI history。保留前缀复用 tree；无关线路
    /// 重建 repository。参考 px0 `NodeTree::ResetToPosition`
    /// （`src/search/classic/node.cc:484-520`）。
    pub fn set_position(&mut self, history: Arc<PositionHistory>) -> Result<GcStats, EnginError> {
        match self.tree.as_mut() {
            Some(tree) => tree.reset_to_history_after_drain(history),
            None => {
                self.tree = Some(Tree::new(history));
                Ok(GcStats::default())
            }
        }
    }

    /// 启动一个独占 job。
    ///
    /// 只有当前 job drain 并归还 worker 后，下一次 `set_position` 才能
    /// prune 或 rewind 保留树；这是 reservation 的边界。
    pub fn begin_search(&mut self, searchmoves: &[String]) -> Result<RunningSearch, EnginError> {
        let tree = self
            .tree
            .as_ref()
            .ok_or(EnginError::Uci("position is not configured".into()))?;
        // px0 `StringsToMovelist`（`src/search/classic/wrapper.cc:78-100`）：保留合法
        // root 请求；非空列表中没有合法着时拒绝。
        let board = tree.root_history().last().board();
        let legal_moves = board.generate_legal_moves();
        let root_move_filter: Vec<_> = searchmoves
            .iter()
            .filter_map(|move_text| board.parse_move(move_text).ok())
            .filter(|mv| legal_moves.contains(mv))
            .collect();
        if !searchmoves.is_empty() && root_move_filter.is_empty() {
            return Err(EnginError::Uci("No legal searchmoves.".into()));
        }
        self.next_generation = self.next_generation.wrapping_add(1);
        let config = SearchConfig {
            root_move_filter: root_move_filter.clone(),
            eval_batch_size: self.mini_batch_size,
            ..SearchConfig::default()
        };
        if !self.worker_pool.matches_config(self.backend.as_ref(), &config) {
            self.worker_pool = Arc::new(WorkerPool::new(self.backend.as_ref(), &config));
        }
        let search = Search::new_with_tree_in_pool(
            Arc::clone(&self.backend),
            SearchGeneration(self.next_generation),
            tree,
            config,
            Arc::clone(&self.worker_pool),
        );
        let root_is_black = tree.root_history().last().is_black_to_move();
        Ok(RunningSearch {
            search,
            root_is_black,
            root_move_filter,
            multi_pv: self.multi_pv,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xiangqi_core::{GameState, PositionHistory, STARTPOS_FEN};

    use crate::neural::backend::UniformBackend;

    use super::{SearchLimits, SearchState};

    #[test]
    fn state_reuses_tree_and_worker_pool_between_completed_searches() {
        let mut state = SearchState::new(Arc::new(UniformBackend::default()));
        let worker_pool = Arc::clone(&state.worker_pool);
        let start = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        state
            .set_position(Arc::new(PositionHistory::from_positions(start.positions())))
            .expect("set startpos");
        let first = state
            .begin_search(&[])
            .expect("start first")
            .run(SearchLimits {
                max_playouts: Some(8),
                deadline: None,
            })
            .expect("first search");
        let best = first.best_move.expect("best move");

        let next = GameState::from_fen_moves(STARTPOS_FEN, &[best.to_string()]).expect("played move");
        state
            .set_position(Arc::new(PositionHistory::from_positions(next.positions())))
            .expect("advance tree");
        let second = state
            .begin_search(&[])
            .expect("start second")
            .run(SearchLimits {
                max_playouts: Some(4),
                deadline: None,
            })
            .expect("second search");
        assert!(second.stats.completed_playouts >= 4);
        assert!(second.best_move.is_some());
        assert!(Arc::ptr_eq(&worker_pool, &state.worker_pool));
    }

    #[test]
    fn mini_batch_size_rebuilds_workers_for_the_next_job() {
        let mut state = SearchState::new(Arc::new(UniformBackend::default()));
        let start = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        state
            .set_position(Arc::new(PositionHistory::from_positions(start.positions())))
            .expect("position");

        state.set_mini_batch_size(2);
        let first = state.begin_search(&[]).expect("first search");
        assert_eq!(first.search.eval_batch_size(), 2);

        // `setoption` 只更新 Engine 状态；已启动 job 继续使用已有 worker。
        state.set_mini_batch_size(4);
        first
            .run(SearchLimits {
                max_playouts: Some(1),
                deadline: None,
            })
            .expect("first result");

        let second = state.begin_search(&[]).expect("second search");
        assert_eq!(second.search.eval_batch_size(), 4);
        second
            .run(SearchLimits {
                max_playouts: Some(1),
                deadline: None,
            })
            .expect("second result");
    }
}
