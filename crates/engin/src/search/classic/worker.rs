//! px0 `src/search/classic/search.h:201-448` 的 P4 worker。
//!
//! P3 仍由 `SearchSession` 单线程直连 `Backend::evaluate()`。
//! P4 worker 七阶段流水线已可单线程跑通；碰撞/task workers/OOO 完整语义
//! 与 UCI 接线仍属开放项。

use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use xiangqi_core::{GameResult, Move, MoveList, PositionHistory};

use crate::EnginError;

use super::backend::{AddInputResult, Backend, BackendComputation, EvalPosition, EvalResult, EvalTicket};
use super::node::{NodeTree, Terminal};
use super::params::SearchParams;
use super::uct::edge_score;

/// px0 `Search` 中与 worker 相关的计数子集 (`search.h:49-200`)。
#[derive(Debug)]
pub struct WorkerSearchState {
    pub stop: Arc<AtomicBool>,
    pub remaining_playouts: AtomicU64,
    /// `go nodes` 预算；0 表示无限制。
    pub nodes_budget: AtomicU64,
    pub thread_count: AtomicUsize,
    pub shared_collisions: Mutex<Vec<(usize, u32)>>,
    pub total_playouts: AtomicU64,
    pub total_batches: AtomicU64,
    pub network_evaluations: AtomicU64,
    pub cum_depth: AtomicU64,
    pub max_depth: AtomicU16,
}

impl Default for WorkerSearchState {
    fn default() -> Self {
        Self::new(Arc::new(AtomicBool::new(false)), i64::MAX)
    }
}

impl WorkerSearchState {
    pub fn new(stop: Arc<AtomicBool>, remaining_playouts: i64) -> Self {
        Self {
            stop,
            remaining_playouts: AtomicU64::new(remaining_playouts.max(0) as u64),
            nodes_budget: AtomicU64::new(0),
            thread_count: AtomicUsize::new(1),
            shared_collisions: Mutex::new(Vec::new()),
            total_playouts: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
            network_evaluations: AtomicU64::new(0),
            cum_depth: AtomicU64::new(0),
            max_depth: AtomicU16::new(0),
        }
    }

    pub fn remaining_playouts(&self) -> i64 {
        self.remaining_playouts.load(Ordering::Acquire) as i64
    }

    pub fn set_remaining_playouts(&self, value: i64) {
        self.remaining_playouts.store(value.max(0) as u64, Ordering::Release);
    }

    pub fn set_nodes_budget(&self, nodes: u32) {
        self.nodes_budget.store(nodes as u64, Ordering::Release);
    }

    pub fn nodes_budget_reached(&self, root_visits: u32) -> bool {
        let budget = self.nodes_budget.load(Ordering::Acquire);
        budget > 0 && root_visits >= budget as u32
    }
}

/// px0 `SearchWorker::NodeToProcess` (`search.h:288-347`)。
#[derive(Clone, Debug)]
pub struct NodeToProcess {
    pub node_idx: usize,
    pub eval: EvalResult,
    pub multivisit: u32,
    pub maxvisit: u32,
    pub depth: u16,
    pub nn_queried: bool,
    pub is_cache_hit: bool,
    pub is_collision: bool,
    pub moves_to_visit: MoveList,
    pub ooo_completed: bool,
    pub eval_ticket: Option<EvalTicket>,
}

impl NodeToProcess {
    /// px0 `NodeToProcess::Visit` (`search.h:334-336`)。
    pub fn visit(node_idx: usize, depth: u16) -> Self {
        Self {
            node_idx,
            eval: EvalResult::default(),
            multivisit: 1,
            maxvisit: 0,
            depth,
            nn_queried: false,
            is_cache_hit: false,
            is_collision: false,
            moves_to_visit: Vec::new(),
            ooo_completed: false,
            eval_ticket: None,
        }
    }

    /// px0 `NodeToProcess::Collision` (`search.h:321-332`)。
    pub fn collision(node_idx: usize, depth: u16, multivisit: u32, maxvisit: u32) -> Self {
        Self {
            node_idx,
            eval: EvalResult::default(),
            multivisit,
            maxvisit,
            depth,
            nn_queried: false,
            is_cache_hit: false,
            is_collision: true,
            moves_to_visit: Vec::new(),
            ooo_completed: false,
            eval_ticket: None,
        }
    }

    /// px0 `NodeToProcess::IsExtendable` (`search.h:289`)。
    pub const fn is_extendable(&self, is_terminal: bool) -> bool {
        !self.is_collision && !is_terminal
    }

    /// px0 `NodeToProcess::CanEvalOutOfOrder` (`search.h:303-305`)。
    pub const fn can_eval_out_of_order(&self, is_terminal: bool) -> bool {
        self.is_cache_hit || is_terminal
    }
}

/// px0 `SearchWorker` (`src/search/classic/search.h:203-448`)。
pub struct SearchWorker<'a> {
    tree: &'a mut NodeTree,
    backend: &'a dyn Backend,
    params: &'a SearchParams,
    search_state: &'a WorkerSearchState,
    minibatch: Vec<NodeToProcess>,
    computation: Option<Box<dyn BackendComputation>>,
    history: PositionHistory,
    target_minibatch_size: usize,
    max_out_of_order: usize,
    task_workers: i32,
    number_out_of_order: usize,
    played_history_len: usize,
}

impl<'a> SearchWorker<'a> {
    /// px0 `SearchWorker::SearchWorker` (`search.h:205-233`)。
    pub fn new(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
    ) -> Self {
        Self::from_parts(tree, backend, params, search_state)
    }

    pub fn with_context(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
    ) -> Self {
        Self::from_parts(tree, backend, params, search_state)
    }

    fn from_parts(
        tree: &'a mut NodeTree,
        backend: &'a dyn Backend,
        params: &'a SearchParams,
        search_state: &'a WorkerSearchState,
    ) -> Self {
        let mut target_minibatch_size = if params.minibatch_size > 0 {
            params.minibatch_size as usize
        } else {
            backend.attributes().recommended_batch_size
        };
        if target_minibatch_size == 0 {
            target_minibatch_size = 1;
        }
        let mut task_workers = params.task_workers_per_search_worker;
        if task_workers < 0 && backend.attributes().runs_on_cpu {
            task_workers = 0;
        }
        let max_out_of_order = std::cmp::max(
            1,
            (params.max_out_of_order_evals_factor * target_minibatch_size as f32) as usize,
        );
        Self {
            history: tree.history().clone(),
            played_history_len: tree.history().len(),
            tree,
            backend,
            params,
            search_state,
            minibatch: Vec::new(),
            computation: None,
            target_minibatch_size,
            max_out_of_order,
            task_workers,
            number_out_of_order: 0,
        }
    }

    /// px0 `SearchWorker::ExecuteOneIteration` (`search.cc:1142-1231`)。
    pub fn execute_one_iteration(&mut self) -> Result<(), EnginError> {
        let root = self.tree.current_head();
        if self.search_state.nodes_budget_reached(self.tree.node(root).n()) {
            self.search_state.stop.store(true, Ordering::Release);
            return Ok(());
        }
        self.initialize_iteration()?;
        self.gather_minibatch()?;
        self.collect_collisions()?;
        self.maybe_prefetch_into_cache()?;
        self.run_nn_computation()?;
        self.fetch_minibatch_results()?;
        self.do_backup_update()?;
        self.update_counters()
    }

    /// 单线程测试入口：重复执行 iteration 直到 root N 达标。
    pub fn run_until_root_visits(&mut self, target: u32) -> Result<(), EnginError> {
        while self.tree.node(self.tree.current_head()).n() < target {
            if self.search_state.stop.load(Ordering::Acquire) {
                break;
            }
            self.execute_one_iteration()?;
        }
        Ok(())
    }

    /// px0 `SearchWorker::InitializeIteration` (`search.cc:1233-1266`)。
    pub fn initialize_iteration(&mut self) -> Result<(), EnginError> {
        self.computation = Some(self.backend.create_computation()?);
        self.minibatch.clear();
        self.minibatch.reserve(2 * self.target_minibatch_size);
        self.history = self.tree.history().clone();
        self.played_history_len = self.history.len();
        Ok(())
    }

    /// px0 `SearchWorker::GatherMinibatch` (`search.cc:1268-1363`) 单线程子集。
    pub fn gather_minibatch(&mut self) -> Result<(), EnginError> {
        if self.task_workers > 0 {
            return Err(EnginError::PortIncomplete(
                "P4 SearchWorker::GatherMinibatch task workers",
            ));
        }
        let root = self.tree.current_head();
        let cur_n = self.tree.node(root).n();
        let remaining_n = self.search_state.remaining_playouts();
        let nodes = cur_n.min(remaining_n.max(0) as u32) as i64;
        let collisions_left = self.params.collisions_left(nodes);
        self.number_out_of_order = 0;

        let mut minibatch_size = 0usize;
        while minibatch_size < self.target_minibatch_size && self.number_out_of_order < self.max_out_of_order {
            if minibatch_size > 0
                && self
                    .computation
                    .as_ref()
                    .map_or(0, |computation| computation.used_batch_size())
                    == 0
            {
                return Ok(());
            }

            let new_start = self.minibatch.len();
            let pick_budget = collisions_left
                .min(self.target_minibatch_size as i32 - minibatch_size as i32)
                .min(self.max_out_of_order as i32 - self.number_out_of_order as i32);
            self.pick_nodes_to_extend(pick_budget.max(0) as u32)?;
            let mut picked_visits = 0usize;
            for item in &self.minibatch[new_start..] {
                if !item.is_collision {
                    minibatch_size += 1;
                    picked_visits += 1;
                }
            }
            self.process_picked_task(new_start, self.minibatch.len())?;

            // px0 consumes collision budget and can increase multivisit here.
            // That translation is still pending; do not spin on a collision
            // that produced no independently evaluable leaf.
            if picked_visits == 0 {
                return Ok(());
            }

            let mut some_ooo = false;
            for item in &self.minibatch[new_start..] {
                if item.ooo_completed {
                    some_ooo = true;
                    break;
                }
            }
            if some_ooo {
                let mut i = self.minibatch.len();
                while i > new_start {
                    i -= 1;
                    if self.minibatch[i].is_collision {
                        let node_idx = self.minibatch[i].node_idx;
                        let multivisit = self.minibatch[i].multivisit;
                        let mut node = node_idx;
                        while let Some(parent) = self.tree.node(node).parent() {
                            self.tree.node_mut(parent).cancel_score_update(multivisit);
                            node = parent;
                            if node == self.tree.current_head() {
                                break;
                            }
                        }
                        self.minibatch.remove(i);
                    } else if self.minibatch[i].ooo_completed {
                        // px0 backs up completed out-of-order entries while
                        // reconciling collisions in GatherMinibatch
                        // (`search.cc:1372-1393`), not in ProcessPickedTask.
                        let item = self.minibatch[i].clone();
                        self.do_backup_update_single_node(&item);
                        self.minibatch.remove(i);
                        minibatch_size = minibatch_size.saturating_sub(1);
                        self.number_out_of_order += 1;
                    }
                }
            }

            for item in &self.minibatch[new_start..] {
                if item.is_collision && self.search_state.stop.load(Ordering::Acquire) {
                    return Ok(());
                }
            }
            if minibatch_size >= self.target_minibatch_size {
                break;
            }
        }
        Ok(())
    }

    /// px0 `SearchWorker::PickNodesToExtend` (`search.cc:1485-1508`) 单线程子集。
    fn pick_nodes_to_extend(&mut self, collision_limit: u32) -> Result<(), EnginError> {
        if collision_limit == 0 {
            return Ok(());
        }
        if self.task_workers > 0 {
            return Err(EnginError::PortIncomplete("P4 PickNodesToExtend task workers"));
        }
        // The full px0 routine batches `collision_limit` visits in one tree
        // walk. Until `PickNodesToExtendTask` is translated, one call may only
        // reserve one visit. The gather loop deliberately invokes it again to
        // fill a single-worker minibatch.
        self.pick_nodes_to_extend_single()
    }

    /// px0 `PickNodesToExtendTask` (`search.cc:1551-1897`) 的单条 selection 路径。
    ///
    /// 一次调用只创建一个 visit。`GatherMinibatch` 反复调用本函数填满 batch，
    /// 因此不能把 `collision_limit` 整体计入 root 的 in-flight。
    fn pick_nodes_to_extend_single(&mut self) -> Result<(), EnginError> {
        let root = self.tree.current_head();
        let mut node_idx = root;
        let mut moves_to_visit = MoveList::new();
        let mut depth = 0u16;
        let mut is_root = true;
        let mut reserved = Vec::new();

        loop {
            let n = self.tree.node(node_idx).n();
            let terminal = self.tree.node(node_idx).is_terminal();
            if n == 0 || terminal {
                if is_root && self.tree.node_mut(node_idx).try_start_score_update() {
                    reserved.push(node_idx);
                    let mut item = NodeToProcess::visit(node_idx, depth + 1);
                    item.moves_to_visit = moves_to_visit;
                    self.minibatch.push(item);
                } else {
                    self.minibatch.push(NodeToProcess::collision(node_idx, depth + 1, 1, 0));
                }
                return Ok(());
            }

            if is_root {
                if !self.tree.node_mut(node_idx).try_start_score_update() {
                    self.minibatch.push(NodeToProcess::collision(node_idx, depth + 1, 1, 0));
                    return Ok(());
                }
                reserved.push(node_idx);
            }

            let best_edge = self.select_best_edge(node_idx, is_root)?;
            let mv = self.tree.node(node_idx).edge(best_edge).mv;
            let child_idx = match self.tree.node(node_idx).child(best_edge) {
                Some(idx) => idx,
                None => self.tree.arena_mut().spawn_child(node_idx, best_edge),
            };
            if !self.tree.node_mut(child_idx).try_start_score_update() {
                for reserved_node in reserved.into_iter().rev() {
                    self.tree.node_mut(reserved_node).cancel_score_update(1);
                }
                self.minibatch
                    .push(NodeToProcess::collision(child_idx, depth + 2, 1, 0));
                return Ok(());
            }
            reserved.push(child_idx);

            let child_n = self.tree.node(child_idx).n();
            let child_terminal = self.tree.node(child_idx).is_terminal();
            if child_n == 0 || child_terminal {
                let mut item = NodeToProcess::visit(child_idx, depth + 2);
                item.moves_to_visit = moves_to_visit;
                item.moves_to_visit.push(mv);
                self.minibatch.push(item);
                return Ok(());
            }

            moves_to_visit.push(mv);
            node_idx = child_idx;
            is_root = false;
            depth += 1;
        }
    }

    fn select_best_edge(&self, parent_idx: usize, is_root: bool) -> Result<usize, EnginError> {
        let draw_score = self.params.draw_score;
        let parent = self.tree.node(parent_idx);
        let mut best_edge = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for edge_idx in 0..parent.num_edges() {
            let child = parent.child(edge_idx).map(|idx| self.tree.node(idx));
            let score = edge_score(
                parent,
                edge_idx,
                child,
                self.tree.arena(),
                self.params,
                is_root,
                draw_score,
            );
            if score > best_score {
                best_score = score;
                best_edge = edge_idx;
            }
        }
        Ok(best_edge)
    }

    /// px0 `SearchWorker::ProcessPickedTask` (`search.cc:1423-1462`)。
    fn process_picked_task(&mut self, start_idx: usize, end_idx: usize) -> Result<(), EnginError> {
        let mut nn_inputs = Vec::new();
        for i in start_idx..end_idx {
            let node_idx = self.minibatch[i].node_idx;
            let depth = self.minibatch[i].depth;
            let moves_to_visit = self.minibatch[i].moves_to_visit.clone();
            let is_terminal = self.tree.node(node_idx).is_terminal();
            if self.minibatch[i].is_extendable(is_terminal) {
                self.extend_node(node_idx, depth, &moves_to_visit)?;
                if !self.tree.node(node_idx).is_terminal() {
                    nn_inputs.push((i, self.history.positions().to_vec(), {
                        self.tree
                            .node(node_idx)
                            .edges()
                            .iter()
                            .map(|edge| edge.mv)
                            .collect::<MoveList>()
                    }));
                }
            }
            if self.params.out_of_order_eval
                && self.minibatch[i].can_eval_out_of_order(self.tree.node(node_idx).is_terminal())
            {
                self.fetch_single_node_result(i)?;
                self.minibatch[i].ooo_completed = true;
            }
        }
        let computation = self
            .computation
            .as_mut()
            .ok_or(EnginError::PortIncomplete("P4 ProcessPickedTask without computation"))?;
        for (index, positions, legal_moves) in nn_inputs {
            let (result, ticket) = computation.add_input(EvalPosition { positions, legal_moves })?;
            self.minibatch[index].nn_queried = true;
            self.minibatch[index].is_cache_hit = result == AddInputResult::FetchedImmediately;
            self.minibatch[index].eval_ticket = Some(ticket);
        }
        Ok(())
    }

    /// px0 `SearchWorker::ExtendNode` (`search.cc:1899-1974`)。
    fn extend_node(&mut self, node_idx: usize, _depth: u16, moves_to_node: &[Move]) -> Result<(), EnginError> {
        let root = self.tree.current_head();
        self.history.trim(self.played_history_len);
        for mv in moves_to_node {
            self.history.append(*mv);
        }
        let board = self.history.last().board();
        let legal_moves = board.generate_legal_moves();
        if legal_moves.is_empty() {
            self.tree.make_terminal(
                node_idx,
                if self.history.is_black_to_move() {
                    GameResult::WhiteWon
                } else {
                    GameResult::BlackWon
                },
                0.0,
                Terminal::EndOfGame,
            );
            return Ok(());
        }
        if node_idx != root {
            if self.history.last().repetitions() >= 2 {
                self.tree
                    .make_terminal(node_idx, self.history.rule_judge(), 0.0, Terminal::EndOfGame);
                return Ok(());
            }
            if !board.has_mating_material() || self.history.last().rule60_ply() >= 120 {
                self.tree
                    .make_terminal(node_idx, GameResult::Draw, 0.0, Terminal::EndOfGame);
                return Ok(());
            }
        }
        self.tree.node_mut(node_idx).create_edges(&legal_moves);
        Ok(())
    }

    /// px0 `SearchWorker::CollectCollisions` (`search.cc:1977-1987`)。
    pub fn collect_collisions(&mut self) -> Result<(), EnginError> {
        for item in &self.minibatch {
            if item.is_collision {
                self.search_state
                    .shared_collisions
                    .lock()
                    .expect("collisions lock")
                    .push((item.node_idx, item.multivisit));
            }
        }
        Ok(())
    }

    /// px0 `SearchWorker::MaybePrefetchIntoCache` (`search.cc:1989-2007`)。
    pub fn maybe_prefetch_into_cache(&mut self) -> Result<(), EnginError> {
        if self.search_state.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        let used = self
            .computation
            .as_ref()
            .map_or(0, |computation| computation.used_batch_size());
        if used == 0 || used >= self.params.max_prefetch_batch as usize {
            return Ok(());
        }
        let budget = self.params.max_prefetch_batch as usize - used;
        let root = self.tree.current_head();
        self.prefetch_into_cache(root, budget, false)?;
        Ok(())
    }

    /// px0 `PrefetchIntoCache` (`search.cc:2010-2099`) 叶子子集。
    fn prefetch_into_cache(
        &mut self,
        node_idx: usize,
        budget: usize,
        _is_odd_depth: bool,
    ) -> Result<usize, EnginError> {
        if budget == 0 {
            return Ok(0);
        }
        let node = self.tree.node(node_idx);
        if node.n_started() == 0 {
            if self
                .backend
                .cached_evaluation(&EvalPosition {
                    positions: self.history.positions().to_vec(),
                    legal_moves: Vec::new(),
                })
                .is_some()
            {
                return Ok(1);
            }
            let legal_moves = self.history.last().board().generate_legal_moves();
            if let Some(computation) = self.computation.as_mut() {
                let _ = computation.add_input(EvalPosition {
                    positions: self.history.positions().to_vec(),
                    legal_moves,
                })?;
            }
            return Ok(1);
        }
        if node.n() == 0 || node.is_terminal() {
            return Ok(0);
        }
        Ok(0)
    }

    /// px0 `SearchWorker::RunNNComputation` (`search.cc:2103-2107`)。
    pub fn run_nn_computation(&mut self) -> Result<(), EnginError> {
        if let Some(computation) = self.computation.as_mut() {
            if computation.used_batch_size() > 0 {
                computation.compute_blocking()?;
            }
        }
        Ok(())
    }

    /// px0 `SearchWorker::FetchMinibatchResults` (`search.cc:2109-2156`)。
    pub fn fetch_minibatch_results(&mut self) -> Result<(), EnginError> {
        for i in 0..self.minibatch.len() {
            self.fetch_single_node_result(i)?;
        }
        Ok(())
    }

    /// px0 `SearchWorker::FetchSingleNodeResult` (`search.cc:2117-2154`)。
    fn fetch_single_node_result(&mut self, index: usize) -> Result<(), EnginError> {
        if self.minibatch[index].is_collision {
            return Ok(());
        }
        let node_idx = self.minibatch[index].node_idx;
        if !self.minibatch[index].nn_queried {
            self.minibatch[index].eval = EvalResult {
                wl: self.tree.node(node_idx).wl(),
                d: self.tree.node(node_idx).d(),
                m: self.tree.node(node_idx).m(),
                policies: Vec::new(),
            };
            return Ok(());
        }
        let ticket = self.minibatch[index]
            .eval_ticket
            .ok_or(EnginError::PortIncomplete("P4 FetchSingleNodeResult missing ticket"))?;
        let mut eval = self
            .computation
            .as_mut()
            .ok_or(EnginError::PortIncomplete(
                "P4 FetchSingleNodeResult without computation",
            ))?
            .take_result(ticket)?;
        eval.wl = -eval.wl;
        if self.tree.node(node_idx).n() == 0 {
            for (edge_idx, policy) in eval.policies.iter().enumerate() {
                self.tree.node_mut(node_idx).edge_mut(edge_idx).set_p(*policy);
            }
            // px0 sorts the just-initialized policy before any child node can
            // be spawned (`node.cc:291-298`, `search.cc:2145-2153`).
            self.tree.node_mut(node_idx).sort_edges();
        }
        self.minibatch[index].eval = eval;
        Ok(())
    }

    /// px0 `SearchWorker::DoBackupUpdate` (`search.cc:2158-2258`) 单线程子集。
    pub fn do_backup_update(&mut self) -> Result<(), EnginError> {
        let mut work_done = self.number_out_of_order > 0;
        let items: Vec<_> = self
            .minibatch
            .iter()
            .filter(|item| !item.is_collision)
            .cloned()
            .collect();
        for item in items {
            self.do_backup_update_single_node(&item);
            work_done = true;
        }
        if work_done {
            self.search_state
                .shared_collisions
                .lock()
                .expect("collisions lock")
                .clear();
            self.search_state.total_batches.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    /// px0 `SearchWorker::DoBackupUpdateSingleNode` (`search.cc:2175-2258`) 子集。
    fn do_backup_update_single_node(&mut self, item: &NodeToProcess) {
        let mut node_idx = item.node_idx;
        let mut v = item.eval.wl;
        let mut d = item.eval.d;
        let mut m = item.eval.m;
        let root = self.tree.current_head();
        loop {
            if self.tree.node(node_idx).is_terminal() {
                v = self.tree.node(node_idx).wl();
                d = self.tree.node(node_idx).d();
                m = self.tree.node(node_idx).m();
            }
            self.tree
                .node_mut(node_idx)
                .finalize_score_update(v, d, m, item.multivisit);
            if node_idx == root {
                break;
            }
            let parent = self.tree.node(node_idx).parent().expect("non-root has parent");
            node_idx = parent;
            v = -v;
            m += 1.0;
        }
        self.search_state
            .total_playouts
            .fetch_add(item.multivisit as u64, Ordering::AcqRel);
        if item.nn_queried && !item.is_cache_hit {
            self.search_state.network_evaluations.fetch_add(1, Ordering::AcqRel);
        }
        self.search_state
            .cum_depth
            .fetch_add(item.depth as u64 * item.multivisit as u64, Ordering::AcqRel);
        loop {
            let current = self.search_state.max_depth.load(Ordering::Acquire);
            if item.depth <= current {
                break;
            }
            if self
                .search_state
                .max_depth
                .compare_exchange(current, item.depth, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
    }

    /// px0 `SearchWorker::UpdateCounters` (`search.cc:2331-2364`) 子集。
    pub fn update_counters(&mut self) -> Result<(), EnginError> {
        let _ = self.search_state.stop.load(Ordering::Acquire);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::sync::Once;

    use xiangqi_core::{initialize_magic_bitboards, GameResult, GameState, STARTPOS_FEN};

    use super::*;
    use crate::search::classic::UniformBackend;

    static INIT: Once = Once::new();

    fn ensure_init() {
        INIT.call_once(initialize_magic_bitboards);
    }

    #[test]
    fn out_of_order_terminal_stays_explicit() {
        ensure_init();
        let mut tree = NodeTree::default();
        let state = GameState::from_fen_moves(STARTPOS_FEN, &[] as &[&str]).expect("startpos");
        tree.reset_to_position(&state.startpos, &state.moves);
        let root = tree.current_head();
        tree.make_terminal(root, GameResult::Draw, 0.0, Terminal::EndOfGame);
        let backend = UniformBackend::default();
        let params = SearchParams {
            out_of_order_eval: true,
            ..SearchParams::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let search_state = WorkerSearchState::new(stop, i64::MAX);
        let mut worker = SearchWorker::new(&mut tree, &backend, &params, &search_state);
        worker.initialize_iteration().expect("init");
        // `PickNodesToExtendTask` always reserves the path before handing an
        // item to ProcessPickedTask (`px0 search.cc:1613-1636`).  Keep this
        // focused OOO test on that post-selection contract.
        assert!(worker.tree.node_mut(root).try_start_score_update());
        worker.minibatch.push(NodeToProcess::visit(root, 1));
        worker.process_picked_task(0, 1).expect("ooo terminal");
        assert!(worker.minibatch[0].ooo_completed);
    }
}
