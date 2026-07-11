//! px0 `src/search/classic/search.h:49-260`、`search.cc:426-808,1900-2258`、`wrapper.cc:53-141`。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use xiangqi_core::{GameResult, GameState, Move, MoveList, PositionHistory};

use crate::callbacks::{BestMoveInfo, ThinkingInfo};
use crate::search::SearchBase;
use crate::uci_loop::GoParams;
use crate::EnginError;

use super::backend::{Backend, EvalResult};
use super::node::{NodeTree, Terminal};
use super::params::SearchParams;

fn compute_cpuct(params: &SearchParams, n: u32, is_root: bool) -> f32 {
    let init = params.cpuct(is_root);
    let k = params.cpuct_factor(is_root);
    let base = params.cpuct_base(is_root);
    if k == 0.0 {
        init
    } else {
        init + k * ((n as f32 + base) / base).ln()
    }
}

fn get_fpu(
    params: &SearchParams,
    node: &super::node::Node,
    arena: &super::node::NodeArena,
    is_root: bool,
    draw_score: f32,
) -> f32 {
    let visited_pol = if is_root {
        1.0
    } else {
        node.visited_policy(arena).max(1.0)
    };
    let value = params.fpu_value(is_root);
    if params.fpu_absolute(is_root) {
        value
    } else {
        -node.q(draw_score) - value * visited_pol.sqrt()
    }
}

fn edge_score(
    parent: &super::node::Node,
    edge_idx: usize,
    child: Option<&super::node::Node>,
    arena: &super::node::NodeArena,
    params: &SearchParams,
    is_root: bool,
    draw_score: f32,
) -> f32 {
    let edge = parent.edge(edge_idx);
    let cpuct = compute_cpuct(params, parent.n(), is_root);
    let u_coeff = cpuct * (parent.children_visits().max(1) as f32).sqrt();
    let fpu = get_fpu(params, parent, arena, is_root, draw_score);
    let q = child
        .filter(|node| node.n() > 0)
        .map(|node| node.q(draw_score))
        .unwrap_or(fpu);
    q + u_coeff * edge.get_p() / (1.0 + child.map(|node| node.n_started()).unwrap_or(0) as f32)
}

/// px0 `Search` 单线程子集。
pub struct SearchSession<'a> {
    pub tree: &'a mut NodeTree,
    pub backend: &'a dyn Backend,
    pub params: &'a SearchParams,
    pub stop: Arc<AtomicBool>,
    pub target_nodes: Option<u32>,
    pub deadline: Option<Instant>,
}

impl<'a> SearchSession<'a> {
    /// px0 `SearchWorker::ExecuteOneIteration` 单线程路径。
    pub fn execute_one_iteration(&mut self) {
        let root = self.tree.current_head();
        // px0 starts a score update for every selected node, including root.
        // Backup below retires this matching in-flight visit.
        if !self.tree.node_mut(root).try_start_score_update() {
            return;
        }
        let path = self.pick_path(root);
        let leaf = *path.last().expect("path has root");
        let history = self.history_at(&path);
        if self.tree.node(leaf).num_edges() == 0 && !self.tree.node(leaf).is_terminal() {
            self.extend_node(leaf, root, &history);
        }
        if self.tree.node(leaf).is_terminal() {
            let eval = EvalResult {
                wl: self.tree.node(leaf).wl(),
                d: self.tree.node(leaf).d(),
                m: self.tree.node(leaf).m(),
                policies: Vec::new(),
            };
            self.backup(&path, &eval);
            return;
        }
        let legal_moves: MoveList = self.tree.node(leaf).edges().iter().map(|edge| edge.mv).collect();
        let eval = self.backend.evaluate(&history, &legal_moves);
        if self.tree.node(leaf).n() == 0 {
            for (idx, policy) in eval.policies.iter().enumerate() {
                self.tree.node_mut(leaf).edge_mut(idx).set_p(*policy);
            }
        }
        self.backup(&path, &eval);
    }

    pub fn run_until_stopped(&mut self) {
        let mut ran_iteration = false;
        loop {
            if ran_iteration
                && (self.stop.load(Ordering::Acquire)
                    || self.deadline.is_some_and(|deadline| Instant::now() >= deadline))
            {
                break;
            }
            if let Some(target) = self.target_nodes {
                let root_n = self.tree.node(self.tree.current_head()).n();
                if ran_iteration && root_n >= target {
                    break;
                }
            }
            self.execute_one_iteration();
            ran_iteration = true;
        }
    }

    /// px0 `Search::GetBestMove` / `GetBestChildNoTemperature` (`search.cc:643-808`)。
    pub fn best_move(&self) -> (Move, Move) {
        let root = self.tree.current_head();
        let root_is_black = self.tree.history().is_black_to_move();
        let best_edge = self.best_child_edge(root);
        let best = best_edge
            .map(|idx| self.tree.node(root).edge(idx).mv)
            .unwrap_or(Move::NULL);
        let ponder = best_edge
            .and_then(|idx| {
                let child = self.tree.node(root).child(idx)?;
                self.best_child_edge(child)
                    .map(|ponder_idx| self.tree.node(child).edge(ponder_idx).mv)
            })
            .unwrap_or(Move::NULL);
        // Edges are stored from the current player's mirrored perspective.
        // UCI output is always absolute board coordinates.
        (
            if root_is_black { best.flip() } else { best },
            if root_is_black { ponder } else { ponder.flip() },
        )
    }

    fn best_child_edge(&self, parent: usize) -> Option<usize> {
        let mut best_idx = None;
        let mut best_n = 0;
        let mut best_q = f32::NEG_INFINITY;
        let mut best_p = f32::NEG_INFINITY;
        for edge_idx in 0..self.tree.node(parent).num_edges() {
            let n = self
                .tree
                .node(parent)
                .child(edge_idx)
                .map(|child| self.tree.node(child).n())
                .unwrap_or(0);
            let q = self
                .tree
                .node(parent)
                .child(edge_idx)
                .map(|child| self.tree.node(child).q(self.params.draw_score))
                .unwrap_or(0.0);
            let p = self.tree.node(parent).edge(edge_idx).get_p();
            let better = n > best_n || (n == best_n && n > 0 && q > best_q) || (n == best_n && n == 0 && p > best_p);
            if better {
                best_idx = Some(edge_idx);
                best_n = n;
                best_q = q;
                best_p = p;
            }
        }
        best_idx
    }

    fn pick_path(&mut self, root: usize) -> Vec<usize> {
        let mut path = vec![root];
        loop {
            let node_idx = *path.last().expect("non-empty path");
            let node = self.tree.node(node_idx);
            if node.is_terminal() || node.num_edges() == 0 {
                break;
            }
            let is_root = node_idx == root;
            let mut best_edge = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for edge_idx in 0..node.num_edges() {
                let child = node.child(edge_idx).map(|idx| self.tree.node(idx));
                let score = edge_score(
                    node,
                    edge_idx,
                    child,
                    self.tree.arena(),
                    self.params,
                    is_root,
                    self.params.draw_score,
                );
                if score > best_score {
                    best_score = score;
                    best_edge = edge_idx;
                }
            }
            let child_idx = match node.child(best_edge) {
                Some(idx) => idx,
                None => self.tree.arena_mut().spawn_child(node_idx, best_edge),
            };
            if !self.tree.node_mut(child_idx).try_start_score_update() {
                break;
            }
            path.push(child_idx);
        }
        path
    }

    fn history_at(&self, path: &[usize]) -> PositionHistory {
        let mut history = self.tree.history().clone();
        for child_idx in path.iter().copied().skip(1) {
            let parent_idx = self.tree.node(child_idx).parent().expect("non-root has parent");
            let edge_idx = self.tree.node(child_idx).edge_index() as usize;
            history.append(self.tree.node(parent_idx).edge(edge_idx).mv);
        }
        history
    }

    /// px0 `SearchWorker::ExtendNode` (`search.cc:1900-1974`) 子集。
    fn extend_node(&mut self, node_idx: usize, root_idx: usize, history: &PositionHistory) {
        let board = history.last().board();
        let legal_moves = board.generate_legal_moves();
        if legal_moves.is_empty() {
            self.tree.make_terminal(
                node_idx,
                if history.is_black_to_move() {
                    GameResult::WhiteWon
                } else {
                    GameResult::BlackWon
                },
                0.0,
                Terminal::EndOfGame,
            );
            return;
        }
        if node_idx != root_idx {
            if history.last().repetitions() >= 2 {
                self.tree
                    .make_terminal(node_idx, history.rule_judge(), 0.0, Terminal::EndOfGame);
                return;
            }
            if !board.has_mating_material() || history.last().rule60_ply() >= 120 {
                self.tree
                    .make_terminal(node_idx, GameResult::Draw, 0.0, Terminal::EndOfGame);
                return;
            }
        }
        self.tree.node_mut(node_idx).create_edges(&legal_moves);
    }

    /// px0 `SearchWorker::DoBackupUpdateSingleNode` (`search.cc:2175-2234`) 子集。
    fn backup(&mut self, path: &[usize], eval: &EvalResult) {
        let mut v = eval.wl;
        let mut d = eval.d;
        let mut m = eval.m;
        for &node_idx in path.iter().rev() {
            if self.tree.node(node_idx).is_terminal() {
                v = self.tree.node(node_idx).wl();
                d = self.tree.node(node_idx).d();
                m = self.tree.node(node_idx).m();
            }
            self.tree.node_mut(node_idx).finalize_score_update(v, d, m, 1);
            v = -v;
            m += 1.0;
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchOutput {
    pub bestmove: BestMoveInfo,
    pub info: ThinkingInfo,
}

struct SearchSharedState {
    tree: NodeTree,
    params: SearchParams,
}

/// px0 `classic::ClassicSearch` wrapper (`wrapper.cc:53-141`)。
pub struct ClassicSearch {
    state: Arc<Mutex<SearchSharedState>>,
    backend: Arc<dyn Backend>,
    stop: Arc<AtomicBool>,
    pub outputs: Vec<SearchOutput>,
}

impl ClassicSearch {
    pub fn new(backend: Box<dyn Backend>) -> Self {
        Self {
            state: Arc::new(Mutex::new(SearchSharedState {
                tree: NodeTree::default(),
                params: SearchParams::default(),
            })),
            backend: Arc::from(backend),
            stop: Arc::new(AtomicBool::new(false)),
            outputs: Vec::new(),
        }
    }

    pub fn total_root_visits(&self) -> u32 {
        let guard = self.state.lock().expect("tree lock");
        guard.tree.node(guard.tree.current_head()).n()
    }

    /// 测试入口：同步跑完固定 nodes 并返回 trace。
    pub fn run_blocking_nodes(&mut self, nodes: u32) -> (Move, u32) {
        self.stop.store(false, Ordering::Release);
        let params = self.state.lock().expect("tree lock").params.clone();
        let mut guard = self.state.lock().expect("tree lock");
        let mut session = SearchSession {
            tree: &mut guard.tree,
            backend: self.backend.as_ref(),
            params: &params,
            stop: Arc::clone(&self.stop),
            target_nodes: Some(nodes),
            deadline: None,
        };
        session.run_until_stopped();
        let (best, _) = session.best_move();
        let visits = guard.tree.node(guard.tree.current_head()).n();
        (best, visits)
    }

    fn run_sync(&mut self, target_nodes: Option<u32>, movetime: Option<Duration>) -> Result<(), EnginError> {
        self.outputs.clear();
        self.stop.store(false, Ordering::Release);
        let params = self.state.lock().expect("tree lock").params.clone();
        let mut guard = self.state.lock().expect("tree lock");
        let mut session = SearchSession {
            tree: &mut guard.tree,
            backend: self.backend.as_ref(),
            params: &params,
            stop: Arc::clone(&self.stop),
            target_nodes,
            deadline: movetime.map(|duration| Instant::now() + duration),
        };
        session.run_until_stopped();
        let (best, ponder) = session.best_move();
        let root_n = guard.tree.node(guard.tree.current_head()).n();
        if !best.is_null() {
            let mut bestmove = BestMoveInfo::new(best);
            bestmove.ponder = ponder;
            self.outputs.push(SearchOutput {
                bestmove,
                info: ThinkingInfo {
                    depth: 1,
                    nodes: root_n as i64,
                    multipv: 1,
                    ..ThinkingInfo::default()
                },
            });
        }
        Ok(())
    }
}

impl SearchBase for ClassicSearch {
    fn new_game(&mut self) -> Result<(), EnginError> {
        let mut guard = self.state.lock().expect("tree lock");
        guard.tree = NodeTree::default();
        self.outputs.clear();
        Ok(())
    }

    fn set_position(&mut self, state: &GameState) -> Result<(), EnginError> {
        let mut guard = self.state.lock().expect("tree lock");
        guard.tree.reset_to_position(&state.startpos, &state.moves);
        Ok(())
    }

    /// P3：单线程 `go nodes` / `go movetime`；完整 stopper 语义后续翻译。
    fn start_search(&mut self, params: &GoParams) -> Result<(), EnginError> {
        if params.infinite {
            return Err(EnginError::PortIncomplete("infinite search stopper"));
        }
        let target_nodes = match params.nodes {
            Some(nodes) if nodes > 0 => Some(nodes as u32),
            Some(_) => return Err(EnginError::Uci("go nodes must be positive".into())),
            None => None,
        };
        let movetime = match params.movetime {
            Some(movetime) if movetime >= 0 => Some(Duration::from_millis(movetime as u64)),
            Some(_) => return Err(EnginError::Uci("go movetime must be non-negative".into())),
            None => None,
        };
        if target_nodes.is_some() || movetime.is_some() {
            return self.run_sync(target_nodes, movetime);
        }
        Err(EnginError::PortIncomplete("time-based search stopper"))
    }

    fn start_clock(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn wait_search(&mut self) -> Result<(), EnginError> {
        Ok(())
    }

    fn stop_search(&mut self) -> Result<(), EnginError> {
        self.stop.store(true, Ordering::Release);
        Ok(())
    }

    fn abort_search(&mut self) -> Result<(), EnginError> {
        self.stop.store(true, Ordering::Release);
        Ok(())
    }
}
