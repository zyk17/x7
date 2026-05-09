//! Alpha-Beta：**静止搜索（吃子 / 应将）**、置换表、杀手着法、MVV-LVA、根节点 policy logit 排序。
//! 面向 NN policy/value + 传统剪枝；不必对齐仅有 eval 的参考引擎。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use xiangqi_core::movegen::{ExtMove, GenType};
use xiangqi_core::types::{Bound, Move, Piece, MAX_MOVES, PIECE_VALUE};
use xiangqi_core::{generate, move_to_uci, Position};

use crate::eval::{evaluate_leaf, terminal_score};
use crate::policy_onnx::PolicyOnnx;
use crate::tt::TranspositionTable;

/// 附加约束：思考截止时刻、节点上限（可与 UCI `movetime` / `nodes` 对应）。
#[derive(Clone, Copy, Default)]
pub struct SearchLimits {
    pub deadline: Option<Instant>,
    pub max_nodes: Option<u64>,
}

impl SearchLimits {
    pub fn none() -> Self {
        Self::default()
    }
}

/// 单次根搜索 / 迭代加深共享句柄（policy、词表、TT、`stop`），压缩 `root_search*` 参数列表。
pub struct RootSearchShared<'a> {
    pub policy: &'a Arc<Mutex<Option<PolicyOnnx>>>,
    pub vocab: &'a HashMap<String, usize>,
    pub vocab_size: usize,
    pub tt: &'a mut TranspositionTable,
    pub stop: Option<&'a AtomicBool>,
}

pub struct RootSearchResult {
    /// 主搜索层数（不含静止延伸）；迭代加深时为最后一轮达到的层数。
    pub main_depth: u32,
    pub best_uci: String,
    pub score_cp: i32,
    /// 迭代加深时为累计节点数。
    pub nodes: u64,
    /// 到达的最大选择性深度（含静止延伸），用于 UCI `seldepth`。
    pub seldepth: u32,
}

const KILLER_PLIES: usize = 64;
/// 静止搜索最大延伸层数（应对连吃链）。
const QS_MAX_PLY: u32 = 14;

struct KillerTable {
    slots: [[Move; 2]; KILLER_PLIES],
}

impl KillerTable {
    fn new() -> Self {
        Self {
            slots: [[Move::none(); 2]; KILLER_PLIES],
        }
    }

    fn slot(&self, ply: usize) -> &[Move; 2] {
        &self.slots[ply.min(KILLER_PLIES - 1)]
    }

    fn store(&mut self, ply: usize, mv: Move) {
        if !mv.is_ok() {
            return;
        }
        let p = ply.min(KILLER_PLIES - 1);
        let s = &mut self.slots[p];
        if mv == s[0] || mv == s[1] {
            return;
        }
        s[1] = s[0];
        s[0] = mv;
    }
}

struct SearchCtx<'a> {
    nodes: &'a mut u64,
    policy: &'a Arc<Mutex<Option<PolicyOnnx>>>,
    logit_table: Option<&'a [f32]>,
    vocab: &'a HashMap<String, usize>,
    root_depth: u32,
    tt: &'a mut TranspositionTable,
    killers: &'a mut KillerTable,
    stop: Option<&'a AtomicBool>,
    limits: SearchLimits,
    seldepth: &'a mut u32,
}

#[inline]
fn should_stop(ctx: &SearchCtx<'_>) -> bool {
    if ctx.stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
        return true;
    }
    if ctx.limits.deadline.is_some_and(|d| Instant::now() >= d) {
        return true;
    }
    if ctx.limits.max_nodes.is_some_and(|max| *ctx.nodes >= max) {
        return true;
    }
    false
}

#[inline]
fn record_seldepth(ctx: &mut SearchCtx<'_>, ply: usize, qs_extra: u32) {
    let d = ply as u32 + qs_extra;
    if d > *ctx.seldepth {
        *ctx.seldepth = d;
    }
}

#[inline]
fn is_capture(pos: &Position, m: Move) -> bool {
    pos.piece_on(m.to_sq()) != Piece::NO_PIECE
}

#[inline]
fn mvv_lva(pos: &Position, m: Move) -> i32 {
    let cap = pos.piece_on(m.to_sq());
    let att = pos.piece_on(m.from_sq());
    if cap == Piece::NO_PIECE {
        return 0;
    }
    let v = PIECE_VALUE[cap.to_usize()];
    let a = PIECE_VALUE[att.to_usize()];
    v * 256 - a
}

fn move_order_key(pos: &Position, m: Move, tt_mv: Option<Move>, killers: &[Move; 2], policy_logit: Option<f32>) -> i64 {
    if Some(m) == tt_mv {
        return i64::MAX;
    }
    if is_capture(pos, m) {
        return 2_000_000_000 + i64::from(mvv_lva(pos, m));
    }
    if m == killers[0] {
        return 1_900_000_000;
    }
    if m == killers[1] {
        return 1_899_000_000;
    }
    if let Some(l) = policy_logit {
        return 1_000_000_000 + (l * 1_000_000.0) as i64;
    }
    0
}

fn collect_legal_moves(pos: &Position) -> Vec<Move> {
    let mut buf = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut buf);
    buf[..n].iter().map(|e| e.mv).collect()
}

/// 静止搜索阶段：非应将时仅扩展吃子（过滤自合法集）。
fn collect_qs_moves(pos: &Position) -> Vec<Move> {
    let legal = collect_legal_moves(pos);
    let in_check = pos.checkers() != 0;
    if in_check {
        return legal;
    }
    legal.into_iter().filter(|&m| is_capture(pos, m)).collect()
}

fn order_moves(
    moves: &mut [Move],
    pos: &Position,
    ply: usize,
    depth_left: u32,
    tt_mv: Option<Move>,
    ctx: &SearchCtx<'_>,
) {
    let killers = ctx.killers.slot(ply);
    let use_policy = ctx.root_depth == depth_left;
    let logits = use_policy.then_some(ctx.logit_table).flatten();
    moves.sort_by_cached_key(|&m| {
        let plog = logits.and_then(|sl| {
            let u = move_to_uci(m);
            ctx.vocab.get(&u).and_then(|&i| sl.get(i)).copied()
        });
        std::cmp::Reverse(move_order_key(pos, m, tt_mv, killers, plog))
    });
}

fn order_qs_captures(moves: &mut [Move], pos: &Position) {
    moves.sort_by_cached_key(|&m| std::cmp::Reverse(mvv_lva(pos, m)));
}

fn static_eval(pos: &Position, ctx: &mut SearchCtx<'_>) -> i32 {
    let mut g = match ctx.policy.lock() {
        Ok(g) => g,
        Err(_) => return crate::eval::material_stm(pos),
    };
    evaluate_leaf(pos, g.as_mut())
}

fn quiescence(pos: &mut Position, ply: usize, qs_ply: u32, mut alpha: i32, beta: i32, ctx: &mut SearchCtx<'_>) -> i32 {
    if should_stop(ctx) {
        return alpha;
    }
    *ctx.nodes += 1;
    record_seldepth(ctx, ply, qs_ply);

    if let Some(s) = terminal_score(pos) {
        return s;
    }
    if qs_ply >= QS_MAX_PLY {
        return static_eval(pos, ctx);
    }

    let in_check = pos.checkers() != 0;
    let stand_pat = static_eval(pos, ctx);

    if !in_check {
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    let mut moves = collect_qs_moves(pos);
    if moves.is_empty() {
        return if in_check {
            terminal_score(pos).unwrap_or(alpha)
        } else {
            alpha
        };
    }
    order_qs_captures(&mut moves, pos);

    for mv in moves {
        if should_stop(ctx) {
            break;
        }
        pos.do_move(mv);
        let sc = -quiescence(pos, ply + 1, qs_ply + 1, -beta, -alpha, ctx);
        pos.undo_move(mv);
        if sc > alpha {
            alpha = sc;
        }
        if alpha >= beta {
            return beta;
        }
    }
    alpha
}

fn negamax(pos: &mut Position, depth_left: u32, ply: usize, mut alpha: i32, beta: i32, ctx: &mut SearchCtx<'_>) -> i32 {
    if should_stop(ctx) {
        return alpha;
    }

    *ctx.nodes += 1;
    record_seldepth(ctx, ply, 0);

    if let Some(s) = terminal_score(pos) {
        return s;
    }

    let mut moves = collect_legal_moves(pos);
    if moves.is_empty() {
        return terminal_score(pos).unwrap_or(0);
    }

    if depth_left == 0 {
        return quiescence(pos, ply, 0, alpha, beta, ctx);
    }

    let key = pos.key();
    let tp = ctx.tt.probe(key, depth_left);
    if let Some(s) = tp.cutoff_score(alpha, beta) {
        return s;
    }
    let tt_mv = tp.best_move.filter(|&m| moves.contains(&m));

    order_moves(&mut moves, pos, ply, depth_left, tt_mv, ctx);

    let alpha_orig = alpha;
    let mut best = i32::MIN / 4;
    let mut best_mv = Move::none();

    for mv in moves {
        if should_stop(ctx) {
            break;
        }
        let quiet = !is_capture(pos, mv);
        pos.do_move(mv);
        let sc = -negamax(pos, depth_left - 1, ply + 1, -beta, -alpha, ctx);
        pos.undo_move(mv);

        if sc > best {
            best = sc;
            best_mv = mv;
        }
        if sc > alpha {
            alpha = sc;
        }
        if sc >= beta {
            if quiet {
                ctx.killers.store(ply, mv);
            }
            ctx.tt.store(key, depth_left, sc, Bound::Lower, mv);
            return sc;
        }
    }

    if best_mv.is_ok() {
        let bound = if best <= alpha_orig { Bound::Upper } else { Bound::Exact };
        ctx.tt.store(key, depth_left, best, bound, best_mv);
    }

    best
}

/// 单一固定深度（主层数 = `depth`）；静止搜索另计 `seldepth`。
pub fn root_search(
    pos: &mut Position,
    depth: u32,
    shared: &mut RootSearchShared<'_>,
    limits: SearchLimits,
) -> Option<RootSearchResult> {
    root_search_impl(pos, depth, shared, limits)
}

/// 迭代加深：深度从 1 到 `max_depth`，最后一轮完整结果作为输出（便于时限内随时 `stop`）。
pub fn root_search_iterative(
    pos: &mut Position,
    max_depth: u32,
    shared: &mut RootSearchShared<'_>,
    limits: SearchLimits,
) -> Option<RootSearchResult> {
    if max_depth == 0 {
        return None;
    }
    let mut last: Option<RootSearchResult> = None;
    let mut total_nodes = 0u64;
    let mut max_sel = 0u32;
    for d in 1..=max_depth {
        if shared.stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
            break;
        }
        if limits.deadline.is_some_and(|t| Instant::now() >= t) {
            break;
        }
        if limits.max_nodes.is_some_and(|max| total_nodes >= max) {
            break;
        }
        let mut eff_limits = limits;
        if let Some(mx) = limits.max_nodes {
            let rem = mx.saturating_sub(total_nodes);
            if rem == 0 {
                break;
            }
            eff_limits.max_nodes = Some(rem);
        }
        if let Some(mut r) = root_search_impl(pos, d, shared, eff_limits) {
            total_nodes += r.nodes;
            max_sel = max_sel.max(r.seldepth);
            r.nodes = total_nodes;
            r.seldepth = max_sel;
            last = Some(r);
        } else {
            break;
        }
    }
    last
}

fn root_search_impl(
    pos: &mut Position,
    depth: u32,
    shared: &mut RootSearchShared<'_>,
    limits: SearchLimits,
) -> Option<RootSearchResult> {
    if depth == 0 {
        return None;
    }

    let mut nodes = 0u64;
    let mut seldepth = 0u32;
    let logit_vec: Option<Vec<f32>> = shared.policy.lock().ok().and_then(|mut g| {
        let net = g.as_mut()?;
        let o = net.eval_fen(&pos.fen()).ok()?;
        (o.logits.len() == shared.vocab_size && shared.vocab_size > 0).then(|| o.logits.clone())
    });
    let logit_slice = logit_vec.as_deref();

    let mut killers = KillerTable::new();
    let mut ctx = SearchCtx {
        nodes: &mut nodes,
        policy: shared.policy,
        logit_table: logit_slice,
        vocab: shared.vocab,
        root_depth: depth,
        tt: shared.tt,
        killers: &mut killers,
        stop: shared.stop,
        limits,
        seldepth: &mut seldepth,
    };

    let mut moves = collect_legal_moves(pos);
    if moves.is_empty() {
        return None;
    }
    // 根节点只做 TT probe 以取得 best_move 排序；此处**不**调用 `TtProbe::cutoff_score`
    //（子树内见 `negamax`）。根上使用近似全窗口 alpha/beta，beta 剪枝在根层不会发生，
    // 若对当前窗口调 cutoff，Upper/Lower 也几乎不可能成立——这是刻意的「根全窗口」策略，
    // 保证遍历所有根着得到真实最优（而非 aspiration 窄窗）。
    let tp = ctx.tt.probe(pos.key(), depth);
    let tt_mv = tp.best_move.filter(|&m| moves.contains(&m));
    order_moves(&mut moves, pos, 0, depth, tt_mv, &ctx);

    let mut best_mv = moves[0];
    let mut best_sc = i32::MIN / 4;
    let mut alpha = i32::MIN / 4;
    let beta = i32::MAX / 4;
    let root_key = pos.key();

    let mut root_interrupted = false;
    for mv in moves {
        if should_stop(&ctx) {
            root_interrupted = true;
            break;
        }
        pos.do_move(mv);
        let sc = -negamax(pos, depth - 1, 1, -beta, -alpha, &mut ctx);
        pos.undo_move(mv);
        if sc > best_sc {
            best_sc = sc;
            best_mv = mv;
        }
        if sc > alpha {
            alpha = sc;
        }
    }

    // 供同局面、同主层深度之后再次搜索时复用 best_move；迭代加深下一轮 depth 更大，probe
    // 时仍可能只命中较浅条目的着法序。中断则不可标 Exact。
    if !root_interrupted && best_mv.is_ok() {
        ctx.tt.store(root_key, depth, best_sc, Bound::Exact, best_mv);
    }

    Some(RootSearchResult {
        main_depth: depth,
        best_uci: move_to_uci(best_mv),
        score_cp: best_sc,
        nodes,
        seldepth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tt::TranspositionTable;
    use xiangqi_core::START_FEN;

    #[test]
    fn startpos_depth1_returns_legal_move() {
        let mut pos = Position::from_fen(START_FEN).unwrap();
        let policy = Arc::new(Mutex::new(None));
        let vocab: HashMap<String, usize> = HashMap::new();
        let mut tt = TranspositionTable::new(1);
        let mut shared = RootSearchShared {
            policy: &policy,
            vocab: &vocab,
            vocab_size: 0,
            tt: &mut tt,
            stop: None,
        };
        let r = root_search(&mut pos, 1, &mut shared, SearchLimits::none()).expect("mv");
        assert!(r.best_uci.len() >= 4);
        assert!(r.seldepth >= 1);
    }

    #[test]
    fn tt_reuse_reduces_nodes_on_second_search() {
        let mut pos = Position::from_fen(START_FEN).unwrap();
        let policy = Arc::new(Mutex::new(None));
        let vocab: HashMap<String, usize> = HashMap::new();
        let mut tt = TranspositionTable::new(4);
        let mut shared = RootSearchShared {
            policy: &policy,
            vocab: &vocab,
            vocab_size: 0,
            tt: &mut tt,
            stop: None,
        };
        let n1 = root_search(&mut pos, 3, &mut shared, SearchLimits::none())
            .unwrap()
            .nodes;
        let mut pos2 = Position::from_fen(START_FEN).unwrap();
        let n2 = root_search(&mut pos2, 3, &mut shared, SearchLimits::none())
            .unwrap()
            .nodes;
        assert!(n2 < n1, "第二次同局面同深度应命中 TT，节点数应下降: n1={n1} n2={n2}");
    }

    #[test]
    fn qs_reports_seldepth_at_least_main_depth() {
        let mut pos = Position::from_fen(START_FEN).unwrap();
        let policy = Arc::new(Mutex::new(None));
        let vocab: HashMap<String, usize> = HashMap::new();
        let mut tt = TranspositionTable::new(4);
        let mut shared = RootSearchShared {
            policy: &policy,
            vocab: &vocab,
            vocab_size: 0,
            tt: &mut tt,
            stop: None,
        };
        let r = root_search(&mut pos, 2, &mut shared, SearchLimits::none()).unwrap();
        assert!(r.seldepth >= 2, "seldepth={}", r.seldepth);
    }
}
