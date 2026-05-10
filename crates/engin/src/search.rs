//! Alpha-Beta：**静止搜索（吃子 / 应将）**、置换表、杀手着法、MVV-LVA、根节点 policy logit 排序。
//! 面向 NN policy/value + 传统剪枝；不必对齐仅有 eval 的参考引擎。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use xiangqi_core::movegen::{ExtMove, GenType};
use xiangqi_core::types::{Bound, Move, Piece, MAX_MOVES, PIECE_VALUE, VALUE_DRAW};
use xiangqi_core::{generate, move_to_uci, write_move_uci_bytes, Position};

use crate::eval::{leaf_score, NNLeafMode, NnEvalSession, NnEvalSite};
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

/// P3 消融：控制 ONNX 在搜索中的消费点（默认 policy + NN 叶子为 `MainLeafOnly`）。
#[derive(Clone, Copy, Debug)]
pub struct SearchAblation {
    /// 根及同层「主层剩余深度 = 全深」时是否用 policy logit 参与排序。
    pub policy_ordering: bool,
    /// 叶子 ONNX value 模式（默认 [`NNLeafMode::MainLeafOnly`]）。
    pub nn_leaf_mode: NNLeafMode,
}

impl Default for SearchAblation {
    fn default() -> Self {
        Self::ALL_ON
    }
}

impl SearchAblation {
    pub const ALL_ON: Self = Self {
        policy_ordering: true,
        nn_leaf_mode: NNLeafMode::MainLeafOnly,
    };
}

/// 单次根搜索 / 迭代加深共享句柄（policy、词表、TT、`stop`），压缩 `root_search*` 参数列表。
pub struct RootSearchShared<'a> {
    pub policy: &'a Arc<Mutex<Option<PolicyOnnx>>>,
    pub vocab: &'a HashMap<String, usize>,
    pub vocab_size: usize,
    pub tt: &'a mut TranspositionTable,
    pub stop: Option<&'a AtomicBool>,
    pub ablation: SearchAblation,
    pub nn_eval: &'a mut NnEvalSession,
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
    pub nn_eval_calls: u64,
    pub nn_eval_cache_hits: u64,
    pub nn_eval_cache_misses: u64,
    pub nn_eval_budget: u64,
    pub nn_eval_budget_used: u64,
    pub nn_eval_budget_exhausted: bool,
    pub nn_eval_main_leaf_calls: u64,
    pub nn_eval_qsearch_calls: u64,
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
    ablation: SearchAblation,
    nn_eval: &'a mut NnEvalSession,
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

/// 无合法着：将死 / 困毙（与 [`crate::eval::terminal_score`] 分值一致）。
#[inline]
fn terminal_mate_or_draw_score(pos: &Position) -> i32 {
    if pos.checkers() != 0 {
        -30_000
    } else {
        VALUE_DRAW
    }
}

#[inline]
fn policy_logit_for_move(
    m: Move,
    logits: &[f32],
    vocab: &HashMap<String, usize>,
    uci_scratch: &mut [u8; 8],
) -> Option<f32> {
    let len = write_move_uci_bytes(m, uci_scratch);
    let u = std::str::from_utf8(&uci_scratch[..len]).ok()?;
    let idx = *vocab.get(u)?;
    logits.get(idx).copied()
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

#[allow(clippy::too_many_arguments)]
fn order_ext_moves_idx(
    n: usize,
    ext_buf: &[ExtMove],
    pos: &Position,
    ply: usize,
    depth_left: u32,
    tt_mv: Option<Move>,
    ctx: &SearchCtx<'_>,
    keys: &mut [i64; MAX_MOVES],
    idx: &mut [usize; MAX_MOVES],
    uci_scratch: &mut [u8; 8],
) {
    let killers = ctx.killers.slot(ply);
    let use_policy = ctx.root_depth == depth_left;
    let logits = use_policy.then_some(ctx.logit_table).flatten();
    for i in 0..n {
        let m = ext_buf[i].mv;
        let plog = logits.and_then(|sl| policy_logit_for_move(m, sl, ctx.vocab, uci_scratch));
        keys[i] = move_order_key(pos, m, tt_mv, killers, plog);
        idx[i] = i;
    }
    idx[..n].sort_unstable_by(|&a, &b| keys[b].cmp(&keys[a]));
}

fn static_eval(
    pos: &Position,
    ctx: &mut SearchCtx<'_>,
    site: NnEvalSite,
    qs_ply: u32,
    in_check: bool,
) -> i32 {
    let mut g = match ctx.policy.lock() {
        Ok(g) => g,
        Err(_) => return crate::eval::material_stm(pos),
    };
    leaf_score(
        pos,
        g.as_mut(),
        ctx.ablation.nn_leaf_mode,
        site,
        qs_ply,
        in_check,
        ctx.nn_eval,
    )
}

fn quiescence(pos: &mut Position, ply: usize, qs_ply: u32, mut alpha: i32, beta: i32, ctx: &mut SearchCtx<'_>) -> i32 {
    if should_stop(ctx) {
        return alpha;
    }
    *ctx.nodes += 1;
    record_seldepth(ctx, ply, qs_ply);

    let in_check = pos.checkers() != 0;
    let mut ext_buf = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];

    if qs_ply >= QS_MAX_PLY {
        let n = generate(pos, GenType::Legal, &mut ext_buf);
        if n == 0 {
            return terminal_mate_or_draw_score(pos);
        }
        return static_eval(pos, ctx, NnEvalSite::Quiescence, qs_ply, in_check);
    }

    let mut cap_moves = [Move::none(); MAX_MOVES];
    // true：走 ext_buf[..qs_n]；false：走 cap_moves[..qs_n]（合法吃子，栈上缓冲）。
    let mut qs_use_ext = false;
    let mut qs_n = 0usize;

    if in_check {
        qs_n = generate(pos, GenType::Legal, &mut ext_buf);
        if qs_n == 0 {
            return terminal_mate_or_draw_score(pos);
        }
        for ext in ext_buf.iter_mut().take(qs_n) {
            ext.value = mvv_lva(pos, ext.mv);
        }
        ext_buf[..qs_n].sort_unstable_by_key(|e| std::cmp::Reverse(e.value));
        qs_use_ext = true;
    } else {
        let cap_n = generate(pos, GenType::Captures, &mut ext_buf);
        for ext in ext_buf.iter().take(cap_n) {
            let m = ext.mv;
            if pos.legal(m) {
                cap_moves[qs_n] = m;
                qs_n += 1;
            }
        }
        if qs_n > 0 {
            cap_moves[..qs_n].sort_unstable_by_key(|b| std::cmp::Reverse(mvv_lva(pos, *b)));
        } else {
            let n_full = generate(pos, GenType::Legal, &mut ext_buf);
            if n_full == 0 {
                return VALUE_DRAW;
            }
            qs_n = 0;
        }
    }

    let site = if qs_ply == 0 {
        NnEvalSite::MainLeafRoot
    } else {
        NnEvalSite::Quiescence
    };
    let stand_pat = static_eval(pos, ctx, site, qs_ply, in_check);

    if !in_check {
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    if qs_n == 0 {
        return alpha;
    }

    for k in 0..qs_n {
        if should_stop(ctx) {
            break;
        }
        let mv = if qs_use_ext {
            ext_buf[k].mv
        } else {
            cap_moves[k]
        };
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

    let mut ext_buf = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut ext_buf);
    if n == 0 {
        return terminal_mate_or_draw_score(pos);
    }
    if depth_left == 0 {
        return quiescence(pos, ply, 0, alpha, beta, ctx);
    }

    let key = pos.key();
    let tp = ctx.tt.probe(key, depth_left);
    if let Some(s) = tp.cutoff_score(alpha, beta) {
        return s;
    }
    let tt_mv = tp
        .best_move
        .filter(|&m| (0..n).any(|j| ext_buf[j].mv == m));

    let mut keys = [0i64; MAX_MOVES];
    let mut idx = [0usize; MAX_MOVES];
    let mut uci_scratch = [0u8; 8];
    order_ext_moves_idx(
        n,
        &ext_buf,
        pos,
        ply,
        depth_left,
        tt_mv,
        ctx,
        &mut keys,
        &mut idx,
        &mut uci_scratch,
    );

    let alpha_orig = alpha;
    let mut best = i32::MIN / 4;
    let mut best_mv = Move::none();

    for k in 0..n {
        let mv = ext_buf[idx[k]].mv;
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
    shared.nn_eval.clear_search();
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
    if let Some(ref mut r) = last {
        r.nn_eval_calls = shared.nn_eval.nn_eval_calls();
        r.nn_eval_cache_hits = shared.nn_eval.cache_hits;
        r.nn_eval_cache_misses = shared.nn_eval.cache_misses;
        r.nn_eval_budget = shared.nn_eval.nn_eval_budget;
        r.nn_eval_budget_used = shared.nn_eval.nn_eval_budget_used;
        r.nn_eval_budget_exhausted = shared.nn_eval.nn_eval_budget_exhausted;
        r.nn_eval_main_leaf_calls = shared.nn_eval.nn_eval_main_leaf_calls;
        r.nn_eval_qsearch_calls = shared.nn_eval.nn_eval_qsearch_calls;
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
    let logit_vec: Option<Vec<f32>> = if shared.ablation.policy_ordering {
        shared.policy.lock().ok().and_then(|mut g| {
            let net = g.as_mut()?;
            let o = net.eval_position(pos).ok()?;
            (o.logits.len() == shared.vocab_size && shared.vocab_size > 0).then(|| o.logits.clone())
        })
    } else {
        None
    };
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
        ablation: shared.ablation,
        nn_eval: shared.nn_eval,
    };

    let mut ext_buf = [ExtMove {
        mv: Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut ext_buf);
    if n == 0 {
        return None;
    }
    // 根节点只做 TT probe 以取得 best_move 排序；此处**不**调用 `TtProbe::cutoff_score`
    //（子树内见 `negamax`）。根上使用近似全窗口 alpha/beta，beta 剪枝在根层不会发生，
    // 若对当前窗口调 cutoff，Upper/Lower 也几乎不可能成立——这是刻意的「根全窗口」策略，
    // 保证遍历所有根着得到真实最优（而非 aspiration 窄窗）。
    let tp = ctx.tt.probe(pos.key(), depth);
    let tt_mv = tp
        .best_move
        .filter(|&m| (0..n).any(|j| ext_buf[j].mv == m));
    let mut keys = [0i64; MAX_MOVES];
    let mut idx = [0usize; MAX_MOVES];
    let mut uci_scratch = [0u8; 8];
    order_ext_moves_idx(
        n,
        &ext_buf,
        pos,
        0,
        depth,
        tt_mv,
        &ctx,
        &mut keys,
        &mut idx,
        &mut uci_scratch,
    );

    let mut best_mv = ext_buf[idx[0]].mv;
    let mut best_sc = i32::MIN / 4;
    let mut alpha = i32::MIN / 4;
    let beta = i32::MAX / 4;
    let root_key = pos.key();

    let mut root_interrupted = false;
    for k in 0..n {
        let mv = ext_buf[idx[k]].mv;
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
        nn_eval_calls: 0,
        nn_eval_cache_hits: 0,
        nn_eval_cache_misses: 0,
        nn_eval_budget: 0,
        nn_eval_budget_used: 0,
        nn_eval_budget_exhausted: false,
        nn_eval_main_leaf_calls: 0,
        nn_eval_qsearch_calls: 0,
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
        let mut nn_eval = NnEvalSession::default();
        let mut shared = RootSearchShared {
            policy: &policy,
            vocab: &vocab,
            vocab_size: 0,
            tt: &mut tt,
            stop: None,
            ablation: SearchAblation::ALL_ON,
            nn_eval: &mut nn_eval,
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
        let mut nn_eval = NnEvalSession::default();
        let mut shared = RootSearchShared {
            policy: &policy,
            vocab: &vocab,
            vocab_size: 0,
            tt: &mut tt,
            stop: None,
            ablation: SearchAblation::ALL_ON,
            nn_eval: &mut nn_eval,
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
        let mut nn_eval = NnEvalSession::default();
        let mut shared = RootSearchShared {
            policy: &policy,
            vocab: &vocab,
            vocab_size: 0,
            tt: &mut tt,
            stop: None,
            ablation: SearchAblation::ALL_ON,
            nn_eval: &mut nn_eval,
        };
        let r = root_search(&mut pos, 2, &mut shared, SearchLimits::none()).unwrap();
        assert!(r.seldepth >= 2, "seldepth={}", r.seldepth);
    }
}
