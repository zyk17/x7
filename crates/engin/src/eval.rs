//! 叶子局面评估：优先 **NN value**（[-1,1]），否则 **物质差**（`xiangqi_core::PIECE_VALUE`，行棋方视角）。

use std::collections::HashMap;

use xiangqi_core::types::{color_of, type_of, Color, Piece, PieceType, PIECE_VALUE, SQUARE_NB, VALUE_DRAW};
use xiangqi_core::Position;

use crate::policy_onnx::PolicyOnnx;

/// ONNX 叶子价值使用策略（见 `docs/onnx-leaf-implementation.md`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NNLeafMode {
    /// 恒为物质差。
    Off,
    /// 仅在从主搜索刚进入静止搜索时的首个评估点用 NN；静止内部与延伸上限处用物质差。
    #[default]
    MainLeafOnly,
    /// 与旧版一致：静止搜索内每次 `static_eval` 均可走 NN。
    AllLeaf,
}

impl NNLeafMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            NNLeafMode::Off => "Off",
            NNLeafMode::MainLeafOnly => "MainLeafOnly",
            NNLeafMode::AllLeaf => "AllLeaf",
        }
    }

    /// UCI `combo` 值（大小写不敏感）。
    pub fn parse_uci(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Some(NNLeafMode::Off),
            "mainleafonly" | "main" | "main-only" => Some(NNLeafMode::MainLeafOnly),
            "allleaf" | "all" => Some(NNLeafMode::AllLeaf),
            _ => None,
        }
    }
}

/// 区分主搜索「叶子根」与静止搜索内部，供 [`NNLeafMode::MainLeafOnly`] 使用。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NnEvalSite {
    /// 主层 `depth_left==0` 后进入静止搜索、`qs_ply==0` 时的评估点。
    MainLeafRoot,
    /// 静止搜索递归节点或 `QS_MAX` 截断处。
    Quiescence,
}

/// 单次根搜索（含迭代加深整段）内的 NN value 缓存与统计。
#[derive(Debug, Default)]
pub struct NnEvalSession {
    pub cache: HashMap<u64, i32>,
    pub cache_hits: u64,
    pub cache_misses: u64,
    /// 单次搜索允许的最大 ONNX **前向**次数（不含仅命中缓存）；`0` 表示不限制。
    pub nn_eval_budget: u64,
    /// 已执行的 ONNX 前向次数（与 `cache_misses` 在无限预算时一致；受预算截断时可能小于 misses 的「理论值」不适用——截断时不增加 miss）。
    pub nn_eval_budget_used: u64,
    /// 是否曾因预算用尽而跳过 ONNX（仍可能通过缓存命中得到 NN 分值）。
    pub nn_eval_budget_exhausted: bool,
    /// 进入 NN 路径（过门控后、含缓存读写）且站点为 [`NnEvalSite::MainLeafRoot`] 的次数。
    pub nn_eval_main_leaf_calls: u64,
    /// 同上，站点为 [`NnEvalSite::Quiescence`]。
    pub nn_eval_qsearch_calls: u64,
}

impl NnEvalSession {
    pub fn clear_search(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.nn_eval_budget_used = 0;
        self.nn_eval_budget_exhausted = false;
        self.nn_eval_main_leaf_calls = 0;
        self.nn_eval_qsearch_calls = 0;
    }

    /// `hits + misses`，即缓存查找次数（成功用 NN 分值的路径）。
    #[inline]
    pub fn nn_eval_calls(&self) -> u64 {
        self.cache_hits.saturating_add(self.cache_misses)
    }
}

/// 行棋方物质优势（不含位置因子），与核心 `PIECE_VALUE` 一致。
pub fn material_stm(pos: &Position) -> i32 {
    let mut w = 0i32;
    let mut b = 0i32;
    for sq in 0..SQUARE_NB {
        let pc = pos.board[sq];
        if pc == Piece::NO_PIECE {
            continue;
        }
        let pt = type_of(pc);
        if matches!(pt, PieceType::NoPieceType | PieceType::KnightTo | PieceType::PawnTo) {
            continue;
        }
        let v = PIECE_VALUE[pc.to_usize()];
        if color_of(pc) == Color::White {
            w += v;
        } else {
            b += v;
        }
    }
    match pos.side_to_move {
        Color::White => w - b,
        Color::Black => b - w,
    }
}

/// `value` ∈ [-1,1] 映射到与物质同量级的 centipawn 尺度（启发式）。
const NN_VALUE_SCALE_CP: f32 = 4000.0;

#[inline]
fn nn_use_for_site(mode: NNLeafMode, site: NnEvalSite) -> bool {
    match mode {
        NNLeafMode::Off => false,
        NNLeafMode::MainLeafOnly => matches!(site, NnEvalSite::MainLeafRoot),
        NNLeafMode::AllLeaf => true,
    }
}

/// 是否允许走 NN 路径（缓存 / 前向），不含「有无 `net`」与预算。
#[inline]
fn nn_allowed(
    mode: NNLeafMode,
    site: NnEvalSite,
    qs_ply: u32,
    in_check: bool,
) -> bool {
    if !nn_use_for_site(mode, site) {
        return false;
    }
    if in_check {
        return false;
    }
    if mode == NNLeafMode::MainLeafOnly && qs_ply != 0 {
        return false;
    }
    true
}

/// 叶子分：按模式 / 站点 / `qs_ply` / 应将门控；带缓存与 ONNX 前向预算。
pub fn leaf_score(
    pos: &Position,
    net: Option<&mut PolicyOnnx>,
    mode: NNLeafMode,
    site: NnEvalSite,
    qs_ply: u32,
    in_check: bool,
    session: &mut NnEvalSession,
) -> i32 {
    if !nn_allowed(mode, site, qs_ply, in_check) {
        return material_stm(pos);
    }
    let Some(n) = net else {
        return material_stm(pos);
    };

    match site {
        NnEvalSite::MainLeafRoot => session.nn_eval_main_leaf_calls += 1,
        NnEvalSite::Quiescence => session.nn_eval_qsearch_calls += 1,
    }

    let key = pos.nn_input_key();
    if let Some(&v) = session.cache.get(&key) {
        session.cache_hits += 1;
        return v;
    }

    if session.nn_eval_budget > 0 && session.nn_eval_budget_used >= session.nn_eval_budget {
        session.nn_eval_budget_exhausted = true;
        return material_stm(pos);
    }

    session.cache_misses += 1;
    session.nn_eval_budget_used += 1;

    let v = if let Ok(out) = n.eval_position(pos) {
        if let Some(val) = out.value {
            (val.clamp(-1.0_f32, 1.0_f32) * NN_VALUE_SCALE_CP) as i32
        } else {
            material_stm(pos)
        }
    } else {
        material_stm(pos)
    };
    session.cache.insert(key, v);
    v
}

/// 兼容旧调用：等价于 `AllLeaf`、非应将、静止内部站点（测试 / 冒烟）。
pub fn evaluate_leaf(pos: &Position, net: Option<&mut PolicyOnnx>) -> i32 {
    let mut sess = NnEvalSession::default();
    leaf_score(
        pos,
        net,
        NNLeafMode::AllLeaf,
        NnEvalSite::Quiescence,
        1,
        false,
        &mut sess,
    )
}

/// 无子可走：将死 / 困毙。
pub fn terminal_score(pos: &Position) -> Option<i32> {
    use xiangqi_core::generate;
    use xiangqi_core::movegen::{ExtMove, GenType};
    use xiangqi_core::types::MAX_MOVES;
    let mut buf = [ExtMove {
        mv: xiangqi_core::types::Move::none(),
        value: 0,
    }; MAX_MOVES];
    let n = generate(pos, GenType::Legal, &mut buf);
    if n != 0 {
        return None;
    }
    if pos.checkers() != 0 {
        Some(-30_000)
    } else {
        Some(VALUE_DRAW)
    }
}
