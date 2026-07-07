//! 中国象棋 **局面表示**（实现曾参考公开引擎常见结构；本文件为独立整理）。
//!
//! 9×10 共 90 格（`SQ_A0`～`SQ_I9`）。提供：棋子放置/移除/移动、FEN 解析与输出、
//! `do_move` / `undo_move`、合法性/将军/牵马检测、Zobrist 键、NNUE 用中间编码占位等。
//!
//! ## 棋盘示意（白方在下，自下而上）
//! ```text
//!   a  b  c  d  e  f  g  h  i
//! 9 r  n  b  a  k  a  b  n  r     ← BLACK rank 9
//! 8 .  .  .  .  .  .  .  .  .     ← rank 8 (river)
//! 7 .  c  .  .  .  .  .  c  .
//! 6 p  .  p  .  p  .  p  .  p
//! 5 .  .  .  .  .  .  .  .  .     ← rank 5 (river)
//! 4 .  .  .  .  .  .  .  .  .
//! 3 P  .  P  .  P  .  P  .  P     ← rank 3
//! 2 .  C  .  .  .  .  .  C  .
//! 1 .  .  .  .  .  .  .  .  .
//! 0 R  N  B  A  K  A  B  N  R     ← WHITE rank 0
//! ```

use std::sync::OnceLock;

use crate::misc::PRNG;
use crate::types::*;

/// 全局 Zobrist 表（种子 `1070372`，与常见皮卡鱼族实现对拍用）。
static GLOBAL_ZOBRIST: OnceLock<Zobrist> = OnceLock::new();

/// 供 [`Position::new`] 共用的 Zobrist 表；避免在 API 中到处传递 `&'static Zobrist`。
pub fn global_zobrist() -> &'static Zobrist {
    GLOBAL_ZOBRIST.get_or_init(|| Zobrist::init(&mut PRNG::new(1070372)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// 方向与步长辅助
// ═══════════════════════════════════════════════════════════════════════════════

/// 棋盘一步的坐标增量（9 列：`NORTH = +9`，`EAST = +1` 等）。
/// 为原始 `i32`，可组合（例如马：`2*SOUTH + WEST`）。
pub const NORTH: i32 = 9;
pub const EAST: i32 = 1;
pub const SOUTH: i32 = -9;
pub const WEST: i32 = -1;
pub const NORTH_EAST: i32 = 10;
pub const SOUTH_EAST: i32 = -8;
pub const SOUTH_WEST: i32 = -10;
pub const NORTH_WEST: i32 = 8;

/// 马走「日」的八个方向（先直两格再拐一格类组合，用步长常量表示）。
pub const KNIGHT_DIRS: [i32; 8] = [
    2 * SOUTH + WEST, // -19
    2 * SOUTH + EAST, // -17
    SOUTH + 2 * WEST, // -11
    SOUTH + 2 * EAST, // -7
    NORTH + 2 * WEST, // 7
    NORTH + 2 * EAST, // 11
    2 * NORTH + WEST, // 17
    2 * NORTH + EAST, // 19
];

/// 象走「田」的四个斜向两格方向。
pub const BISHOP_DIRS: [i32; 4] = [2 * NORTH_EAST, 2 * SOUTH_EAST, 2 * SOUTH_WEST, 2 * NORTH_WEST];

/// 判断 `from` 到 `to` 是否为棋盘上的**一步**（不绕边；切比雪夫距离 ≤ 1）。
pub fn is_valid_step(from: Square, to: Square) -> bool {
    is_ok(from) && is_ok(to) && sq_distance(from, to) <= 1
}

/// 两格之间的切比雪夫距离：`max(|列差|, |行差|)`。
pub fn sq_distance(a: Square, b: Square) -> u32 {
    let df = (file_of(a) as i32 - file_of(b) as i32).abs() as u32;
    let dr = (rank_of(a) as i32 - rank_of(b) as i32).abs() as u32;
    df.max(dr)
}

/// 是否在九宫：白方 0～2 行、黑方 7～9 行且列 3～5。
pub fn is_in_palace(s: Square) -> bool {
    let file = file_of(s) as u8;
    let rank = rank_of(s) as u8;
    file >= 3 && file <= 5 && (rank <= 2 || rank >= 7)
}

/// 兵卒 `c` 是否已过河（过河后可横走）。
pub fn has_crossed_river(s: Square, c: Color) -> bool {
    let rank = rank_of(s) as u8;
    match c {
        Color::White => rank >= 5,
        Color::Black => rank <= 4,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 攻击演算（迭代；未使用 magic bitboard）
// ═══════════════════════════════════════════════════════════════════════════════

/// 内部使用的 128 位位棋盘（便于集合运算）。
pub type Bitboard = u128;

/// 单格置位的位棋盘。
pub fn square_bb(s: Square) -> Bitboard {
    1u128 << (s as u8)
}

/// 位棋盘中置 1 的位数。
pub fn popcount(b: Bitboard) -> u32 {
    b.count_ones()
}

/// 是否多于一个格子被置位。
pub fn more_than_one(b: Bitboard) -> bool {
    (b & (b - 1)) != 0
}

/// 遍历位棋盘中所有置位的格子。
pub struct BitIter {
    bb: Bitboard,
}
impl Iterator for BitIter {
    type Item = Square;
    fn next(&mut self) -> Option<Square> {
        if self.bb == 0 {
            return None;
        }
        let idx = self.bb.trailing_zeros() as u8;
        self.bb &= self.bb - 1;
        Some(unsafe { std::mem::transmute(idx) })
    }
}
pub fn bit_iter(b: Bitboard) -> BitIter {
    BitIter { bb: b }
}

/// `a` 与 `b` 之间同线格子（不含 `a`、含 `b`）；不同线则为空。
pub fn between_bb(a: Square, b: Square) -> Bitboard {
    let mut result = 0u128;
    let ok_rook = rook_attacks_on_empty(a) & square_bb(b) != 0;
    let ok_knight = knight_attacks_on_empty(a) & square_bb(b) != 0;

    if ok_rook {
        let d = if rank_of(a) == rank_of(b) {
            if file_of(b) as i32 > file_of(a) as i32 {
                EAST
            } else {
                WEST
            }
        } else if file_of(a) == file_of(b) {
            if rank_of(b) as i32 > rank_of(a) as i32 {
                NORTH
            } else {
                SOUTH
            }
        } else {
            0
        };
        if d != 0 {
            let mut s = a as i32 + d;
            while s != b as i32 && s >= 0 && s < SQUARE_NB as i32 {
                result |= 1u128 << (s as u8);
                s += d;
            }
        }
    }
    if ok_knight {
        let d = (b as i32 - a as i32) as i8;
        let block_bb = knight_block_square(a, d);
        if block_bb != 0 {
            result |= block_bb;
        }
    }
    result | square_bb(b)
}

/// 马腿所在格（给定方向）；返回单比特位棋盘，无效则为 0。
pub fn knight_block_square(from: Square, dir: i8) -> Bitboard {
    let to_val = from as i32 + dir as i32;
    if to_val < 0 || to_val >= SQUARE_NB as i32 {
        return 0;
    }
    let to = unsafe { std::mem::transmute(to_val as u8) };
    if sq_distance(from, to) > 3 {
        return 0;
    }

    let file_delta = file_of(to) as i32 - file_of(from) as i32;
    let rank_delta = rank_of(to) as i32 - rank_of(from) as i32;
    let df = file_delta.abs();
    let dr = rank_delta.abs();
    let block_sq_val = if df > 1 && dr > 1 {
        // 象向两格斜：阻挡为中间格
        from as i32 + (dir as i32 / 2)
    } else if df == 2 {
        // 马：马腿为横向一步格
        let step = if file_delta > 0 { EAST } else { WEST };
        from as i32 + step
    } else {
        // 马：马腿为纵向一步格
        let step = if rank_delta > 0 { NORTH } else { SOUTH };
        from as i32 + step
    };
    if block_sq_val < 0 || block_sq_val >= SQUARE_NB as i32 {
        return 0;
    }
    1u128 << (block_sq_val as u8)
}

/// 车式直线滑动攻击，遇子即停（含首个阻挡子）。
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut att = 0u128;
    for &dir in &[NORTH, SOUTH, EAST, WEST] {
        let mut s = sq as i32;
        loop {
            s += dir;
            if s < 0 || s >= SQUARE_NB as i32 {
                break;
            }
            let cur: Square = unsafe { std::mem::transmute(s as u8) };
            if !is_valid_step(unsafe { std::mem::transmute((s - dir) as u8) }, cur) {
                break;
            }
            att |= square_bb(cur);
            if occupied & square_bb(cur) != 0 {
                break;
            }
        }
    }
    att
}

/// 空棋盘上的车攻击（用于伪攻击表）。
pub fn rook_attacks_on_empty(sq: Square) -> Bitboard {
    rook_attacks(sq, 0)
}

/// 炮：滑动同车；吃子须隔**恰好一枚**炮架（「山」）。
pub fn cannon_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut att = 0u128;
    for &dir in &[NORTH, SOUTH, EAST, WEST] {
        let mut hurdle = false;
        let mut s = sq as i32;
        loop {
            s += dir;
            if s < 0 || s >= SQUARE_NB as i32 {
                break;
            }
            let cur: Square = unsafe { std::mem::transmute(s as u8) };
            if !is_valid_step(unsafe { std::mem::transmute((s - dir) as u8) }, cur) {
                break;
            }

            if occupied & square_bb(cur) != 0 {
                if !hurdle {
                    hurdle = true; // 第一枚子为炮架，炮不吃炮架
                } else {
                    att |= square_bb(cur); // 第二枚子可为目标（吃子）
                    break;
                }
            } else if !hurdle {
                att |= square_bb(cur); // 空格为走子目标（非吃）
            }
        }
    }
    att
}

/// 马「日」字攻击；**马腿**必须为空。
pub fn knight_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut att = 0u128;
    for &dir in &KNIGHT_DIRS {
        let to_val = sq as i32 + dir;
        if to_val < 0 || to_val >= SQUARE_NB as i32 {
            continue;
        }
        let to: Square = unsafe { std::mem::transmute(to_val as u8) };
        if sq_distance(sq, to) > 2 {
            continue;
        }
        let block = knight_block_square(sq, dir as i8);
        if block & occupied == 0 {
            att |= square_bb(to);
        }
    }
    att
}

/// 空棋盘上的马伪攻击。
pub fn knight_attacks_on_empty(sq: Square) -> Bitboard {
    knight_attacks(sq, 0)
}

/// 象「田」字攻击；**田心**必须为空。
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut att = 0u128;
    // 象不能过河：仅限己方半盘
    let rank = rank_of(sq) as u8;
    let own_half: Bitboard = if rank > 4 {
        // 黑方半盘：第 5～9 行，位 45～89
        !((1u128 << 45) - 1)
    } else {
        // 白方半盘：第 0～4 行，位 0～44
        (1u128 << 45) - 1
    };

    for &dir in &BISHOP_DIRS {
        let to_val = sq as i32 + dir;
        if to_val < 0 || to_val >= SQUARE_NB as i32 {
            continue;
        }
        let to: Square = unsafe { std::mem::transmute(to_val as u8) };
        if sq_distance(sq, to) > 2 {
            continue;
        }
        let block = knight_block_square(sq, dir as i8);
        if block & occupied == 0 {
            att |= square_bb(to);
        }
    }
    att & own_half
}

/// 空棋盘上的象伪攻击。
pub fn bishop_attacks_on_empty(sq: Square) -> Bitboard {
    bishop_attacks(sq, 0)
}

/// 将/帅：九宫内向四邻走一步。
pub fn king_attacks(sq: Square) -> Bitboard {
    let mut att = 0u128;
    for &step in &[NORTH, SOUTH, EAST, WEST] {
        let to_val = sq as i32 + step;
        if to_val < 0 || to_val >= SQUARE_NB as i32 {
            continue;
        }
        let to: Square = unsafe { std::mem::transmute(to_val as u8) };
        if is_valid_step(sq, to) && is_in_palace(to) {
            att |= square_bb(to);
        }
    }
    att
}

/// 士：九宫内向斜走一步。
pub fn advisor_attacks(sq: Square) -> Bitboard {
    let mut att = 0u128;
    for &step in &[NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST] {
        let to_val = sq as i32 + step;
        if to_val < 0 || to_val >= SQUARE_NB as i32 {
            continue;
        }
        let to: Square = unsafe { std::mem::transmute(to_val as u8) };
        if is_valid_step(sq, to) && is_in_palace(to) {
            att |= square_bb(to);
        }
    }
    att
}

/// 兵卒攻击：未过河仅向前；过河后可向前或横向。
pub fn pawn_attacks(sq: Square, c: Color) -> Bitboard {
    let forward = match c {
        Color::White => NORTH,
        Color::Black => SOUTH,
    };
    let mut att = 0u128;
    let fwd_val = sq as i32 + forward;
    if fwd_val >= 0 && fwd_val < SQUARE_NB as i32 {
        let fwd: Square = unsafe { std::mem::transmute(fwd_val as u8) };
        if is_valid_step(sq, fwd) {
            att |= square_bb(fwd);
        }
    }
    if has_crossed_river(sq, c) {
        for &side in &[EAST, WEST] {
            let s_val = sq as i32 + side;
            if s_val >= 0 && s_val < SQUARE_NB as i32 {
                let s_sq: Square = unsafe { std::mem::transmute(s_val as u8) };
                if is_valid_step(sq, s_sq) {
                    att |= square_bb(s_sq);
                }
            }
        }
    }
    att
}

/// 在给定全棋盘占用 `occupied` 下，子力在 `sq` 的攻击位棋盘。
pub fn piece_attacks(pt: PieceType, sq: Square, occupied: Bitboard, c: Color) -> Bitboard {
    match pt {
        PieceType::Rook => rook_attacks(sq, occupied),
        PieceType::Cannon => cannon_attacks(sq, occupied),
        PieceType::Knight => knight_attacks(sq, occupied),
        PieceType::Bishop => bishop_attacks(sq, occupied),
        PieceType::King => king_attacks(sq),
        PieceType::Advisor => advisor_attacks(sq),
        PieceType::Pawn => pawn_attacks(sq, c),
        _ => 0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Zobrist 哈希
// ═══════════════════════════════════════════════════════════════════════════════

pub struct Zobrist {
    /// 棋子-格子随机键
    pub psq: [[Key; SQUARE_NB]; PIECE_NB],
    /// 行棋方键（轮到黑方时异或）
    pub side: Key,
    /// 「无兵」键
    pub no_pawns: Key,
}

impl Zobrist {
    pub fn init(rng: &mut PRNG) -> Self {
        let mut psq = [[0u64; SQUARE_NB]; PIECE_NB];
        // 仅遍历有效棋子值（跳过 NO_PIECE=0 与间隔 8）
        let valid_pieces: [Piece; 14] = [
            Piece::W_ROOK,
            Piece::W_ADVISOR,
            Piece::W_CANNON,
            Piece::W_PAWN,
            Piece::W_KNIGHT,
            Piece::W_BISHOP,
            Piece::W_KING,
            Piece::B_ROOK,
            Piece::B_ADVISOR,
            Piece::B_CANNON,
            Piece::B_PAWN,
            Piece::B_KNIGHT,
            Piece::B_BISHOP,
            Piece::B_KING,
        ];
        for &pc in &valid_pieces {
            let pc_val = pc.0 as usize;
            for sq_val in 0..SQUARE_NB {
                psq[pc_val][sq_val] = rng.rand();
            }
        }
        Zobrist {
            psq,
            side: rng.rand(),
            no_pawns: rng.rand(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// StateInfo — do_move / undo_move 时压栈、弹栈的增量状态
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct StateInfo {
    // ── 走子时拷贝保存 ──
    pub pawn_key: Key,
    pub minor_piece_key: Key,
    pub non_pawn_key: [Key; 2],
    pub major_material: [Value; 2],
    pub check10: [i16; 2],
    pub rule60: i32,
    pub plies_from_null: i32,

    // ── 每次重新计算 ──
    pub key: Key,
    pub checkers_bb: Bitboard,
    pub blockers_for_king: [Bitboard; 2],
    pub pinners: [Bitboard; 2],
    pub check_squares: [Bitboard; PIECE_TYPE_NB],
    pub need_full_check: bool,
    pub captured_piece: Piece,
    pub r#move: Move,
}

impl Default for StateInfo {
    fn default() -> Self {
        StateInfo {
            pawn_key: 0,
            minor_piece_key: 0,
            non_pawn_key: [0; 2],
            major_material: [0; 2],
            check10: [0; 2],
            rule60: 0,
            plies_from_null: 0,
            key: 0,
            checkers_bb: 0,
            blockers_for_king: [0; 2],
            pinners: [0; 2],
            check_squares: [0; PIECE_TYPE_NB],
            need_full_check: false,
            captured_piece: Piece::NO_PIECE,
            r#move: Move::none(),
        }
    }
}

/// [`Position::do_move`] 前的最小快照：键值、计数器与上一着元数据。
/// `checkers_bb` / pinners / `check_squares` 等在 [`Position::undo_move`] 中随局面重算。
#[derive(Debug, Clone, Copy)]
pub struct UndoFrame {
    pub pawn_key: Key,
    pub minor_piece_key: Key,
    pub non_pawn_key: [Key; 2],
    pub major_material: [Value; 2],
    pub check10: [i16; 2],
    pub rule60: i32,
    pub plies_from_null: i32,
    pub key: Key,
    pub captured_piece: Piece,
    pub r#move: Move,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Position — 核心局面
// ═══════════════════════════════════════════════════════════════════════════════

/// FEN 棋子字符表：`" RACPNBK racpnbk"`
pub const PIECE_TO_CHAR: &str = " RACPNBK racpnbk";

pub struct Position {
    /// 每格棋子；空为 `NO_PIECE`。
    pub board: [Piece; SQUARE_NB],

    /// 各 [`Piece`] 枚数（按下标索引）。
    pub piece_count: [i32; PIECE_NB],

    /// NNUE 用中间编码（增量特征计数）。
    pub mid_encoding: [u64; 2],

    /// 当前增量状态。
    pub state: StateInfo,

    /// 行棋方。
    pub side_to_move: Color,

    /// 已执行半着数。
    pub game_ply: i32,

    /// 重复检测用布隆过滤器。
    pub filter: BloomFilter,

    /// 紧凑撤销快照（见 [`UndoFrame`]），由 [`Self::do_move`] 压栈。
    pub undo_stack: Vec<UndoFrame>,

    /// Zobrist 表（通常来自全局一次初始化）。
    pub zobrist: &'static Zobrist,
}

impl Clone for Position {
    fn clone(&self) -> Self {
        Self {
            board: self.board,
            piece_count: self.piece_count,
            mid_encoding: self.mid_encoding,
            state: self.state.clone(),
            side_to_move: self.side_to_move,
            game_ply: self.game_ply,
            filter: self.filter.clone(),
            undo_stack: self.undo_stack.clone(),
            zobrist: self.zobrist,
        }
    }
}

impl Position {
    /// 创建空局面（无子）；须再调用 [`Self::set_fen`] 初始化。
    pub fn new(zobrist: &'static Zobrist) -> Self {
        Position {
            board: [Piece::NO_PIECE; SQUARE_NB],
            piece_count: [0; PIECE_NB],
            mid_encoding: [0; 2],
            state: StateInfo::default(),
            side_to_move: Color::White,
            game_ply: 0,
            filter: BloomFilter::new(),
            undo_stack: Vec::new(),
            zobrist,
        }
    }

    /// 空棋盘并使用全局 Zobrist（库调用方常用）。
    pub fn new_with_global_zobrist() -> Self {
        Self::new(global_zobrist())
    }

    /// 为搜索构造当前局面快照。
    ///
    /// 搜索只需要当前局面真相，不需要外部对局历史，因此不复制 `undo_stack`。
    pub fn clone_for_search(&self) -> Self {
        Self {
            board: self.board,
            piece_count: self.piece_count,
            mid_encoding: self.mid_encoding,
            state: self.state.clone(),
            side_to_move: self.side_to_move,
            game_ply: self.game_ply,
            filter: self.filter.clone(),
            undo_stack: Vec::new(),
            zobrist: self.zobrist,
        }
    }

    /// 自 FEN 解析为新局面（使用全局 Zobrist）。
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut pos = Self::new_with_global_zobrist();
        pos.set_fen(fen)?;
        Ok(pos)
    }

    // ── 子力访问 ─────────────────────────────────────────────────────────

    pub fn piece_on(&self, s: Square) -> Piece {
        self.board[s as usize]
    }

    pub fn empty(&self, s: Square) -> bool {
        self.piece_on(s) == Piece::NO_PIECE
    }

    pub fn count_piece(&self, c: Color, pt: PieceType) -> i32 {
        self.piece_count[make_piece(c, pt).0 as usize]
    }

    pub fn total_pieces(&self) -> i32 {
        self.piece_count[make_piece(Color::White, PieceType::NoPieceType).0 as usize]
            + self.piece_count[make_piece(Color::Black, PieceType::NoPieceType).0 as usize]
    }

    pub fn king_square(&self, c: Color) -> Square {
        let king = make_piece(c, PieceType::King);
        for sq_val in 0..SQUARE_NB {
            let s: Square = unsafe { std::mem::transmute(sq_val as u8) };
            if self.board[sq_val] == king {
                return s;
            }
        }
        panic!("King not found for {:?}", c);
    }

    // ── 全盘占用位棋盘 ───────────────────────────────────────────────────────

    pub fn occupancy(&self) -> Bitboard {
        let mut bb = 0u128;
        for sq_val in 0..SQUARE_NB {
            if self.board[sq_val] != Piece::NO_PIECE {
                bb |= 1u128 << sq_val;
            }
        }
        bb
    }

    pub fn color_bb(&self, c: Color) -> Bitboard {
        let mut bb = 0u128;
        for sq_val in 0..SQUARE_NB {
            if self.board[sq_val] != Piece::NO_PIECE && color_of(self.board[sq_val]) == c {
                bb |= 1u128 << sq_val;
            }
        }
        bb
    }

    pub fn piece_type_bb(&self, pt: PieceType) -> Bitboard {
        let mut bb = 0u128;
        for sq_val in 0..SQUARE_NB {
            if self.board[sq_val] != Piece::NO_PIECE && type_of(self.board[sq_val]) == pt {
                bb |= 1u128 << sq_val;
            }
        }
        bb
    }

    pub fn pieces_c_pt(&self, c: Color, pt: PieceType) -> Bitboard {
        self.color_bb(c) & self.piece_type_bb(pt)
    }

    // ── 位棋盘：将军子 / 攻击者 ──────────────────────────────────────────────

    /// 能攻击格子 `s` 的所有对方子力位棋盘。
    pub fn attackers_to(&self, s: Square) -> Bitboard {
        let occupied = self.occupancy();
        let mut att = 0u128;

        // 兵卒攻击方向（「指向目标格」语义）
        for &c in &[Color::White, Color::Black] {
            let pawns = self.pieces_c_pt(c, PieceType::Pawn);
            let mut bb = pawns;
            while bb != 0 {
                let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
                if pawn_attacks(psq, c) & square_bb(s) != 0 {
                    att |= square_bb(psq);
                }
                bb &= bb - 1;
            }
        }

        // 各子力类型：攻击是否覆盖 s
        for &pt in &[PieceType::Rook, PieceType::Cannon, PieceType::Knight, PieceType::Bishop] {
            let pieces_bb = self.piece_type_bb(pt);
            let mut bb = pieces_bb;
            while bb != 0 {
                let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
                let c = color_of(self.board[psq as usize]);
                if piece_attacks(pt, psq, occupied, c) & square_bb(s) != 0 {
                    att |= square_bb(psq);
                }
                bb &= bb - 1;
            }
        }

        // 士、将：用简单方向检测
        for &pt in &[PieceType::Advisor, PieceType::King] {
            let pieces_bb = self.piece_type_bb(pt);
            let mut bb = pieces_bb;
            while bb != 0 {
                let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
                if piece_attacks(pt, psq, occupied, Color::White) & square_bb(s) != 0 {
                    att |= square_bb(psq);
                }
                bb &= bb - 1;
            }
        }

        att
    }

    /// 颜色为 `c` 且能将军格子 `s` 的子力位棋盘。
    pub fn checkers_to(&self, c: Color, s: Square) -> Bitboard {
        let occupied = self.occupancy();
        self.checkers_to_with_occupied(c, s, occupied, None)
    }

    /// 在给定占用位棋盘下，颜色为 `c` 且能将军格子 `s` 的子力位棋盘。
    pub fn checkers_to_with_occupied(
        &self,
        c: Color,
        s: Square,
        occupied: Bitboard,
        captured_square: Option<Square>,
    ) -> Bitboard {
        let mut att = 0u128;

        // 兵
        let pawns = self.pieces_c_pt(c, PieceType::Pawn);
        let mut bb = pawns;
        while bb != 0 {
            let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
            if Some(psq) == captured_square {
                bb &= bb - 1;
                continue;
            }
            if pawn_attacks(psq, c) & square_bb(s) != 0 {
                att |= square_bb(psq);
            }
            bb &= bb - 1;
        }

        // 马
        let knights = self.pieces_c_pt(c, PieceType::Knight);
        let mut bb = knights;
        while bb != 0 {
            let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
            if Some(psq) == captured_square {
                bb &= bb - 1;
                continue;
            }
            if knight_attacks(psq, occupied) & square_bb(s) != 0 {
                att |= square_bb(psq);
            }
            bb &= bb - 1;
        }

        // 车与将/帅（飞将检测）
        let rooks_kings = self.pieces_c_pt(c, PieceType::Rook) | self.pieces_c_pt(c, PieceType::King);
        let mut bb = rooks_kings;
        while bb != 0 {
            let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
            if Some(psq) == captured_square {
                bb &= bb - 1;
                continue;
            }
            if rook_attacks(psq, occupied) & square_bb(s) != 0 {
                att |= square_bb(psq);
            }
            bb &= bb - 1;
        }

        // 炮
        let cannons = self.pieces_c_pt(c, PieceType::Cannon);
        let mut bb = cannons;
        while bb != 0 {
            let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
            if Some(psq) == captured_square {
                bb &= bb - 1;
                continue;
            }
            if cannon_attacks(psq, occupied) & square_bb(s) != 0 {
                att |= square_bb(psq);
            }
            bb &= bb - 1;
        }

        att
    }

    /// 当前行棋方王（将/帅）所受将军子。
    pub fn checkers(&self) -> Bitboard {
        self.state.checkers_bb
    }

    /// 更新双方的阻挡子与「钉住」来源。
    pub fn set_check_info(&mut self) {
        self.update_blockers(Color::White);
        self.update_blockers(Color::Black);

        let us = self.side_to_move;
        let them = !us;
        let ksq = self.king_square(them);
        let occupied = self.occupancy();

        // 空头炮等需完整合法性检查的情形
        self.state.need_full_check = self.checkers() != 0
            || (rook_attacks(self.king_square(us), 0) & self.pieces_c_pt(them, PieceType::Cannon) != 0);

        // 将军格：从对方王视角，各子力类型可在哪些格「将军」
        self.state.check_squares[PieceType::Pawn as usize] = {
            let mut bb = 0u128;
            for &c in &[Color::White, Color::Black] {
                if c == us {
                    // 己方兵卒能走到哪些格以将军对方王？
                    let _to_bb = pawn_attacks(ksq, them);
                    // 从王格反推：己方兵应在哪些格
                    let forward = match us {
                        Color::White => NORTH,
                        Color::Black => SOUTH,
                    };
                    let from_val = ksq as i32 - forward;
                    if from_val >= 0 && from_val < SQUARE_NB as i32 {
                        bb |= square_bb(unsafe { std::mem::transmute(from_val as u8) });
                    }
                    if has_crossed_river(ksq, us) {
                        for &side in &[EAST, WEST] {
                            let from_val = ksq as i32 - side;
                            if from_val >= 0 && from_val < SQUARE_NB as i32 {
                                bb |= square_bb(unsafe { std::mem::transmute(from_val as u8) });
                            }
                        }
                    }
                }
            }
            bb
        };
        self.state.check_squares[PieceType::Knight as usize] = {
            // 能马踏王格的来源格集合
            let mut bb = 0u128;
            for &dir in &KNIGHT_DIRS {
                let from_val = ksq as i32 + dir;
                if from_val >= 0 && from_val < SQUARE_NB as i32 {
                    let from: Square = unsafe { std::mem::transmute(from_val as u8) };
                    if sq_distance(ksq, from) <= 2 {
                        bb |= square_bb(from);
                    }
                }
            }
            bb
        };
        self.state.check_squares[PieceType::Cannon as usize] = cannon_attacks(ksq, occupied);
        self.state.check_squares[PieceType::Rook as usize] = rook_attacks(ksq, occupied);
        self.state.check_squares[PieceType::King as usize] = 0;
        self.state.check_squares[PieceType::Advisor as usize] = 0;
        self.state.check_squares[PieceType::Bishop as usize] = 0;

        // 空头炮「闪露」路径上的格
        let hollow = self.state.check_squares[PieceType::Rook as usize] & self.pieces_c_pt(us, PieceType::Cannon);
        if hollow != 0 {
            let mut h = hollow;
            let mut discover = 0u128;
            while h != 0 {
                let csq = unsafe { std::mem::transmute(h.trailing_zeros() as u8) };
                let between = between_bb(csq, ksq);
                // 炮与王之间线段上的格可因闪击形成将军
                discover |= between;
                h &= h - 1;
            }
            for pt_val in 1..PIECE_TYPE_NB {
                self.state.check_squares[pt_val] |= discover;
            }
        }
    }

    /// 计算 `blockers_for_king[c]` 与 `pinners[!c]`。
    fn update_blockers(&mut self, c: Color) {
        let ksq = self.king_square(c);
        let them = !c;
        let occupied = self.occupancy();

        self.state.blockers_for_king[c as usize] = 0;
        self.state.pinners[them as usize] = 0;

        // 远狙：沿开放线攻击王格的对方子
        let snipers = {
            let rook_att = rook_attacks(ksq, 0);
            let knight_att = knight_attacks(ksq, 0);
            (rook_att
                & (self.piece_type_bb(PieceType::Rook)
                    | self.piece_type_bb(PieceType::Cannon)
                    | self.piece_type_bb(PieceType::King)))
                | (knight_att & self.piece_type_bb(PieceType::Knight))
        } & self.color_bb(them);

        let mut sniper_bb = snipers;
        while sniper_bb != 0 {
            let sniper_sq = unsafe { std::mem::transmute(sniper_bb.trailing_zeros() as u8) };
            let is_cannon = type_of(self.piece_on(sniper_sq)) == PieceType::Cannon;

            // 王与远狙子之间线段
            let between = between_bb(ksq, sniper_sq);
            let b = if is_cannon {
                between & (occupied ^ square_bb(sniper_sq))
            } else {
                between & (occupied ^ square_bb(sniper_sq))
            };

            if b != 0 {
                if !is_cannon && !more_than_one(b) {
                    self.state.blockers_for_king[c as usize] |= b;
                    if b & self.color_bb(c) != 0 {
                        self.state.pinners[them as usize] |= square_bb(sniper_sq);
                    }
                } else if is_cannon && popcount(b) == 2 {
                    self.state.blockers_for_king[c as usize] |= b;
                    if b & self.color_bb(c) != 0 {
                        self.state.pinners[them as usize] |= square_bb(sniper_sq);
                    }
                }
            }
            sniper_bb &= sniper_bb - 1;
        }
    }

    // ── 合法性 ─────────────────────────────────────────────────────────────

    /// 伪合法着是否真合法。
    pub fn legal(&self, m: Move) -> bool {
        assert!(m.is_ok());
        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.piece_on(from);
        let captured = self.piece_on(to);
        // 与 `do_move` 一致：不允许「吃将」；伪合法生成仍可能产生此类目标格。
        if captured != Piece::NO_PIECE && type_of(captured) == PieceType::King {
            return false;
        }
        let occupied = (self.occupancy() ^ square_bb(from)) | square_bb(to);
        let captured_square = (captured != Piece::NO_PIECE).then_some(to);

        // 将/帅：目标格须不受对方攻击
        if type_of(pc) == PieceType::King {
            return self.checkers_to_with_occupied(them, to, occupied, captured_square) == 0;
        }

        // 快路径：非将着且明显合法
        // 1. Not pinned / 2. Pinned but moving along the pin line
        if !self.state.need_full_check {
            let is_pinned = self.state.blockers_for_king[us as usize] & square_bb(from) != 0;
            if !is_pinned {
                return true;
            }
            let cannon_pinned = type_of(pc) == PieceType::Cannon;
            let is_capture = self.piece_on(to) != Piece::NO_PIECE;
            if (!cannon_pinned || !is_capture) && aligned(from, to, self.king_square(us)) {
                return true;
            }
        }

        // 一般情形：走后王不得仍被将军
        self.checkers_to_with_occupied(them, self.king_square(us), occupied, captured_square) == 0
    }

    /// 该着是否将军。
    pub fn gives_check(&self, m: Move) -> bool {
        assert!(m.is_ok());
        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq();
        let ksq = self.king_square(them);
        let pt = type_of(self.piece_on(from));

        // 直接将军
        if pt == PieceType::Cannon {
            if self.state.check_squares[PieceType::Rook as usize] & square_bb(from) != 0 && aligned(from, to, ksq) {
                if self.piece_on(to) != Piece::NO_PIECE {
                    // 吃子：判断 to 是否在炮与王之间射线上
                    let ray = between_bb(ksq, from);
                    if ray & square_bb(to) != 0 {
                        return true;
                    }
                }
            }
        } else if self.state.check_squares[pt as usize] & square_bb(to) != 0 {
            return true;
        }

        // 闪击：移开阻挡子后露出远狙
        if self.state.blockers_for_king[them as usize] & square_bb(from) != 0
            && (!aligned(from, to, ksq) || self.piece_on(to) != Piece::NO_PIECE)
        {
            return true;
        }

        false
    }

    // ── do_move / undo_move ──────────────────────────────────────────────────

    /// 执行着法（调用方须已保证合法）。
    pub fn do_move(&mut self, m: Move) {
        assert!(m.is_ok(), "Invalid move passed to do_move: raw={:x}", m.raw());

        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.board[from as usize];
        let captured = self.board[to as usize];

        assert!(pc != Piece::NO_PIECE);
        assert!(color_of(pc) == us);
        assert!(
            captured == Piece::NO_PIECE || type_of(captured) != PieceType::King,
            "Cannot capture the king!"
        );

        // ── 修改 self.state 前先快照旧字段 ──
        let old_pawn_key = self.state.pawn_key;
        let old_minor_key = self.state.minor_piece_key;
        let old_non_pawn_key = self.state.non_pawn_key;
        let old_major_mat = self.state.major_material;
        let old_check10 = self.state.check10;
        let old_rule60 = self.state.rule60;
        let old_plies_from_null = self.state.plies_from_null;
        let old_key = self.state.key;

        // 布隆过滤器计数
        self.filter.set(old_key, self.filter.get(old_key).wrapping_add(1));

        self.undo_stack.push(UndoFrame {
            pawn_key: self.state.pawn_key,
            minor_piece_key: self.state.minor_piece_key,
            non_pawn_key: self.state.non_pawn_key,
            major_material: self.state.major_material,
            check10: self.state.check10,
            rule60: self.state.rule60,
            plies_from_null: self.state.plies_from_null,
            key: self.state.key,
            captured_piece: self.state.captured_piece,
            r#move: self.state.r#move,
        });

        // ── 初始化新状态字段 ──
        self.state.r#move = m;
        self.state.captured_piece = Piece::NO_PIECE;
        self.state.pawn_key = old_pawn_key;
        self.state.minor_piece_key = old_minor_key;
        self.state.non_pawn_key = old_non_pawn_key;
        self.state.major_material = old_major_mat;
        self.state.check10 = old_check10;
        self.state.rule60 = old_rule60;
        self.state.plies_from_null = old_plies_from_null;
        self.state.key = old_key ^ self.zobrist.side;
        self.state.checkers_bb = 0;
        self.state.blockers_for_king = [0; 2];
        self.state.pinners = [0; 2];
        self.state.check_squares = [0; PIECE_TYPE_NB];
        self.state.need_full_check = false;

        // ── 计数器 ──
        self.game_ply += 1;
        self.state.plies_from_null += 1;
        let gives_check = self.gives_check(m);
        if !gives_check {
            self.state.rule60 += 1;
        } else {
            self.state.check10[us as usize] += 1;
            if self.state.check10[us as usize] <= 10 {
                self.state.rule60 += 1;
            }
        }

        // ── 在棋盘上执行着法 ──
        if captured != Piece::NO_PIECE {
            self.remove_piece(to);
            self.state.captured_piece = captured;
        }
        self.move_piece(from, to);

        // ── 更新 Zobrist 键 ──
        self.state.key ^= self.zobrist.psq[pc.0 as usize][from as usize] ^ self.zobrist.psq[pc.0 as usize][to as usize];
        if captured != Piece::NO_PIECE {
            self.state.key ^= self.zobrist.psq[captured.0 as usize][to as usize];
        }
        self.state.key ^= self.zobrist.side;

        // 更新兵/轻子/非兵子键
        if type_of(pc) == PieceType::Pawn {
            self.state.pawn_key ^=
                self.zobrist.psq[pc.0 as usize][from as usize] ^ self.zobrist.psq[pc.0 as usize][to as usize];
            if captured != Piece::NO_PIECE && type_of(captured) == PieceType::Pawn {
                self.state.pawn_key ^= self.zobrist.psq[captured.0 as usize][to as usize];
            }
        } else {
            self.state.non_pawn_key[us as usize] ^=
                self.zobrist.psq[pc.0 as usize][from as usize] ^ self.zobrist.psq[pc.0 as usize][to as usize];
            if type_of(pc) as u8 & 1 != 0 && type_of(pc) != PieceType::Rook {
                self.state.minor_piece_key ^=
                    self.zobrist.psq[pc.0 as usize][from as usize] ^ self.zobrist.psq[pc.0 as usize][to as usize];
            }
        }
        if captured != Piece::NO_PIECE {
            if type_of(captured) == PieceType::Pawn {
                self.state.pawn_key ^= self.zobrist.psq[captured.0 as usize][to as usize];
            } else {
                self.state.non_pawn_key[them as usize] ^= self.zobrist.psq[captured.0 as usize][to as usize];
                if type_of(captured) as u8 & 1 != 0 {
                    self.state.major_material[them as usize] -= PIECE_VALUE[captured.0 as usize];
                    if type_of(captured) != PieceType::Rook {
                        self.state.minor_piece_key ^= self.zobrist.psq[captured.0 as usize][to as usize];
                    }
                }
            }
        }

        // 切换行棋方
        self.side_to_move = them;

        // 重算将军信息
        self.state.checkers_bb = self.checkers_to(us, self.king_square(them));
        self.set_check_info();
    }

    /// 撤销上一着，恢复先前状态。
    pub fn undo_move(&mut self, m: Move) {
        assert!(m.is_ok());

        let to = m.to_sq();
        let from = m.from_sq();
        let captured = self.state.captured_piece;

        // 将子移回
        self.move_piece(to, from);

        // 若有吃子则恢复被吃子
        if captured != Piece::NO_PIECE {
            self.put_piece(captured, to);
        }

        let prev = self.undo_stack.pop().expect("undo_move: empty undo_stack");
        self.state.pawn_key = prev.pawn_key;
        self.state.minor_piece_key = prev.minor_piece_key;
        self.state.non_pawn_key = prev.non_pawn_key;
        self.state.major_material = prev.major_material;
        self.state.check10 = prev.check10;
        self.state.rule60 = prev.rule60;
        self.state.plies_from_null = prev.plies_from_null;
        self.state.key = prev.key;
        self.state.captured_piece = prev.captured_piece;
        self.state.r#move = prev.r#move;

        // 恢复行棋方与半着数（不在 StateInfo 中，由 Position 持有）
        self.side_to_move = !self.side_to_move;
        self.game_ply -= 1;

        let us = self.side_to_move;
        let them = !us;
        self.state.checkers_bb = self.checkers_to(them, self.king_square(us));
        self.set_check_info();
    }

    // ── 棋盘子力操作 ───────────────────────────────────────────────────────

    pub fn put_piece(&mut self, pc: Piece, s: Square) {
        self.board[s as usize] = pc;
        self.piece_count[pc.0 as usize] += 1;
        self.piece_count[make_piece(color_of(pc), PieceType::NoPieceType).0 as usize] += 1;
    }

    pub fn remove_piece(&mut self, s: Square) {
        let pc = self.board[s as usize];
        self.board[s as usize] = Piece::NO_PIECE;
        self.piece_count[pc.0 as usize] -= 1;
        self.piece_count[make_piece(color_of(pc), PieceType::NoPieceType).0 as usize] -= 1;
    }

    pub fn move_piece(&mut self, from: Square, to: Square) {
        if from == to {
            return;
        } // 无操作或错误
        let pc = self.board[from as usize];
        self.board[from as usize] = Piece::NO_PIECE;
        self.board[to as usize] = pc;
    }

    // ── FEN 读写 ─────────────────────────────────────────────────────────────

    /// 自 FEN 串设置局面；失败返回 `Err` 信息。
    pub fn set_fen(&mut self, fen: &str) -> Result<(), String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 {
            return Err("Invalid FEN: too few fields".into());
        }

        // 清空棋盘
        self.board = [Piece::NO_PIECE; SQUARE_NB];
        self.piece_count = [0; PIECE_NB];
        self.mid_encoding = [0; 2]; // TODO: BalanceEncoding
        self.undo_stack.clear();

        // 解析棋子面（从第 9 行到第 0 行）
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 10 {
            return Err("Invalid FEN: expected 10 ranks".into());
        }
        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let rank = 9 - rank_idx as i32;
            let mut file = 0;
            for ch in rank_str.chars() {
                if ch.is_ascii_digit() {
                    file += ch.to_digit(10).unwrap() as i32;
                } else if let Some(idx) = PIECE_TO_CHAR.find(ch) {
                    if idx == 0 {
                        return Err(format!("Invalid FEN: unknown piece '{}'", ch));
                    }
                    let pc: Piece = unsafe { std::mem::transmute(idx as u8) };
                    let sq = make_square(unsafe { std::mem::transmute(file as u8) }, unsafe {
                        std::mem::transmute(rank as u8)
                    });
                    self.put_piece(pc, sq);
                    file += 1;
                } else {
                    return Err(format!("Invalid FEN: unexpected char '{}'", ch));
                }
            }
            if file != 9 {
                return Err("Invalid FEN: rank doesn't have 9 files".into());
            }
        }

        // 行棋方
        self.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("Invalid FEN: side to move must be 'w' or 'b'".into()),
        };

        // rule60（半回合计数）
        let rule60: i32 = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        self.state.rule60 = rule60;

        // 全回合计数
        let fullmove: i32 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
        self.game_ply = std::cmp::max(2 * (fullmove - 1), 0) + (self.side_to_move == Color::Black) as i32;

        // 计算初始键与其它状态
        self.init_state();

        Ok(())
    }

    /// 计算初始 Zobrist 与其它状态字段。
    fn init_state(&mut self) {
        let us = self.side_to_move;
        let them = !us;

        self.state.key = 0;
        self.state.minor_piece_key = 0;
        self.state.non_pawn_key = [0; 2];
        self.state.pawn_key = self.zobrist.no_pawns;
        self.state.major_material = [0; 2];

        for sq_val in 0..SQUARE_NB {
            let _s: Square = unsafe { std::mem::transmute(sq_val as u8) };
            let pc = self.board[sq_val];
            if pc == Piece::NO_PIECE {
                continue;
            }
            let pt = type_of(pc);
            let c = color_of(pc);
            self.state.key ^= self.zobrist.psq[pc.0 as usize][sq_val];

            if pt == PieceType::Pawn {
                self.state.pawn_key ^= self.zobrist.psq[pc.0 as usize][sq_val];
            } else {
                self.state.non_pawn_key[c as usize] ^= self.zobrist.psq[pc.0 as usize][sq_val];
                if pt != PieceType::King && (pt as u8 & 1) != 0 {
                    // 大子（车、炮等）
                    self.state.major_material[c as usize] += PIECE_VALUE[pc.0 as usize];
                    if pt != PieceType::Rook {
                        self.state.minor_piece_key ^= self.zobrist.psq[pc.0 as usize][sq_val];
                    }
                }
            }
        }

        if us == Color::Black {
            self.state.key ^= self.zobrist.side;
        }

        // 设置将军子
        self.state.checkers_bb = self.checkers_to(them, self.king_square(us));

        // 设置将军信息（阻挡、钉住、将军格）
        self.set_check_info();
    }

    /// 输出表示当前局面的 FEN 串。
    pub fn fen(&self) -> String {
        let mut result = String::new();
        for rank_idx in (0..10).rev() {
            let rank: Rank = unsafe { std::mem::transmute(rank_idx as u8) };
            let mut empty = 0;
            for file_idx in 0..9 {
                let file: File = unsafe { std::mem::transmute(file_idx as u8) };
                let sq = make_square(file, rank);
                let pc = self.piece_on(sq);
                if pc == Piece::NO_PIECE {
                    empty += 1;
                } else {
                    if empty > 0 {
                        result.push_str(&empty.to_string());
                        empty = 0;
                    }
                    let ch = PIECE_TO_CHAR.as_bytes()[pc.0 as usize] as char;
                    result.push(ch);
                }
            }
            if empty > 0 {
                result.push_str(&empty.to_string());
            }
            if rank_idx > 0 {
                result.push('/');
            }
        }
        let side = if self.side_to_move == Color::White { 'w' } else { 'b' };
        let fullmove = 1 + (self.game_ply - (self.side_to_move == Color::Black) as i32) / 2;
        result.push_str(&format!(" {} - - {} {}", side, self.state.rule60, fullmove));
        result
    }

    // ── 键访问 ───────────────────────────────────────────────────────────────

    /// 置换表等用的完整键：在 [`Self::state`] 的 Zobrist 上混入 **rule60** 与 **重复检测过滤器**（见 [`Self::adjust_key60`]）。
    pub fn key(&self) -> Key {
        self.adjust_key60(self.state.key)
    }

    /// **未**经 `rule60` / 重复状态调整的 Zobrist，仅随 **棋子摆放** 与 **行棋方** 变化。
    ///
    /// 与典型「棋盘平面 + 行棋方」神经网络输入的等价类一致；若 value 网络不消费走子历史，
    /// 用它做缓存键可避免与 [`Self::key`] 相同的局面因反复着法计数不同而重复推理。
    #[inline]
    pub fn nn_input_key(&self) -> Key {
        self.state.key
    }

    fn adjust_key60(&self, k: Key) -> Key {
        let mut key = k;
        if self.state.rule60 >= 14 {
            key ^= make_key(((self.state.rule60 - 14) / 8) as u64);
        }
        if self.filter.get(self.state.key) != 0 {
            key ^= make_key(14);
        }
        key
    }

    pub fn pawn_key(&self) -> Key {
        self.state.pawn_key
    }

    pub fn non_pawn_key(&self, c: Color) -> Key {
        self.state.non_pawn_key[c as usize]
    }

    pub fn minor_piece_key(&self) -> Key {
        self.state.minor_piece_key
    }

    pub fn major_material(&self, c: Color) -> Value {
        self.state.major_material[c as usize]
    }
}

// ── 共线检测 ──────────────────────────────────────────────────────────────

/// 三格 `a`、`b`、`c` 是否共线（同一行或同一列）。
pub fn aligned(a: Square, b: Square, c: Square) -> bool {
    let ra = rank_of(a) as i32;
    let fa = file_of(a) as i32;
    let rb = rank_of(b) as i32;
    let fb = file_of(b) as i32;
    let rc = rank_of(c) as i32;
    let fc = file_of(c) as i32;

    // 同列
    if fa == fb && fb == fc {
        return true;
    }
    // 同行
    if ra == rb && rb == rc {
        return true;
    }
    false
}
