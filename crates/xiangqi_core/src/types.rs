//! 象棋引擎核心类型（源自 Pikafish / pikafish-rust）。
//!
//! 定义格子、棋子、着法、位棋盘、分值等。棋盘为 **9 列 × 10 行**（90 格），
//! 下标 0（A0）～89（I9）；纵线 A～I，横线 0～9。含走子生成用的特殊子力类型（`KNIGHT_TO`、`PAWN_TO`）。

// ── 常量 ────────────────────────────────────────────────────────────────────

pub const MAX_MOVES: usize = 128;
pub const MAX_PLY: i32 = 246;

// ── Zobrist 键 ───────────────────────────────────────────────────────────────

pub type Key = u64;

/// Zobrist 键混合用 PRNG 种子函数。
pub const fn make_key(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

// ── 位棋盘 ───────────────────────────────────────────────────────────────────

/// 128 位位棋盘，可容纳 90 格。
pub type Bitboard = u128;

// ── 颜色（行棋方）────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

pub const COLOR_NB: usize = 2;

impl Color {
    pub const fn to_usize(self) -> usize {
        self as usize
    }

    pub const fn to_int(self) -> i32 {
        self as i32
    }
}

impl std::ops::Not for Color {
    type Output = Color;
    fn not(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

// ── 边界标志（置换表）────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3, // Upper | Lower
}

// ── 分值 ─────────────────────────────────────────────────────────────────────

pub type Value = i32;

pub const VALUE_ZERO: Value = 0;
pub const VALUE_DRAW: Value = 0;
pub const VALUE_NONE: Value = 32002;
pub const VALUE_INFINITE: Value = 32001;
pub const VALUE_MATE: Value = 32000;
pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY;
pub const VALUE_MATED_IN_MAX_PLY: Value = -VALUE_MATE_IN_MAX_PLY;

pub const fn is_valid(v: Value) -> bool {
    v != VALUE_NONE
}

pub const fn is_win(v: Value) -> bool {
    // assert!(is_valid(v));  -- assertion not const-stable
    v >= VALUE_MATE_IN_MAX_PLY
}

pub const fn is_loss(v: Value) -> bool {
    v <= VALUE_MATED_IN_MAX_PLY
}

pub const fn is_decisive(v: Value) -> bool {
    is_win(v) || is_loss(v)
}

pub const fn mate_in(ply: i32) -> Value {
    VALUE_MATE - ply
}

pub const fn mated_in(ply: i32) -> Value {
    -VALUE_MATE + ply
}

// Piece material values
pub const ROOK_VALUE: Value = 1305;
pub const ADVISOR_VALUE: Value = 219;
pub const CANNON_VALUE: Value = 773;
pub const PAWN_VALUE: Value = 144;
pub const KNIGHT_VALUE: Value = 720;
pub const BISHOP_VALUE: Value = 187;

// ── 搜索深度 ─────────────────────────────────────────────────────────────────

pub type Depth = i32;

pub const DEPTH_QS: Depth = 0;
pub const DEPTH_UNSEARCHED: Depth = -2;
pub const DEPTH_ENTRY_OFFSET: Depth = -3;

// ── 格子 ─────────────────────────────────────────────────────────────────────

/// Squares on the 9×10 xiangqi board, indexed row-major from A0 to I9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum Square {
    SQ_A0, SQ_B0, SQ_C0, SQ_D0, SQ_E0, SQ_F0, SQ_G0, SQ_H0, SQ_I0,
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1, SQ_I1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2, SQ_I2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3, SQ_I3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4, SQ_I4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5, SQ_I5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6, SQ_I6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7, SQ_I7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8, SQ_I8,
    SQ_A9, SQ_B9, SQ_C9, SQ_D9, SQ_E9, SQ_F9, SQ_G9, SQ_H9, SQ_I9,
}

pub const SQUARE_ZERO: usize = 0;
pub const SQUARE_NB: usize = 90;
// SQ_NONE is used as a sentinel value. We represent it as Option<Square>::None
// in Rust. For compatibility with C++ code that needs a numeric sentinel, use:
pub const SQ_NONE_IDX: u8 = 90;

impl Square {
    pub const fn to_int(self) -> i32 {
        self as i32
    }

    pub const fn to_usize(self) -> usize {
        self as usize
    }

    pub const fn to_u8(self) -> u8 {
        self as u8
    }
}

// ── 纵线（列）────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum File {
    FileA, FileB, FileC, FileD, FileE, FileF, FileG, FileH, FileI,
}

pub const FILE_NB: usize = 9;

impl File {
    pub const fn to_int(self) -> i32 {
        self as i32
    }

    pub const fn to_usize(self) -> usize {
        self as usize
    }
}

// ── 横线（行）────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Rank {
    Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9,
}

pub const RANK_NB: usize = 10;

impl Rank {
    pub const fn to_int(self) -> i32 {
        self as i32
    }

    pub const fn to_usize(self) -> usize {
        self as usize
    }
}

// ── 方向 ───────────────────────────────────────────────────────────────────────

/// Board offsets for xiangqi. The board is stored in row-major order with
/// 9 files per rank, so NORTH = +9, EAST = +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Direction {
    North = 9,
    East = 1,
    South = -9,
    West = -1,
    NorthEast = 10,
    SouthEast = -8,
    SouthWest = -10,
    NorthWest = 8,
}

// ── Square / File / Rank 辅助函数 ───────────────────────────────────────────

pub const fn is_ok(s: Square) -> bool {
    (s as u8) < SQUARE_NB as u8
}

pub const fn file_of(s: Square) -> File {
    unsafe { std::mem::transmute((s as u8) % FILE_NB as u8) }
}

pub const fn rank_of(s: Square) -> Rank {
    unsafe { std::mem::transmute((s as u8) / FILE_NB as u8) }
}

pub const fn make_square(f: File, r: Rank) -> Square {
    unsafe { std::mem::transmute((r as u8) * FILE_NB as u8 + (f as u8)) }
}

/// Mirror rank: A0 ↔ A9
pub const fn flip_rank(s: Square) -> Square {
    make_square(file_of(s), unsafe { std::mem::transmute(RANK_NB as u8 - 1 - rank_of(s) as u8) })
}

/// Mirror file: A0 ↔ I0
pub const fn flip_file(s: Square) -> Square {
    make_square(
        unsafe { std::mem::transmute(FILE_NB as u8 - 1 - file_of(s) as u8) },
        rank_of(s),
    )
}

// ── 子力类型 ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum PieceType {
    NoPieceType = 0,
    Rook,
    Advisor,
    Cannon,
    Pawn,
    Knight,
    Bishop,
    King,
    KnightTo,  // special: "by knight" direction for path checking
    PawnTo,    // special: "pawn attack to" direction
}

pub const PIECE_TYPE_NB: usize = 8;
pub const ALL_PIECES: PieceType = unsafe { std::mem::transmute(0u8) }; // 0, used as ALL_PIECES index sentinel

// ── 棋子（含颜色）────────────────────────────────────────────────────────────

/// Piece is a newtype around u8, matching the C++ encoding:
/// bits 0-2: piece type (0-7), bit 3: color (0=WHITE, 1=BLACK).
/// Values 0 = NO_PIECE, 8 = unused gap.
/// This approach lets us use pieces as array indices freely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Piece(pub u8);

impl Piece {
    pub const NO_PIECE: Piece = Piece(0);
    pub const W_ROOK: Piece = Piece(1);
    pub const W_ADVISOR: Piece = Piece(2);
    pub const W_CANNON: Piece = Piece(3);
    pub const W_PAWN: Piece = Piece(4);
    pub const W_KNIGHT: Piece = Piece(5);
    pub const W_BISHOP: Piece = Piece(6);
    pub const W_KING: Piece = Piece(7);
    // gap at 8
    pub const B_ROOK: Piece = Piece(9);
    pub const B_ADVISOR: Piece = Piece(10);
    pub const B_CANNON: Piece = Piece(11);
    pub const B_PAWN: Piece = Piece(12);
    pub const B_KNIGHT: Piece = Piece(13);
    pub const B_BISHOP: Piece = Piece(14);
    pub const B_KING: Piece = Piece(15);

    pub fn to_usize(self) -> usize { self.0 as usize }
    pub fn to_u8(self) -> u8 { self.0 }
}

pub const PIECE_NB: usize = 16;

pub const PIECE_VALUE: [Value; PIECE_NB] = [
    VALUE_ZERO, ROOK_VALUE, ADVISOR_VALUE, CANNON_VALUE,
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, VALUE_ZERO,
    VALUE_ZERO, ROOK_VALUE, ADVISOR_VALUE, CANNON_VALUE,
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, VALUE_ZERO,
];

pub const fn make_piece(c: Color, pt: PieceType) -> Piece {
    Piece(((c as u8) << 3) | pt as u8)
}

pub const fn type_of(pc: Piece) -> PieceType {
    // SAFETY: lower 3 bits always map to a valid PieceType variant (0-7)
    unsafe { std::mem::transmute(pc.0 & 7) }
}

pub const fn color_of(pc: Piece) -> Color {
    // SAFETY: bit 3 is either 0 (WHITE) or 1 (BLACK)
    unsafe { std::mem::transmute(pc.0 >> 3) }
}

impl std::ops::Not for Piece {
    type Output = Piece;
    /// Swap piece color: B_ROOK ↔ W_ROOK, etc.
    fn not(self) -> Piece {
        Piece(self.0 ^ 8)
    }
}

// ── 着法编码 ───────────────────────────────────────────────────────────────────

/// A 16-bit move encoding:
/// - bits 0-6:  destination square (0..89)
/// - bits 7-13: origin square (0..89)
///
/// Special values: Move::none() = 0, Move::null() = 129.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(u16);

impl Move {
    pub const fn none() -> Self {
        Move(0)
    }

    pub const fn null() -> Self {
        Move(129)
    }

    pub const fn make(from: Square, to: Square) -> Self {
        Move(((from as u16) << 7) | (to as u16))
    }

    pub const fn from_sq(self) -> Square {
        // assert!(self.is_ok()); -- const limitation
        unsafe { std::mem::transmute(((self.0 >> 7) & 0x7F) as u8) }
    }

    pub const fn to_sq(self) -> Square {
        unsafe { std::mem::transmute((self.0 & 0x7F) as u8) }
    }

    pub const fn to_sq_unchecked(self) -> Square {
        unsafe { std::mem::transmute((self.0 & 0x7F) as u8) }
    }

    pub const fn is_ok(self) -> bool {
        self.0 != Move::none().0 && self.0 != Move::null().0
    }

    pub const fn raw(self) -> u16 {
        self.0
    }
}

impl From<(Square, Square)> for Move {
    fn from((from, to): (Square, Square)) -> Self {
        Move::make(from, to)
    }
}

// ── BloomFilter（重复检测近似）──────────────────────────────────────────────

pub const FILTER_SIZE: u64 = 1 << 14;

#[derive(Clone)]
pub struct BloomFilter {
    pub table: [u8; FILTER_SIZE as usize],
}

impl BloomFilter {
    pub fn new() -> Self {
        BloomFilter {
            table: [0; FILTER_SIZE as usize],
        }
    }

    pub fn get(&self, key: Key) -> u8 {
        self.table[(key & (FILTER_SIZE - 1)) as usize]
    }

    pub fn set(&mut self, key: Key, value: u8) {
        self.table[(key & (FILTER_SIZE - 1)) as usize] = value;
    }

    pub fn index_mut(&mut self, key: Key) -> &mut u8 {
        &mut self.table[(key & (FILTER_SIZE - 1)) as usize]
    }
}

impl std::ops::Index<Key> for BloomFilter {
    type Output = u8;
    fn index(&self, key: Key) -> &u8 {
        &self.table[(key & (FILTER_SIZE - 1)) as usize]
    }
}

impl std::ops::IndexMut<Key> for BloomFilter {
    fn index_mut(&mut self, key: Key) -> &mut u8 {
        &mut self.table[(key & (FILTER_SIZE - 1)) as usize]
    }
}
