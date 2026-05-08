//! 中国象棋 **局面表示**（源自 Pikafish / pikafish-rust）。
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

/// Global Zobrist keys (seed `1070372`, same as Pikafish / pikafish-rust `main`).
static GLOBAL_ZOBRIST: OnceLock<Zobrist> = OnceLock::new();

/// Shared Zobrist table for [`Position::new`]; avoids threading `&'static Zobrist` through APIs.
pub fn global_zobrist() -> &'static Zobrist {
    GLOBAL_ZOBRIST.get_or_init(|| Zobrist::init(&mut PRNG::new(1070372)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Direction helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Board step offsets. On a 9-file board, North = +9, East = +1, etc.
/// These are raw i32 values that can be combined (e.g., 2*SOUTH + WEST for knight moves).
pub const NORTH: i32 = 9;
pub const EAST: i32 = 1;
pub const SOUTH: i32 = -9;
pub const WEST: i32 = -1;
pub const NORTH_EAST: i32 = 10;
pub const SOUTH_EAST: i32 = -8;
pub const SOUTH_WEST: i32 = -10;
pub const NORTH_WEST: i32 = 8;

/// Knight directions (日 shape, 2 squares one way + 1 square perpendicular).
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

/// Bishop directions (田 shape, 2 squares diagonally).
pub const BISHOP_DIRS: [i32; 4] = [2 * NORTH_EAST, 2 * SOUTH_EAST, 2 * SOUTH_WEST, 2 * NORTH_WEST];

/// Check if the step from `from` to `to` is a valid board step
/// (no wrapping around edges). Uses Chebyshev distance.
pub fn is_valid_step(from: Square, to: Square) -> bool {
    is_ok(from) && is_ok(to) && sq_distance(from, to) <= 1
}

/// Distance between two squares (Chebyshev: max(file_diff, rank_diff)).
pub fn sq_distance(a: Square, b: Square) -> u32 {
    let df = (file_of(a) as i32 - file_of(b) as i32).abs() as u32;
    let dr = (rank_of(a) as i32 - rank_of(b) as i32).abs() as u32;
    df.max(dr)
}

/// Check if a rank is in the palace (ranks 0-2 for WHITE, ranks 7-9 for BLACK).
pub fn is_in_palace(s: Square) -> bool {
    let file = file_of(s) as u8;
    let rank = rank_of(s) as u8;
    file >= 3 && file <= 5 && (rank <= 2 || rank >= 7)
}

/// Check if a pawn of `c` has crossed the river (can move sideways).
pub fn has_crossed_river(s: Square, c: Color) -> bool {
    let rank = rank_of(s) as u8;
    match c {
        Color::White => rank >= 5,
        Color::Black => rank <= 4,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Attack computation (iterative, no magic bitboards)
// ═══════════════════════════════════════════════════════════════════════════════

/// Bitboard (u128) used internally for cheap set operations.
pub type Bitboard = u128;

/// Convert a square to a bitboard with a single bit set.
pub fn square_bb(s: Square) -> Bitboard {
    1u128 << (s as u8)
}

/// Popcount of a bitboard.
pub fn popcount(b: Bitboard) -> u32 {
    b.count_ones()
}

/// Is more than one bit set?
pub fn more_than_one(b: Bitboard) -> bool {
    (b & (b - 1)) != 0
}

/// Iterate set bits in a bitboard.
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

/// Squares between a and b (excluding a, including b); empty set if not on same line.
pub fn between_bb(a: Square, b: Square) -> Bitboard {
    let mut result = 0u128;
    let ok_rook = rook_attacks_on_empty(a) & square_bb(b) != 0;
    let ok_knight = knight_attacks_on_empty(a) & square_bb(b) != 0;

    if ok_rook {
        let d = if rank_of(a) == rank_of(b) {
            if file_of(b) as i32 > file_of(a) as i32 { EAST } else { WEST }
        } else if file_of(a) == file_of(b) {
            if rank_of(b) as i32 > rank_of(a) as i32 { NORTH } else { SOUTH }
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

/// Compute the blocking square for a knight's leg in the given direction.
/// Returns a bitboard with the single blocking square, or 0 if invalid.
pub fn knight_block_square(from: Square, dir: i8) -> Bitboard {
    let to_val = from as i32 + dir as i32;
    if to_val < 0 || to_val >= SQUARE_NB as i32 {
        return 0;
    }
    let to = unsafe { std::mem::transmute(to_val as u8) };
    if sq_distance(from, to) > 3 {
        return 0;
    }

    let df = (file_of(to) as i32 - file_of(from) as i32).abs();
    let dr = (rank_of(to) as i32 - rank_of(from) as i32).abs();
    let block_sq_val = if df > 1 && dr > 1 {
        // Bishop-like (2-diagonal): block is the center square
        from as i32 + (dir as i32 / 2)
    } else if df == 2 {
        // Knight: block is the square one step horizontally
        let step = if to_val > from as i32 { EAST } else { WEST };
        from as i32 + step
    } else {
        // Knight: block is the square one step vertically
        let step = if to_val > from as i32 { NORTH } else { SOUTH };
        from as i32 + step
    };
    if block_sq_val < 0 || block_sq_val >= SQUARE_NB as i32 {
        return 0;
    }
    1u128 << (block_sq_val as u8)
}

/// Straight-line sliding attacks (Rook-style), stops at the first occupied square (inclusive).
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut att = 0u128;
    for &dir in &[NORTH, SOUTH, EAST, WEST] {
        let mut s = sq as i32;
        loop {
            s += dir;
            if s < 0 || s >= SQUARE_NB as i32 { break; }
            let cur: Square = unsafe { std::mem::transmute(s as u8) };
            if !is_valid_step(unsafe { std::mem::transmute((s - dir) as u8) }, cur) { break; }
            att |= square_bb(cur);
            if occupied & square_bb(cur) != 0 { break; }
        }
    }
    att
}

/// Rook attacks on an empty board (for pseudo-attack tables).
pub fn rook_attacks_on_empty(sq: Square) -> Bitboard {
    rook_attacks(sq, 0)
}

/// Cannon attacks: slides like a rook, but to capture must hop over exactly one piece (the "hurdle").
pub fn cannon_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut att = 0u128;
    for &dir in &[NORTH, SOUTH, EAST, WEST] {
        let mut hurdle = false;
        let mut s = sq as i32;
        loop {
            s += dir;
            if s < 0 || s >= SQUARE_NB as i32 { break; }
            let cur: Square = unsafe { std::mem::transmute(s as u8) };
            if !is_valid_step(unsafe { std::mem::transmute((s - dir) as u8) }, cur) { break; }

            if occupied & square_bb(cur) != 0 {
                if !hurdle {
                    hurdle = true; // First piece is the hurdle, cannons don't capture it
                } else {
                    att |= square_bb(cur); // Second piece can be captured
                    break;
                }
            } else if !hurdle {
                att |= square_bb(cur); // Empty square can be moved to (non-capture)
            }
        }
    }
    att
}

/// Knight attacks (日 shape). The "leg" square must be empty.
pub fn knight_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut att = 0u128;
    for &dir in &KNIGHT_DIRS {
        let to_val = sq as i32 + dir;
        if to_val < 0 || to_val >= SQUARE_NB as i32 { continue; }
        let to: Square = unsafe { std::mem::transmute(to_val as u8) };
        if sq_distance(sq, to) > 2 { continue; }
        let block = knight_block_square(sq, dir as i8);
        if block & occupied == 0 {
            att |= square_bb(to);
        }
    }
    att
}

/// Knight pseudo-attacks (empty board).
pub fn knight_attacks_on_empty(sq: Square) -> Bitboard {
    knight_attacks(sq, 0)
}

/// Bishop attacks (田 shape, 2 squares diagonally). The center ("eye") must be empty.
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let mut att = 0u128;
    // Bishop restricted to own half of the board
    let rank = rank_of(sq) as u8;
    let own_half: Bitboard = if rank > 4 {
        // Black half: ranks 5-9, bits 45-89
        !((1u128 << 45) - 1)
    } else {
        // White half: ranks 0-4, bits 0-44
        (1u128 << 45) - 1
    };

    for &dir in &BISHOP_DIRS {
        let to_val = sq as i32 + dir;
        if to_val < 0 || to_val >= SQUARE_NB as i32 { continue; }
        let to: Square = unsafe { std::mem::transmute(to_val as u8) };
        if sq_distance(sq, to) > 2 { continue; }
        let block = knight_block_square(sq, dir as i8);
        if block & occupied == 0 {
            att |= square_bb(to);
        }
    }
    att & own_half
}

/// Bishop pseudo-attacks (empty board).
pub fn bishop_attacks_on_empty(sq: Square) -> Bitboard {
    bishop_attacks(sq, 0)
}

/// King attacks (one step orthogonal, restricted to palace).
pub fn king_attacks(sq: Square) -> Bitboard {
    let mut att = 0u128;
    for &step in &[NORTH, SOUTH, EAST, WEST] {
        let to_val = sq as i32 + step;
        if to_val < 0 || to_val >= SQUARE_NB as i32 { continue; }
        let to: Square = unsafe { std::mem::transmute(to_val as u8) };
        if is_valid_step(sq, to) && is_in_palace(to) {
            att |= square_bb(to);
        }
    }
    att
}

/// Advisor attacks (one step diagonal, restricted to palace).
pub fn advisor_attacks(sq: Square) -> Bitboard {
    let mut att = 0u128;
    for &step in &[NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST] {
        let to_val = sq as i32 + step;
        if to_val < 0 || to_val >= SQUARE_NB as i32 { continue; }
        let to: Square = unsafe { std::mem::transmute(to_val as u8) };
        if is_valid_step(sq, to) && is_in_palace(to) {
            att |= square_bb(to);
        }
    }
    att
}

/// Pawn attacks (one step forward; after crossing river, can also move sideways).
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

/// Get generic attacks for a piece at `sq` given `occupied`.
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
// Zobrist Hashing
// ═══════════════════════════════════════════════════════════════════════════════

pub struct Zobrist {
    /// Piece-square random keys
    pub psq: [[Key; SQUARE_NB]; PIECE_NB],
    /// Side-to-move key (XORed when it's Black's turn)
    pub side: Key,
    /// No pawns key
    pub no_pawns: Key,
}

impl Zobrist {
    pub fn init(rng: &mut PRNG) -> Self {
        let mut psq = [[0u64; SQUARE_NB]; PIECE_NB];
        // Only iterate over valid piece values (skip NO_PIECE=0 and the gap at 8)
        let valid_pieces: [Piece; 14] = [
            Piece::W_ROOK, Piece::W_ADVISOR, Piece::W_CANNON,
            Piece::W_PAWN, Piece::W_KNIGHT, Piece::W_BISHOP, Piece::W_KING,
            Piece::B_ROOK, Piece::B_ADVISOR, Piece::B_CANNON,
            Piece::B_PAWN, Piece::B_KNIGHT, Piece::B_BISHOP, Piece::B_KING,
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
// StateInfo — incremental state that gets pushed/popped on do_move/undo_move
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct StateInfo {
    // ── Copied when making a move ──
    pub pawn_key: Key,
    pub minor_piece_key: Key,
    pub non_pawn_key: [Key; 2],
    pub major_material: [Value; 2],
    pub check10: [i16; 2],
    pub rule60: i32,
    pub plies_from_null: i32,

    // ── Recomputed each time ──
    pub key: Key,
    pub checkers_bb: Bitboard,
    pub previous: Option<Box<StateInfo>>,
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
            previous: None,
            blockers_for_king: [0; 2],
            pinners: [0; 2],
            check_squares: [0; PIECE_TYPE_NB],
            need_full_check: false,
            captured_piece: Piece::NO_PIECE,
            r#move: Move::none(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Position — the core board representation
// ═══════════════════════════════════════════════════════════════════════════════

/// The FEN piece characters: " RACPNBK racpnbk"
pub const PIECE_TO_CHAR: &str = " RACPNBK racpnbk";

pub struct Position {
    /// Piece on each square; NO_PIECE if empty.
    pub board: [Piece; SQUARE_NB],

    /// Piece counts (indexed by Piece value).
    pub piece_count: [i32; PIECE_NB],

    /// NNUE mid-encoding (incremental feature count).
    pub mid_encoding: [u64; 2],

    /// Current state pointer.
    pub state: StateInfo,

    /// Side to move.
    pub side_to_move: Color,

    /// Number of half-moves played.
    pub game_ply: i32,

    /// Bloom filter for fast repetition detection.
    pub filter: BloomFilter,

    /// Zobrist keys (initialized once globally).
    pub zobrist: &'static Zobrist,
}

impl Position {
    /// Create an empty position (no pieces). Must call [`Self::set_fen`] to initialize.
    pub fn new(zobrist: &'static Zobrist) -> Self {
        Position {
            board: [Piece::NO_PIECE; SQUARE_NB],
            piece_count: [0; PIECE_NB],
            mid_encoding: [0; 2],
            state: StateInfo::default(),
            side_to_move: Color::White,
            game_ply: 0,
            filter: BloomFilter::new(),
            zobrist,
        }
    }

    /// Empty board with global Zobrist keys (typical for library callers).
    pub fn new_with_global_zobrist() -> Self {
        Self::new(global_zobrist())
    }

    /// Parse FEN into a fresh position (global Zobrist).
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut pos = Self::new_with_global_zobrist();
        pos.set_fen(fen)?;
        Ok(pos)
    }

    // ── Piece access ─────────────────────────────────────────────────────────

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

    // ── Occupancy bitboard ───────────────────────────────────────────────────

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

    // ── Bitboard-based checkers / attackers ──────────────────────────────────

    /// All pieces that attack square `s`.
    pub fn attackers_to(&self, s: Square) -> Bitboard {
        let occupied = self.occupancy();
        let mut att = 0u128;

        // Pawn attackers (the "pawn_attacks_to" direction)
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

        // For each piece type, check if its attacks reach s
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

        // Advisor and King: use simple direction checks
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

    /// Pieces of `c` that give check to square `s`.
    pub fn checkers_to(&self, c: Color, s: Square) -> Bitboard {
        let occupied = self.occupancy();
        let mut att = 0u128;

        // Pawns
        let pawns = self.pieces_c_pt(c, PieceType::Pawn);
        let mut bb = pawns;
        while bb != 0 {
            let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
            if pawn_attacks(psq, c) & square_bb(s) != 0 {
                att |= square_bb(psq);
            }
            bb &= bb - 1;
        }

        // Knights
        let knights = self.pieces_c_pt(c, PieceType::Knight);
        let mut bb = knights;
        while bb != 0 {
            let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
            if knight_attacks(psq, occupied) & square_bb(s) != 0 {
                att |= square_bb(psq);
            }
            bb &= bb - 1;
        }

        // Rooks and Kings (flying general)
        let rooks_kings = self.pieces_c_pt(c, PieceType::Rook) | self.pieces_c_pt(c, PieceType::King);
        let mut bb = rooks_kings;
        while bb != 0 {
            let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
            if rook_attacks(psq, occupied) & square_bb(s) != 0 {
                att |= square_bb(psq);
            }
            bb &= bb - 1;
        }

        // Cannons
        let cannons = self.pieces_c_pt(c, PieceType::Cannon);
        let mut bb = cannons;
        while bb != 0 {
            let psq = unsafe { std::mem::transmute(bb.trailing_zeros() as u8) };
            if cannon_attacks(psq, occupied) & square_bb(s) != 0 {
                att |= square_bb(psq);
            }
            bb &= bb - 1;
        }

        att
    }

    /// Get checkers (pieces giving check to the current side's king).
    pub fn checkers(&self) -> Bitboard {
        self.state.checkers_bb
    }

    /// Update blockers and pinners for both sides.
    pub fn set_check_info(&mut self) {
        self.update_blockers(Color::White);
        self.update_blockers(Color::Black);

        let us = self.side_to_move;
        let them = !us;
        let ksq = self.king_square(them);
        let occupied = self.occupancy();

        // Hollow cannon detection
        self.state.need_full_check =
            self.checkers() != 0
                || (rook_attacks(self.king_square(us), 0) & self.pieces_c_pt(them, PieceType::Cannon) != 0);

        // Check squares: from opponent king's perspective, where would each piece type give check?
        self.state.check_squares[PieceType::Pawn as usize] = {
            let mut bb = 0u128;
            for &c in &[Color::White, Color::Black] {
                if c == us {
                    // Where can pawns of 'us' go to check the opponent king?
                    let _to_bb = pawn_attacks(ksq, them);
                    // Actually: from ksq, where would a pawn of us need to be?
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
            // Knight that attacks ksq
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

        // Hollow cannon discovery squares
        let hollow = self.state.check_squares[PieceType::Rook as usize]
            & self.pieces_c_pt(us, PieceType::Cannon);
        if hollow != 0 {
            let mut h = hollow;
            let mut discover = 0u128;
            while h != 0 {
                let csq = unsafe { std::mem::transmute(h.trailing_zeros() as u8) };
                let between = between_bb(csq, ksq);
                // Actually the hollow cannon gives discovered check on the squares between itself and ksq
                discover |= between;
                h &= h - 1;
            }
            for pt_val in 1..PIECE_TYPE_NB {
                self.state.check_squares[pt_val] |= discover;
            }
        }
    }

    /// Compute blockers_for_king[c] and pinners[!c].
    fn update_blockers(&mut self, c: Color) {
        let ksq = self.king_square(c);
        let them = !c;
        let occupied = self.occupancy();

        self.state.blockers_for_king[c as usize] = 0;
        self.state.pinners[them as usize] = 0;

        // Snipers: opponent pieces that attack ksq along open lines
        let snipers = {
            let rook_att = rook_attacks(ksq, 0);
            let knight_att = knight_attacks(ksq, 0);
            (rook_att & (self.piece_type_bb(PieceType::Rook) | self.piece_type_bb(PieceType::Cannon) | self.piece_type_bb(PieceType::King)))
                | (knight_att & self.piece_type_bb(PieceType::Knight))
        } & self.color_bb(them);

        let mut sniper_bb = snipers;
        while sniper_bb != 0 {
            let sniper_sq = unsafe { std::mem::transmute(sniper_bb.trailing_zeros() as u8) };
            let is_cannon = type_of(self.piece_on(sniper_sq)) == PieceType::Cannon;

            // Between ksq and sniper
            let between = between_bb(ksq, sniper_sq);
            let b = if is_cannon {
                between & (occupied ^ square_bb(sniper_sq))
            } else {
                between & occupied
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

    // ── Legality ─────────────────────────────────────────────────────────────

    /// Is a pseudo-legal move truly legal?
    pub fn legal(&self, m: Move) -> bool {
        assert!(m.is_ok());
        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq();
        let pc = self.piece_on(from);
        let occupied = (self.occupancy() ^ square_bb(from)) | square_bb(to);

        // King move: destination must not be attacked
        if type_of(pc) == PieceType::King {
            return self.checkers_to(them, to) & occupied == 0;
        }

        // Fast path: non-king moves that are clearly legal
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

        // General case: king must not be in check after the move
        let checkers = self.checkers_to(them, self.king_square(us));
        checkers & !square_bb(to) == 0
    }

    /// Check if a move gives check.
    pub fn gives_check(&self, m: Move) -> bool {
        assert!(m.is_ok());
        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq();
        let ksq = self.king_square(them);
        let pt = type_of(self.piece_on(from));

        // Direct check
        if pt == PieceType::Cannon {
            if self.state.check_squares[PieceType::Rook as usize] & square_bb(from) != 0
                && aligned(from, to, ksq)
            {
                if self.piece_on(to) != Piece::NO_PIECE {
                    // Capture: check if to is between the cannon and king
                    let ray = between_bb(ksq, from);
                    if ray & square_bb(to) != 0 {
                        return true;
                    }
                }
            }
        } else if self.state.check_squares[pt as usize] & square_bb(to) != 0 {
            return true;
        }

        // Discovered check: moving a blocker reveals a pinner
        if self.state.blockers_for_king[them as usize] & square_bb(from) != 0
            && (!aligned(from, to, ksq) || self.piece_on(to) != Piece::NO_PIECE)
        {
            return true;
        }

        false
    }

    // ── do_move / undo_move ──────────────────────────────────────────────────

    /// Execute a move. Assumes the move is legal.
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
        assert!(captured == Piece::NO_PIECE || type_of(captured) != PieceType::King,
            "Cannot capture the king!");

        // ── Snapshot old state values before modifying self.state ──
        let old_st = self.state.clone();
        let old_pawn_key = old_st.pawn_key;
        let old_minor_key = old_st.minor_piece_key;
        let old_non_pawn_key = old_st.non_pawn_key;
        let old_major_mat = old_st.major_material;
        let old_check10 = old_st.check10;
        let old_rule60 = old_st.rule60;
        let old_plies_from_null = old_st.plies_from_null;
        let old_key = old_st.key;

        // Bloom filter
        self.filter.set(old_key, self.filter.get(old_key).wrapping_add(1));

        // Save old state
        self.state.previous = Some(Box::new(old_st));

        // ── Initialize new state ──
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

        // ── Increment counters ──
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

        // ── Execute the move on the board ──
        if captured != Piece::NO_PIECE {
            self.remove_piece(to);
            self.state.captured_piece = captured;
        }
        self.move_piece(from, to);

        // ── Update Zobrist keys ──
        self.state.key ^= self.zobrist.psq[pc.0 as usize][from as usize]
            ^ self.zobrist.psq[pc.0 as usize][to as usize];
        if captured != Piece::NO_PIECE {
            self.state.key ^= self.zobrist.psq[captured.0 as usize][to as usize];
        }
        self.state.key ^= self.zobrist.side;

        // Update pawn/minor/non-pawn keys
        if type_of(pc) == PieceType::Pawn {
            self.state.pawn_key ^=
                self.zobrist.psq[pc.0 as usize][from as usize]
                ^ self.zobrist.psq[pc.0 as usize][to as usize];
            if captured != Piece::NO_PIECE && type_of(captured) == PieceType::Pawn {
                self.state.pawn_key ^= self.zobrist.psq[captured.0 as usize][to as usize];
            }
        } else {
            self.state.non_pawn_key[us as usize] ^=
                self.zobrist.psq[pc.0 as usize][from as usize]
                ^ self.zobrist.psq[pc.0 as usize][to as usize];
            if type_of(pc) as u8 & 1 != 0 && type_of(pc) != PieceType::Rook {
                self.state.minor_piece_key ^=
                    self.zobrist.psq[pc.0 as usize][from as usize]
                    ^ self.zobrist.psq[pc.0 as usize][to as usize];
            }
        }
        if captured != Piece::NO_PIECE {
            if type_of(captured) == PieceType::Pawn {
                self.state.pawn_key ^= self.zobrist.psq[captured.0 as usize][to as usize];
            } else {
                self.state.non_pawn_key[them as usize] ^=
                    self.zobrist.psq[captured.0 as usize][to as usize];
                if type_of(captured) as u8 & 1 != 0 {
                    self.state.major_material[them as usize] -=
                        PIECE_VALUE[captured.0 as usize];
                    if type_of(captured) != PieceType::Rook {
                        self.state.minor_piece_key ^=
                            self.zobrist.psq[captured.0 as usize][to as usize];
                    }
                }
            }
        }

        // Toggle side
        self.side_to_move = them;

        // Recompute check info
        self.state.checkers_bb = self.checkers_to(us, self.king_square(them));
        self.set_check_info();
    }

    /// Undo the last move, restoring the previous state.
    pub fn undo_move(&mut self, m: Move) {
        assert!(m.is_ok());

        let to = m.to_sq();
        let from = m.from_sq();
        let captured = self.state.captured_piece;

        // Move piece back
        self.move_piece(to, from);

        // Restore captured piece if any
        if captured != Piece::NO_PIECE {
            self.put_piece(captured, to);
        }

        // Pop previous state (this restores side, keys, checkers, etc.)
        if let Some(prev) = self.state.previous.take() {
            self.state = *prev;
        }

        // Restore side-to-move and ply (were in the previous state but StateInfo
        // doesn't store them — Position owns them directly)
        self.side_to_move = !self.side_to_move;
        self.game_ply -= 1;
    }

    // ── Board manipulation ───────────────────────────────────────────────────

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
        if from == to { return; }  // No-op, or error
        let pc = self.board[from as usize];
        self.board[from as usize] = Piece::NO_PIECE;
        self.board[to as usize] = pc;
    }

    // ── FEN I/O ──────────────────────────────────────────────────────────────

    /// Set position from a FEN string. Returns Err(msg) on failure.
    pub fn set_fen(&mut self, fen: &str) -> Result<(), String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 {
            return Err("Invalid FEN: too few fields".into());
        }

        // Clear board
        self.board = [Piece::NO_PIECE; SQUARE_NB];
        self.piece_count = [0; PIECE_NB];
        self.mid_encoding = [0; 2]; // TODO: BalanceEncoding

        // Parse piece placement (ranks 9 down to 0)
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
                    let sq = make_square(
                        unsafe { std::mem::transmute(file as u8) },
                        unsafe { std::mem::transmute(rank as u8) },
                    );
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

        // Side to move
        self.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("Invalid FEN: side to move must be 'w' or 'b'".into()),
        };

        // Rule60 counter (halfmove clock)
        let rule60: i32 = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        self.state.rule60 = rule60;

        // Fullmove number
        let fullmove: i32 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
        self.game_ply = std::cmp::max(2 * (fullmove - 1), 0) + (self.side_to_move == Color::Black) as i32;

        // Compute initial state
        self.init_state();

        Ok(())
    }

    /// Compute the initial Zobrist hash and other state fields.
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
                    // Major piece (rook, cannon, etc.)
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

        // Set checkers
        self.state.checkers_bb = self.checkers_to(them, self.king_square(us));

        // Set check info (blockers, pinners, check squares)
        self.set_check_info();
    }

    /// Generate a FEN string representing the current position.
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

    // ── Key access ───────────────────────────────────────────────────────────

    pub fn key(&self) -> Key {
        self.adjust_key60(self.state.key)
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

// ── Alignment check ──────────────────────────────────────────────────────────

/// Check if squares a, b, c are aligned (on the same row or column).
pub fn aligned(a: Square, b: Square, c: Square) -> bool {
    let ra = rank_of(a) as i32;
    let fa = file_of(a) as i32;
    let rb = rank_of(b) as i32;
    let fb = file_of(b) as i32;
    let rc = rank_of(c) as i32;
    let fc = file_of(c) as i32;

    // Same file
    if fa == fb && fb == fc {
        return true;
    }
    // Same rank
    if ra == rb && rb == rc {
        return true;
    }
    false
}
