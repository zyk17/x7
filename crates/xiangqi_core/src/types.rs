//! px0 `src/chess/types.h:31-222`。

use std::fmt;

pub const FILE_NB: u8 = 9;
pub const RANK_NB: u8 = 10;
pub const SQUARE_NB: u8 = FILE_NB * RANK_NB;

/// px0 `PieceType`：索引和字符表必须保持一致。
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum PieceType {
    Rook = 0,
    Advisor = 1,
    Cannon = 2,
    Pawn = 3,
    Knight = 4,
    Bishop = 5,
    King = 6,
    KnightTo = 7,
    PawnToOurs = 8,
    PawnToTheirs = 9,
}

impl PieceType {
    pub const PIECE_TYPE_NB: u8 = 7;

    pub const fn from_idx(idx: u8) -> Option<Self> {
        match idx {
            0 => Some(Self::Rook),
            1 => Some(Self::Advisor),
            2 => Some(Self::Cannon),
            3 => Some(Self::Pawn),
            4 => Some(Self::Knight),
            5 => Some(Self::Bishop),
            6 => Some(Self::King),
            7 => Some(Self::KnightTo),
            8 => Some(Self::PawnToOurs),
            9 => Some(Self::PawnToTheirs),
            _ => None,
        }
    }

    pub const fn from_fen_char(ch: char) -> Option<Self> {
        match ch {
            'r' | 'R' => Some(Self::Rook),
            'a' | 'A' => Some(Self::Advisor),
            'c' | 'C' => Some(Self::Cannon),
            'p' | 'P' => Some(Self::Pawn),
            'n' | 'N' => Some(Self::Knight),
            'b' | 'B' => Some(Self::Bishop),
            'k' | 'K' => Some(Self::King),
            _ => None,
        }
    }

    pub const fn fen_char(self, uppercase: bool) -> char {
        let ch = match self {
            Self::Rook => 'r',
            Self::Advisor => 'a',
            Self::Cannon => 'c',
            Self::Pawn => 'p',
            Self::Knight => 'n',
            Self::Bishop => 'b',
            Self::King => 'k',
            Self::KnightTo | Self::PawnToOurs | Self::PawnToTheirs => '?',
        };
        if uppercase {
            ch.to_ascii_uppercase()
        } else {
            ch
        }
    }

    /// px0 `PieceType::ToString` (`types.h:38-40`).
    pub fn to_string(self, uppercase: bool) -> String {
        self.fen_char(uppercase).to_string()
    }

    /// px0 `PieceType::IsValid` (`types.h:41`).
    pub const fn is_valid(self) -> bool {
        (self as u8) < Self::PIECE_TYPE_NB
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct File(u8);

impl File {
    pub const A: Self = Self(0);
    pub const B: Self = Self(1);
    pub const C: Self = Self(2);
    pub const D: Self = Self(3);
    pub const E: Self = Self(4);
    pub const F: Self = Self(5);
    pub const G: Self = Self(6);
    pub const H: Self = Self(7);
    pub const I: Self = Self(8);
    pub const INVALID: Self = Self(0x80);

    pub const fn from_idx(idx: u8) -> Option<Self> {
        if idx < FILE_NB {
            Some(Self(idx))
        } else {
            None
        }
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    pub const fn is_valid(self) -> bool {
        self.0 < FILE_NB
    }

    pub fn offset(self, delta: i32) -> Self {
        let idx = self.0 as i32 + delta;
        if idx < 0 || idx >= FILE_NB as i32 {
            Self::INVALID
        } else {
            Self(idx as u8)
        }
    }

    pub const fn parse(ch: char) -> Option<Self> {
        match ch {
            'a'..='i' => Self::from_idx(ch as u8 - b'a'),
            'A'..='I' => Self::from_idx(ch as u8 - b'A'),
            _ => None,
        }
    }

    pub const fn flip(self) -> Self {
        Self(8 - self.0)
    }

    /// px0 `File::ToString` (`types.h:71-73`).
    pub fn to_string(self, uppercase: bool) -> String {
        let base = if uppercase { b'A' } else { b'a' };
        char::from(base.wrapping_add(self.0)).to_string()
    }

    /// px0 `File::Flop` (`types.h:74`).
    pub fn flop_in_place(&mut self) {
        *self = self.flip();
    }
}

impl Default for File {
    /// px0 `File()` initializes to an off-board value (`types.h:65`).
    fn default() -> Self {
        Self::INVALID
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Rank(u8);

impl Rank {
    pub const R0: Self = Self(0);
    pub const R1: Self = Self(1);
    pub const R2: Self = Self(2);
    pub const R3: Self = Self(3);
    pub const R4: Self = Self(4);
    pub const R5: Self = Self(5);
    pub const R6: Self = Self(6);
    pub const R7: Self = Self(7);
    pub const R8: Self = Self(8);
    pub const R9: Self = Self(9);
    pub const INVALID: Self = Self(0x80);

    pub const fn from_idx(idx: u8) -> Option<Self> {
        if idx < RANK_NB {
            Some(Self(idx))
        } else {
            None
        }
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    pub const fn is_valid(self) -> bool {
        self.0 < RANK_NB
    }

    pub fn offset(self, delta: i32) -> Self {
        let idx = self.0 as i32 + delta;
        if idx < 0 || idx >= RANK_NB as i32 {
            Self::INVALID
        } else {
            Self(idx as u8)
        }
    }

    pub const fn parse(ch: char) -> Option<Self> {
        match ch {
            '0'..='9' => Self::from_idx(ch as u8 - b'0'),
            _ => None,
        }
    }

    pub const fn flip(self) -> Self {
        Self(9 - self.0)
    }

    /// px0 `Rank::ToString` (`types.h:106`).
    pub fn as_text(self) -> String {
        char::from(b'0'.wrapping_add(self.0)).to_string()
    }

    /// px0 `Rank::Flip` (`types.h:105`).
    pub fn flip_in_place(&mut self) {
        *self = self.flip();
    }
}

/// px0 square order：`a0 = 0`，每 rank 连续 9 格。
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Square(u8);

impl Square {
    pub const INVALID: Self = Self(u8::MAX);

    pub const fn new(file: File, rank: Rank) -> Self {
        if !file.is_valid() || !rank.is_valid() {
            return Self::INVALID;
        }
        let idx = rank.index() as u16 * FILE_NB as u16 + file.index() as u16;
        if idx < SQUARE_NB as u16 {
            Self(idx as u8)
        } else {
            Self::INVALID
        }
    }

    pub const fn from_idx(idx: u8) -> Option<Self> {
        if idx < SQUARE_NB {
            Some(Self(idx))
        } else {
            None
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        let bytes = text.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = File::parse(bytes[0] as char)?;
        let rank = Rank::parse(bytes[1] as char)?;
        Some(Self::new(file, rank))
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    pub const fn file(self) -> Option<File> {
        File::from_idx(self.0 % FILE_NB)
    }

    pub const fn rank(self) -> Option<Rank> {
        Rank::from_idx(self.0 / FILE_NB)
    }

    pub const fn flip(self) -> Self {
        match (self.file(), self.rank()) {
            (Some(file), Some(rank)) => Self::new(file, rank.flip()),
            _ => Self::INVALID,
        }
    }

    pub fn flip_in_place(&mut self) {
        *self = self.flip();
    }

    /// px0 `Square::ToString` (`types.h:126-128`).
    pub fn to_string(self, uppercase: bool) -> String {
        match (self.file(), self.rank()) {
            (Some(file), Some(rank)) => format!("{}{}", file.to_string(uppercase), rank.as_text()),
            _ => "--".to_owned(),
        }
    }

    pub fn is_valid(self) -> bool {
        self.index() < SQUARE_NB
    }

    /// px0 `types.h:135-143`：`(rank_delta, file_delta)`。
    pub fn offset(self, rank_delta: i32, file_delta: i32) -> Self {
        match (self.file(), self.rank()) {
            (Some(file), Some(rank)) => {
                let file = file.offset(file_delta);
                let rank = rank.offset(rank_delta);
                if !file.is_valid() || !rank.is_valid() {
                    Self::INVALID
                } else {
                    Self::new(file, rank)
                }
            }
            _ => Self::INVALID,
        }
    }

    pub fn offset_by(self, direction: Direction) -> Self {
        self.offset(direction.0, direction.1)
    }
}

/// px0 `board.cc:79`：`(rank_delta, file_delta)`。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Direction(pub i32, pub i32);

pub const NORTH: Direction = Direction(1, 0);
pub const EAST: Direction = Direction(0, 1);
pub const SOUTH: Direction = Direction(-1, 0);
pub const WEST: Direction = Direction(0, -1);
pub const NORTH_WEST: Direction = Direction(1, -1);
pub const NORTH_EAST: Direction = Direction(1, 1);
pub const SOUTH_WEST: Direction = Direction(-1, -1);
pub const SOUTH_EAST: Direction = Direction(-1, 1);

pub fn file_distance(a: File, b: File) -> i32 {
    (a.index() as i32 - b.index() as i32).abs()
}

pub fn rank_distance(a: Rank, b: Rank) -> i32 {
    (a.index() as i32 - b.index() as i32).abs()
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.file(), self.rank()) {
            (Some(file), Some(rank)) => write!(f, "{}{}", (b'a' + file.index()) as char, rank.index()),
            _ => f.write_str("--"),
        }
    }
}

/// px0 `Move`：to 在低 7 位，from 在 bit 7-13。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Move(u16);

impl Move {
    pub const NULL: Self = Self(0);

    pub const fn new(from: Square, to: Square) -> Self {
        Self(((from.index() as u16) << 7) | to.index() as u16)
    }

    /// px0 `Move::White` (`types.h:158-160`).
    pub const fn white(from: Square, to: Square) -> Self {
        Self::new(from, to)
    }

    pub const fn from(self) -> Square {
        match Square::from_idx(((self.0 >> 7) & 0x7f) as u8) {
            Some(square) => square,
            None => Square::INVALID,
        }
    }

    pub const fn to(self) -> Square {
        match Square::from_idx((self.0 & 0x7f) as u8) {
            Some(square) => square,
            None => Square::INVALID,
        }
    }

    pub const fn raw(self) -> u16 {
        self.0
    }

    /// px0 `Move::raw_data` (`types.h:178`).
    pub const fn raw_data(self) -> u16 {
        self.raw()
    }

    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    pub const fn flip(self) -> Self {
        Self::new(self.from().flip(), self.to().flip())
    }

    /// px0 `Move::Flip` (`types.h:166-171`).
    pub fn flip_in_place(&mut self) {
        *self = self.flip();
    }

    /// px0 `Move::ToString` (`types.h:210-212`).
    pub fn to_uci(self) -> String {
        format!("{}{}", self.from().to_string(false), self.to().to_string(false))
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from(), self.to())
    }
}

pub type MoveList = Vec<Move>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_and_move_match_px0_layout() {
        let a0 = Square::new(File::A, Rank::from_idx(0).unwrap());
        let i9 = Square::new(File::I, Rank::from_idx(9).unwrap());
        assert_eq!(a0.index(), 0);
        assert_eq!(i9.index(), 89);
        let mv = Move::new(a0, i9);
        assert_eq!(mv.from(), a0);
        assert_eq!(mv.to(), i9);
        assert_eq!(mv.to_string(), "a0i9");
    }

    #[test]
    fn scalar_types_match_px0_helpers() {
        assert!(!File::default().is_valid());
        assert_eq!(File::H.to_string(false), "h");
        assert_eq!(File::H.to_string(true), "H");
        assert_eq!(Rank::R8.as_text(), "8");
        assert_eq!(PieceType::Cannon.to_string(true), "C");
        assert!(PieceType::King.is_valid());
        assert!(!PieceType::KnightTo.is_valid());

        let mut mv = Move::white(Square::new(File::B, Rank::R2), Square::new(File::H, Rank::R7));
        assert_eq!(mv.to_uci(), "b2h7");
        mv.flip_in_place();
        assert_eq!(mv.to_uci(), "b7h2");
    }
}
