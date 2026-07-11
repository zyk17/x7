//! px0 `src/chess/bitboard.h:31-174` 与 `utils/bititer.h:39-121`。

use crate::types::{File, Rank, Square};

/// px0 `bititer.h:78-89`。
pub fn mirror_board(bits: u128) -> u128 {
    const SEQ1: u128 = 0x00001FFFFFFFFFFF;
    const SEQ2: u128 = (0x00000000000000FFu128 << 64) | 0x8000000007FC0000;
    const SEQ3: u128 = 0x7FFFE0000003FFFF;
    const SEQ4: u128 = (0x000000000001FF00u128 << 64) | 0x003FE00FF80001FF;

    let mut v = bits;
    v = ((v & SEQ1) << 45) | ((v >> 45) & SEQ1);
    let fixed = v & SEQ2;
    v = ((v & SEQ3) << 27) | ((v >> 27) & SEQ3);
    v = ((v & SEQ4) << 9) | ((v >> 9) & SEQ4);
    v | fixed
}

fn lowest_bit(value: u128) -> u32 {
    if value as u64 != 0 {
        (value as u64).trailing_zeros()
    } else {
        (value >> 64).trailing_zeros() + 64
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct BitBoard(u128);

impl BitBoard {
    pub const EMPTY: Self = Self(0);

    pub const fn from_square(square: Square) -> Self {
        if square.index() < 90 {
            Self(1u128 << square.index())
        } else {
            Self::EMPTY
        }
    }

    pub const fn from_bits(bits: u128) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u128 {
        self.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// px0 `bitboard.h:76-88`（NO_POPCNT 稀疏路径）。
    pub fn count_few(self) -> u32 {
        let mut x = self.0;
        let mut count = 0u32;
        while x != 0 {
            count += 1;
            x &= x - 1;
        }
        count
    }

    pub const fn contains(self, square: Square) -> bool {
        square.index() < 90 && (self.0 & (1u128 << square.index())) != 0
    }

    pub fn set(&mut self, square: Square) {
        if square.index() < 90 {
            self.0 |= 1u128 << square.index();
        }
    }

    pub fn reset(&mut self, square: Square) {
        if square.index() < 90 {
            self.0 &= !(1u128 << square.index());
        }
    }

    /// px0 `bitboard.h:92-94`。
    pub fn set_if(&mut self, square: Square, cond: bool) {
        if cond && square.index() < 90 {
            self.0 |= 1u128 << square.index();
        }
    }

    pub fn clear(&mut self) {
        self.0 = 0;
    }

    /// px0 `bitboard.h:111`。
    pub fn mirror(&mut self) {
        self.0 = mirror_board(self.0);
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub fn subtract_square(self, square: Square) -> Self {
        if square.index() < 90 {
            Self(self.0 & !(1u128 << square.index()))
        } else {
            self
        }
    }

    pub fn debug_string(self) -> String {
        let mut res = String::new();
        for rank in (0..10).rev() {
            for file in 0..9 {
                let square = Square::new(File::from_idx(file).unwrap(), Rank::from_idx(rank).unwrap());
                res.push(if self.contains(square) { '#' } else { '.' });
            }
            res.push('\n');
        }
        res
    }

    pub fn iter(self) -> BitBoardIter {
        BitBoardIter { value: self.0 }
    }
}

impl std::ops::BitOrAssign for BitBoard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAndAssign for BitBoard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::SubAssign for BitBoard {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 &= !rhs.0;
    }
}

impl IntoIterator for BitBoard {
    type Item = Square;
    type IntoIter = BitBoardIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// px0 `bititer.h:94-110`。
pub struct BitBoardIter {
    value: u128,
}

impl Iterator for BitBoardIter {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.value == 0 {
            return None;
        }
        let bit = lowest_bit(self.value);
        self.value &= self.value - 1;
        Square::from_idx(bit as u8)
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{File, Rank, Square};

    use super::*;

    #[test]
    fn stores_all_ninety_squares() {
        let a0 = Square::new(File::A, Rank::from_idx(0).unwrap());
        let i9 = Square::new(File::I, Rank::from_idx(9).unwrap());
        let board = BitBoard::from_square(a0).union(BitBoard::from_square(i9));
        assert!(board.contains(a0));
        assert!(board.contains(i9));
        assert_eq!(board.count(), 2);
    }

    #[test]
    fn iter_visits_set_bits() {
        let a0 = Square::new(File::A, Rank::from_idx(0).unwrap());
        let i9 = Square::new(File::I, Rank::from_idx(9).unwrap());
        let squares: Vec<_> = BitBoard::from_square(a0)
            .union(BitBoard::from_square(i9))
            .into_iter()
            .collect();
        assert_eq!(squares, vec![a0, i9]);
    }
}
