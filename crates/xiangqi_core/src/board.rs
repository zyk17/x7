//! 棋盘表示、FEN 与合法着。来源：px0 board。

use std::sync::OnceLock;

use crate::bitboard::BitBoard;
use crate::board_attacks::get_attacks;
use crate::board_masks::{ADVISOR_SQUARES, PALACE, bishop_bb, pawn_bb};
use crate::hashcat::hash_cat_u128s;
use crate::{CoreError, File, Move, MoveList, PieceType, Rank, Square};

pub use crate::board_attacks::initialize_magic_bitboards;

pub const STARTPOS_FEN: &str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

static STARTPOS_BOARD: OnceLock<ChessBoard> = OnceLock::new();

pub fn startpos_board() -> &'static ChessBoard {
    STARTPOS_BOARD.get_or_init(|| ChessBoard::from_fen(STARTPOS_FEN).expect("valid startpos FEN").0)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FenState {
    pub rule60_ply: u32,
    pub game_ply: u32,
}

/// 棋盘始终以当前行棋方为 `ours` 视角保存。
#[derive(Clone, Debug)]
pub struct ChessBoard {
    ours: BitBoard,
    theirs: BitBoard,
    rooks: BitBoard,
    advisors: BitBoard,
    cannons: BitBoard,
    pawns: BitBoard,
    knights: BitBoard,
    bishops: BitBoard,
    our_king: Square,
    their_king: Square,
    flipped: bool,
    rule_id: [u8; 90],
}

impl PartialEq for ChessBoard {
    fn eq(&self, other: &Self) -> bool {
        self.flipped == other.flipped &&
        self.our_king == other.our_king &&
        self.their_king == other.their_king &&
        self.ours == other.ours &&
        self.theirs == other.theirs &&
        self.rooks == other.rooks &&
        self.advisors == other.advisors &&
        self.cannons == other.cannons &&
        self.pawns == other.pawns &&
        self.knights == other.knights &&
        self.bishops == other.bishops
    }
}

impl Eq for ChessBoard {}

impl Default for ChessBoard {
    fn default() -> Self {
        Self {
            ours: BitBoard::EMPTY,
            theirs: BitBoard::EMPTY,
            rooks: BitBoard::EMPTY,
            advisors: BitBoard::EMPTY,
            cannons: BitBoard::EMPTY,
            pawns: BitBoard::EMPTY,
            knights: BitBoard::EMPTY,
            bishops: BitBoard::EMPTY,
            our_king: Square::INVALID,
            their_king: Square::INVALID,
            flipped: false,
            rule_id: [0; 90],
        }
    }
}

impl ChessBoard {
    pub fn from_fen(fen: &str) -> Result<(Self, FenState), CoreError> {
        initialize_magic_bitboards();
        let mut board = Self::default();
        let mut state = FenState {
            rule60_ply: 0,
            game_ply: 1,
        };

        let mut rank = Rank::R9;
        let mut file_idx = 0i32;
        let mut pos = 0usize;
        let bytes = fen.as_bytes();

        let complain = |msg: &str| CoreError::InvalidFen(format!("{msg}: {fen}"));

        fn skip_whitespace(
            bytes: &[u8],
            pos: &mut usize,
            where_at: Option<&str>,
            fen: &str,
        ) -> Result<bool, CoreError> {
            if let Some(where_at) = where_at
                && *pos < bytes.len()
                && bytes[*pos] != b' '
            {
                return Err(CoreError::InvalidFen(format!("space expected {where_at}: {fen}")));
            }
            while *pos < bytes.len() && bytes[*pos] == b' ' {
                *pos += 1;
            }
            Ok(*pos == bytes.len())
        }

        skip_whitespace(bytes, &mut pos, None, fen)?;

        while pos < bytes.len() {
            let c = bytes[pos] as char;
            if c == ' ' {
                break;
            }
            if c == '/' {
                if rank.index() == 0 {
                    return Err(complain("too many ranks"));
                }
                rank = Rank::from_idx(rank.index() - 1).unwrap();
                file_idx = 0;
                pos += 1;
                continue;
            }
            if c.is_ascii_digit() {
                file_idx += (c as u8 - b'0') as i32;
                if file_idx > 9 {
                    return Err(complain("too many files"));
                }
                pos += 1;
                continue;
            }
            let piece = PieceType::from_fen_char(c)
                .filter(|p| p.is_valid())
                .ok_or_else(|| complain("invalid character as piece"))?;
            let file = File::from_idx(file_idx as u8);
            if file.is_none() || !rank.is_valid() {
                return Err(complain("piece out of board"));
            }
            let file = file.unwrap();
            let sq = Square::new(file, rank);
            if piece == PieceType::Advisor
                && BitBoard::from_square(sq)
                    .intersection(BitBoard::from_bits(ADVISOR_SQUARES))
                    .is_empty()
            {
                return Err(complain("advisor not on an advisor square"));
            } else if piece == PieceType::King
                && BitBoard::from_square(sq)
                    .intersection(BitBoard::from_bits(PALACE))
                    .is_empty()
            {
                return Err(complain("king not in palace"));
            } else if piece == PieceType::Pawn {
                let is_theirs = c.is_ascii_lowercase();
                if !BitBoard::from_square(sq).difference(pawn_bb(is_theirs)).is_empty() {
                    return Err(complain("pawn in wrong place"));
                }
            } else if piece == PieceType::Bishop && !BitBoard::from_square(sq).difference(bishop_bb()).is_empty() {
                return Err(complain("bishop in wrong place"));
            }

            board.put_piece(sq, piece, c.is_ascii_lowercase());
            file_idx += 1;
            pos += 1;
        }

        fn validate_board(board: &ChessBoard, fen: &str) -> Result<(), CoreError> {
            if board.is_valid() {
                Ok(())
            } else {
                Err(CoreError::InvalidFen(format!("inconsistent board: {fen}")))
            }
        }

        if skip_whitespace(bytes, &mut pos, Some("after the board"), fen)? {
            validate_board(&board, fen)?;
            return Ok((board, state));
        }

        let mut our = 0u8;
        let mut their = 0u8;
        for sq in (board.ours.union(board.theirs)).into_iter() {
            board.rule_id[sq.index() as usize] = if board.ours.contains(sq) {
                let id = our;
                our += 1;
                id
            } else {
                let id = their;
                their += 1;
                id
            };
        }

        let side_to_move = bytes[pos].to_ascii_lowercase();
        pos += 1;
        if side_to_move == b'b' {
            board.mirror();
        } else if side_to_move != b'w' {
            return Err(complain("invalid side to move"));
        }
        if skip_whitespace(bytes, &mut pos, Some("after side to move"), fen)? {
            validate_board(&board, fen)?;
            return Ok((board, state));
        }

        if bytes[pos] == b'-' {
            pos += 1;
        }
        if skip_whitespace(bytes, &mut pos, Some("after castling"), fen)? {
            validate_board(&board, fen)?;
            return Ok((board, state));
        }

        if bytes[pos] == b'-' {
            pos += 1;
        }
        if skip_whitespace(bytes, &mut pos, Some("after en passant"), fen)? {
            validate_board(&board, fen)?;
            return Ok((board, state));
        }

        fn parse_int(fen: &str, pos: &mut usize, error_msg: &str) -> Result<u32, CoreError> {
            let end = fen[*pos..].find(' ').map(|idx| *pos + idx).unwrap_or(fen.len());
            let num = &fen[*pos..end];
            let value: u32 = num
                .parse()
                .map_err(|_| CoreError::InvalidFen(format!("{error_msg}: {fen}")))?;
            *pos = end;
            Ok(value)
        }

        state.rule60_ply = parse_int(fen, &mut pos, "bad rule 60 halfmoves")?;
        if skip_whitespace(bytes, &mut pos, Some("after rule-60 clock"), fen)? {
            validate_board(&board, fen)?;
            return Ok((board, state));
        }

        state.game_ply = parse_int(fen, &mut pos, "bad total moves")?;
        if !skip_whitespace(bytes, &mut pos, Some("after total moves"), fen)? {
            return Err(complain("extra characters"));
        }

        validate_board(&board, fen)?;
        Ok((board, state))
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn mirror(&mut self) {
        self.ours.mirror();
        self.theirs.mirror();
        std::mem::swap(&mut self.ours, &mut self.theirs);
        self.rooks.mirror();
        self.advisors.mirror();
        self.cannons.mirror();
        self.pawns.mirror();
        self.knights.mirror();
        self.bishops.mirror();
        self.our_king.flip_in_place();
        self.their_king.flip_in_place();
        std::mem::swap(&mut self.our_king, &mut self.their_king);
        self.flipped = !self.flipped;
    }

    pub fn generate_pseudolegal_moves(&self) -> MoveList {
        initialize_magic_bitboards();
        let occupied = self.ours.union(self.theirs);
        let mut result = Vec::with_capacity(60);

        for source in self.ours.into_iter() {
            if self.rooks.contains(source) {
                for destination in get_attacks(PieceType::Rook, source, occupied)
                    .difference(self.ours)
                    .into_iter()
                {
                    result.push(Move::new(source, destination));
                }
                continue;
            }
            if self.advisors.contains(source) {
                for destination in get_attacks(PieceType::Advisor, source, BitBoard::EMPTY)
                    .difference(self.ours)
                    .into_iter()
                {
                    result.push(Move::new(source, destination));
                }
                continue;
            }
            if self.cannons.contains(source) {
                let attacks = get_attacks(PieceType::Rook, source, occupied)
                    .difference(occupied)
                    .union(get_attacks(PieceType::Cannon, source, occupied).intersection(self.theirs));
                for destination in attacks.into_iter() {
                    result.push(Move::new(source, destination));
                }
                continue;
            }
            if self.pawns.contains(source) {
                for destination in get_attacks(PieceType::Pawn, source, BitBoard::EMPTY)
                    .difference(self.ours)
                    .into_iter()
                {
                    result.push(Move::new(source, destination));
                }
                continue;
            }
            if self.knights.contains(source) {
                for destination in get_attacks(PieceType::Knight, source, occupied)
                    .difference(self.ours)
                    .into_iter()
                {
                    result.push(Move::new(source, destination));
                }
                continue;
            }
            if self.bishops.contains(source) {
                for destination in get_attacks(PieceType::Bishop, source, occupied)
                    .difference(self.ours)
                    .into_iter()
                {
                    result.push(Move::new(source, destination));
                }
                continue;
            }
            if source == self.our_king {
                for destination in get_attacks(PieceType::King, source, BitBoard::EMPTY)
                    .difference(self.ours)
                    .into_iter()
                {
                    result.push(Move::new(source, destination));
                }
            }
        }

        result
    }

    pub fn apply_move(&mut self, mv: Move) -> bool {
        let from = mv.from();
        let to = mv.to();

        self.ours.reset(from);
        self.ours.set(to);

        let reset_60_moves = self.theirs.contains(to);
        if reset_60_moves {
            self.reset_captured_piece(to);
        }

        if from == self.our_king {
            self.our_king = to;
            return reset_60_moves;
        }

        self.move_piece_marker(from, to);
        self.clear_piece_marker(from);

        let mut from_id = from;
        let mut to_id = to;
        if self.flipped {
            from_id.flip_in_place();
            to_id.flip_in_place();
        }
        self.rule_id[to_id.index() as usize] = self.rule_id[from_id.index() as usize];
        self.rule_id[from_id.index() as usize] = 0;

        reset_60_moves
    }

    pub fn is_legal_move(&self, mv: Move) -> bool {
        self.is_legal_move_for::<true>(mv)
    }

    fn is_legal_move_for<const OUR: bool>(&self, mv: Move) -> bool {
        let mut occupied = self.ours.union(self.theirs);
        occupied.reset(mv.from());
        occupied.set(mv.to());

        let mut our_king = self.our_king;
        let mut their_king = self.their_king;
        if !OUR {
            std::mem::swap(&mut our_king, &mut their_king);
        }

        let ksq = if our_king == mv.from() { mv.to() } else { our_king };
        if get_attacks(PieceType::Rook, ksq, occupied).contains(their_king) {
            return false;
        }

        if ksq != our_king {
            return self.checkers_to::<OUR>(ksq, occupied).is_empty();
        }

        let mut checkers = self.checkers_to::<OUR>(ksq, occupied);
        checkers.reset(mv.to());
        checkers.is_empty()
    }

    pub fn generate_legal_moves(&self) -> MoveList {
        self.generate_pseudolegal_moves()
            .into_iter()
            .filter(|mv| self.is_legal_move(*mv))
            .collect()
    }

    pub fn is_under_check(&self) -> bool {
        !self
            .checkers_to::<true>(self.our_king, self.ours.union(self.theirs))
            .is_empty()
    }

    fn checkers_to<const OUR: bool>(&self, ksq: Square, occupied: BitBoard) -> BitBoard {
        let mut checkers = get_attacks(PieceType::Rook, ksq, occupied).intersection(self.rooks);
        checkers = checkers.union(get_attacks(PieceType::Cannon, ksq, occupied).intersection(self.cannons));
        let pawn_pt = if OUR {
            PieceType::PawnToOurs
        } else {
            PieceType::PawnToTheirs
        };
        checkers = checkers.union(get_attacks(pawn_pt, ksq, BitBoard::EMPTY).intersection(self.pawns));
        checkers = checkers.union(get_attacks(PieceType::KnightTo, ksq, occupied).intersection(self.knights));
        checkers.intersection(if OUR { self.theirs } else { self.ours })
    }

    pub fn recaptures_to(&self, sq: Square) -> BitBoard {
        let occupied = self.ours.union(self.theirs);
        let mut attackers = get_attacks(PieceType::Rook, sq, occupied).intersection(self.rooks);
        attackers = attackers.union(get_attacks(PieceType::Advisor, sq, BitBoard::EMPTY).intersection(self.advisors));
        attackers = attackers.union(get_attacks(PieceType::Cannon, sq, occupied).intersection(self.cannons));
        attackers = attackers.union(get_attacks(PieceType::PawnToOurs, sq, BitBoard::EMPTY).intersection(self.pawns));
        attackers = attackers.union(get_attacks(PieceType::KnightTo, sq, occupied).intersection(self.knights));
        attackers = attackers.union(get_attacks(PieceType::Bishop, sq, occupied).intersection(self.bishops));
        attackers = attackers.union(
            get_attacks(PieceType::King, sq, BitBoard::EMPTY).intersection(BitBoard::from_square(self.their_king)),
        );
        attackers.intersection(self.theirs)
    }

    pub fn has_mating_material(&self) -> bool {
        if self.pawns.count() == 0 && self.rooks.count() == 0 && self.knights.count() == 0 {
            let level = mating_draw_level(self);
            if level != DrawLevel::No {
                if level == DrawLevel::Mate {
                    for mv in self.generate_legal_moves() {
                        let mut after = self.clone();
                        after.apply_move(mv);
                        after.mirror();
                        if after.generate_legal_moves().is_empty() {
                            return true;
                        }
                    }
                }
                return false;
            }
        }
        true
    }

    pub fn us_chased(&self) -> u16 {
        let mut chase = 0u16;

        let mut add_chase = |attacker_type: PieceType, attacker: BitBoard| {
            for from in (attacker.intersection(self.ours)).into_iter() {
                let mut attacks =
                    get_attacks(attacker_type, from, self.ours.union(self.theirs)).intersection(self.theirs);
                attacks = attacks.difference(self.kings());
                attacks = attacks.difference(
                    self.pawns
                        .intersection(BitBoard::from_bits(crate::board_masks::HALF_BB[1])),
                );

                let mut candidates = BitBoard::EMPTY;
                if matches!(attacker_type, PieceType::Knight | PieceType::Cannon) {
                    candidates = attacks.intersection(self.rooks);
                }
                if matches!(attacker_type, PieceType::Advisor | PieceType::Bishop) {
                    candidates = attacks.intersection(self.rooks.union(self.knights).union(self.cannons));
                }
                attacks = attacks.difference(candidates);
                for to in candidates.into_iter() {
                    if self.is_legal_move(Move::new(from, to)) {
                        chase |= self.make_chase(to);
                    }
                }

                for to in attacks.into_iter() {
                    let mv = Move::new(from, to);
                    if !self.is_legal_move(mv) {
                        continue;
                    }
                    let mut true_chase = true;
                    let mut after = self.clone();
                    after.apply_move(mv);
                    let recaptures = after.recaptures_to(to);
                    for s in recaptures.into_iter() {
                        if after.is_legal_move_for::<false>(Move::new(s, to)) {
                            true_chase = false;
                            break;
                        }
                    }
                    if !true_chase {
                        continue;
                    }
                    if attacker.contains(to) {
                        let pin = attacker_type == PieceType::Knight
                            && !get_attacks(PieceType::Knight, to, self.ours.union(self.theirs)).contains(from);
                        if pin || !self.is_legal_move_for::<false>(Move::new(to, from)) {
                            chase |= self.make_chase(to);
                        }
                    } else {
                        chase |= self.make_chase(to);
                    }
                }
            }
        };

        add_chase(PieceType::Rook, self.rooks);
        add_chase(PieceType::Advisor, self.advisors);
        add_chase(PieceType::Cannon, self.cannons);
        add_chase(PieceType::Knight, self.knights);
        add_chase(PieceType::Bishop, self.bishops);
        chase
    }

    pub fn them_chased(&self) -> u16 {
        let mut board = self.clone();
        board.mirror();
        board.us_chased()
    }

    fn make_chase(&self, mut to: Square) -> u16 {
        if self.flipped {
            to.flip_in_place();
        }
        1u16 << self.rule_id[to.index() as usize]
    }

    pub fn parse_move(&self, move_str: &str) -> Result<Move, CoreError> {
        if move_str.len() != 4 {
            return Err(CoreError::InvalidFen(format!("invalid move: {move_str}")));
        }
        let bytes = move_str.as_bytes();
        let from_file =
            File::parse(bytes[0] as char).ok_or_else(|| CoreError::InvalidFen(format!("invalid move: {move_str}")))?;
        let mut from_rank =
            Rank::parse(bytes[1] as char).ok_or_else(|| CoreError::InvalidFen(format!("invalid move: {move_str}")))?;
        let to_file =
            File::parse(bytes[2] as char).ok_or_else(|| CoreError::InvalidFen(format!("invalid move: {move_str}")))?;
        let mut to_rank =
            Rank::parse(bytes[3] as char).ok_or_else(|| CoreError::InvalidFen(format!("invalid move: {move_str}")))?;
        if !from_file.is_valid() || !from_rank.is_valid() || !to_file.is_valid() || !to_rank.is_valid() {
            return Err(CoreError::InvalidFen(format!("invalid move: {move_str}")));
        }
        if self.flipped {
            from_rank = from_rank.flip();
            to_rank = to_rank.flip();
        }
        let from = Square::new(from_file, from_rank);
        let to = Square::new(to_file, to_rank);
        if !self.ours.contains(from) {
            return Err(CoreError::InvalidFen(format!("invalid move: {move_str}")));
        }
        Ok(Move::new(from, to))
    }

    pub fn debug_string(&self) -> String {
        format!("https://xiangqiai.com/#/{}", board_to_fen(self))
    }

    pub const fn ours(&self) -> BitBoard {
        self.ours
    }
    pub const fn theirs(&self) -> BitBoard {
        self.theirs
    }
    pub const fn rooks(&self) -> BitBoard {
        self.rooks
    }
    pub const fn advisors(&self) -> BitBoard {
        self.advisors
    }
    pub const fn cannons(&self) -> BitBoard {
        self.cannons
    }
    pub const fn pawns(&self) -> BitBoard {
        self.pawns
    }
    pub const fn knights(&self) -> BitBoard {
        self.knights
    }
    pub const fn bishops(&self) -> BitBoard {
        self.bishops
    }
    pub fn kings(&self) -> BitBoard {
        BitBoard::from_square(self.our_king).union(BitBoard::from_square(self.their_king))
    }
    pub const fn our_king(&self) -> Square {
        self.our_king
    }
    pub const fn their_king(&self) -> Square {
        self.their_king
    }
    pub const fn flipped(&self) -> bool {
        self.flipped
    }
    pub fn hash(&self) -> u64 {
        let meta =
            ((self.our_king.index() as u128) << 16) | ((self.their_king.index() as u128) << 8) | (self.flipped as u128);
        hash_cat_u128s(&[
            self.ours.bits(),
            self.theirs.bits(),
            self.rooks.bits(),
            self.advisors.bits(),
            self.cannons.bits(),
            self.pawns.bits(),
            self.knights.bits(),
            self.bishops.bits(),
            meta,
        ])
    }

    fn put_piece(&mut self, square: Square, piece: PieceType, is_theirs: bool) {
        if is_theirs {
            self.theirs.set(square);
        } else {
            self.ours.set(square);
        }
        match piece {
            PieceType::Rook => self.rooks.set(square),
            PieceType::Advisor => self.advisors.set(square),
            PieceType::Cannon => self.cannons.set(square),
            PieceType::Pawn => self.pawns.set(square),
            PieceType::Knight => self.knights.set(square),
            PieceType::Bishop => self.bishops.set(square),
            PieceType::King => {
                if is_theirs {
                    self.their_king = square;
                } else {
                    self.our_king = square;
                }
            }
            _ => {}
        }
    }

    fn reset_captured_piece(&mut self, square: Square) {
        self.theirs.reset(square);
        self.clear_piece_marker(square);
    }

    fn move_piece_marker(&mut self, from: Square, to: Square) {
        self.rooks.set_if(to, self.rooks.contains(from));
        self.advisors.set_if(to, self.advisors.contains(from));
        self.cannons.set_if(to, self.cannons.contains(from));
        self.pawns.set_if(to, self.pawns.contains(from));
        self.knights.set_if(to, self.knights.contains(from));
        self.bishops.set_if(to, self.bishops.contains(from));
    }

    fn clear_piece_marker(&mut self, square: Square) {
        self.rooks.reset(square);
        self.advisors.reset(square);
        self.cannons.reset(square);
        self.pawns.reset(square);
        self.knights.reset(square);
        self.bishops.reset(square);
    }

    fn is_valid(&self) -> bool {
        let all = self.ours().union(self.theirs());
        let bbs = [
            self.rooks(),
            self.advisors(),
            self.cannons(),
            self.pawns(),
            self.knights(),
            self.bishops(),
            self.kings(),
        ];
        let union: BitBoard = bbs.iter().copied().fold(BitBoard::EMPTY, |a, b| a.union(b));
        if union != all {
            return false;
        }
        if !self
            .advisors()
            .difference(BitBoard::from_bits(ADVISOR_SQUARES))
            .is_empty()
        {
            return false;
        }
        for i in 0..bbs.len() {
            for j in (i + 1)..bbs.len() {
                if bbs[i].intersects(bbs[j]) {
                    return false;
                }
            }
        }
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DrawLevel {
    No,
    Direct,
    Mate,
}

fn mating_draw_level(board: &ChessBoard) -> DrawLevel {
    if board.cannons().count() == 0 {
        return DrawLevel::Direct;
    }
    if board.cannons().count() == 1 {
        let mut cannon_side_occ = board.ours();
        let mut non_cannon_side_occ = board.theirs();
        if cannon_side_occ.intersection(board.cannons()).is_empty() {
            std::mem::swap(&mut cannon_side_occ, &mut non_cannon_side_occ);
        }
        if board.advisors().intersection(cannon_side_occ).is_empty() {
            if board.advisors().intersection(non_cannon_side_occ).is_empty() {
                return DrawLevel::Direct;
            }
            if board.advisors().intersection(non_cannon_side_occ).count() == 1 {
                return if board.bishops().intersection(cannon_side_occ).is_empty() {
                    DrawLevel::Direct
                } else {
                    DrawLevel::Mate
                };
            }
            if board.bishops().intersection(cannon_side_occ).is_empty() {
                return DrawLevel::Mate;
            }
        }
    }
    if board.cannons().intersection(board.ours()).count() == 1
        && board.cannons().intersection(board.theirs()).count() == 1
        && board.advisors().count() == 0
    {
        return if board.bishops().count() == 0 {
            DrawLevel::Direct
        } else {
            DrawLevel::Mate
        };
    }
    DrawLevel::No
}

pub fn board_to_fen(board: &ChessBoard) -> String {
    let mut board = board.clone();
    let black_to_move = board.flipped();
    if black_to_move {
        board.mirror();
    }

    let mut result = String::new();
    let mut rank = Rank::R9;
    while rank.is_valid() {
        let mut empty = 0u32;
        let mut file = File::A;
        while file.index() <= File::I.index() {
            let square = Square::new(file, rank);
            let piece = piece_at(&board, square);
            if piece != '\0' {
                if empty > 0 {
                    result.push_str(&empty.to_string());
                    empty = 0;
                }
                result.push(piece);
            } else {
                empty += 1;
            }
            file = file.offset(1);
        }
        if empty > 0 {
            result.push_str(&empty.to_string());
        }
        if rank.index() != 0 {
            result.push('/');
        }
        if rank.index() == 0 {
            break;
        }
        rank = Rank::from_idx(rank.index() - 1).unwrap();
    }
    result.push(' ');
    result.push_str(if black_to_move { "b" } else { "w" });
    result
}

fn piece_at(board: &ChessBoard, square: Square) -> char {
    if !board.ours().contains(square) && !board.theirs().contains(square) {
        return '\0';
    }
    let mut c = if board.rooks().contains(square) {
        'R'
    } else if board.advisors().contains(square) {
        'A'
    } else if board.cannons().contains(square) {
        'C'
    } else if board.pawns().contains(square) {
        'P'
    } else if board.knights().contains(square) {
        'N'
    } else if board.bishops().contains(square) {
        'B'
    } else if board.kings().contains(square) {
        'K'
    } else {
        '\0'
    };
    if board.theirs().contains(square) {
        c = c.to_ascii_lowercase();
    }
    c
}
