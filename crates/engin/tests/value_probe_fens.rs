//! `VALUE_PROBE_CASES` 中 FEN 可解析。

use xiangqi_core::Position;

use engin::VALUE_PROBE_CASES;

#[test]
fn value_probe_fens_parse() {
    for c in VALUE_PROBE_CASES {
        Position::from_fen(c.fen).unwrap_or_else(|e| panic!("{} {}: {e}", c.id, c.fen));
    }
}
