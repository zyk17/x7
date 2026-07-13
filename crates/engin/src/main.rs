//! px0 UCI stdin/stdout entry with P4 classic search.

use std::io::{self, BufRead};

use engin::{ClassicEngine, StdoutUciResponder, UciLoop, UciOptions};

fn main() {
    let stdin = io::stdin();
    let mut responder = StdoutUciResponder::default();
    let mut options = UciOptions::populate_defaults();
    // UniformBackend is reserved for P3/P4 tests. Until the px0 WeightsFile
    // UCI configuration is translated, production UCI must report unavailable
    // instead of returning a heuristic move.
    let mut engine = ClassicEngine::unavailable();
    let mut uci = UciLoop::new(&mut responder, &mut options, &mut engine);

    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        match uci.process_line(&line, env!("CARGO_PKG_VERSION")) {
            Ok(true) => {}
            Ok(false) => break,
            Err(error) => eprintln!("UCI error: {error}"),
        }
    }
}
