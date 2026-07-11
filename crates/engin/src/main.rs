//! px0 UCI stdin/stdout entry with P3 classic search.

use std::io::{self, BufRead};

use engin::{ClassicEngine, StdoutUciResponder, UciLoop, UciOptions};

fn main() {
    let stdin = io::stdin();
    let mut responder = StdoutUciResponder::default();
    let mut options = UciOptions::populate_defaults();
    let mut engine = ClassicEngine::uniform();
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
