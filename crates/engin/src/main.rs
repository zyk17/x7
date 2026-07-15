//! px0 UCI stdin/stdout entry with P4 classic search.

use std::io::{self, BufRead};
use std::path::PathBuf;

use engin::{ClassicEngine, EngineController, StdoutUciResponder, UciLoop, UciOptions};

/// Resolves the bundled formal ONNX weight before UCI starts.
///
/// px0 exposes the same `WeightsFile` option in
/// `src/neural/shared_params.cc:43-50` and creates its backend from options in
/// `src/engine.cc:153-167`.  This port has no px0 weight autodiscovery or
/// backend registry, so its distributable equivalent is an ONNX file in the
/// app resource layout: `engin.exe` plus a sibling `x7.onnx`. The current
/// working directory and compile-time repository location are development
/// fallbacks; neither is required by the packaged layout.
fn default_weights_file() -> PathBuf {
    let mut candidates = Vec::new();
    if let Ok(executable) = std::env::current_exe() {
        if let Some(directory) = executable.parent() {
            candidates.push(directory.join("x7.onnx"));
        }
    }
    candidates.push(PathBuf::from("x7.onnx"));
    let mut dev_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    assert!(dev_root.pop(), "engin crate has a parent directory");
    assert!(dev_root.pop(), "engin crate is nested below the workspace root");
    let dev_path = dev_root.join("data").join("x7.onnx");
    candidates.push(dev_path.clone());
    candidates.into_iter().find(|path| path.is_file()).unwrap_or(dev_path)
}

fn main() {
    let stdin = io::stdin();
    let mut responder = StdoutUciResponder::default();
    let mut options = UciOptions::populate_defaults();
    options.weights_file = default_weights_file().to_string_lossy().into_owned();
    let mut engine = ClassicEngine::new();
    // Seed the formal `WeightsFile` before `UciLoop` registers the responder.
    // The backend is created by the first SetPosition, matching px0's
    // configuration update order (`src/engine.cc:187-197`).
    engine
        .set_uci_options(&options)
        .expect("default UCI options must be valid");
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
