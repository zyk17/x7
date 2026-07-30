//! 本地真实 ONNX 的 stream UCI 生命周期回归。
//!
//! 该测试只在仓库 `data/x7.onnx` 存在时运行，避免 CI 下载模型；覆盖的语义是
//! ARCHITECTURE.md 要求的 `position -> go -> stop -> position -> go`。搜索实现参考
//! LC3 Overview 的 generation/owned-event 生命周期。

use std::path::PathBuf;
use std::sync::Once;
use std::time::Duration;

use engin::{Engine, UciLoop, VecUciResponder};
use xiangqi_core::initialize_magic_bitboards;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(initialize_magic_bitboards);
}

fn local_onnx() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root")
        .join("data/x7.onnx")
}

#[test]
fn real_onnx_stop_then_new_position_has_two_bestmoves() {
    ensure_init();
    let onnx = local_onnx();
    if !onnx.is_file() {
        eprintln!("skip real ONNX regression: {} is absent", onnx.display());
        return;
    }

    let mut engine = Engine::from_onnx_file(&onnx).expect("load local ONNX");
    let mut responder = VecUciResponder::default();
    let mut uci = UciLoop::new(&mut responder, &mut engine);

    // 先完成一次真实 NN 调用，避免随后 stop 恰好落在 DirectML 初始化之前。
    uci.process_line("position startpos", "test").expect("position");
    uci.process_line("go nodes 8", "test").expect("warmup go");
    uci.process_line("wait", "test").expect("warmup wait");

    uci.process_line("go infinite", "test").expect("infinite go");
    std::thread::sleep(Duration::from_millis(50));
    uci.process_line("stop", "test").expect("stop");
    uci.process_line("position startpos moves c3c4", "test")
        .expect("replace position");
    uci.process_line("go nodes 16", "test").expect("second go");
    uci.process_line("wait", "test").expect("second wait");
    drop(uci);

    let bestmoves: Vec<_> = responder
        .responses
        .iter()
        .filter(|line| line.starts_with("bestmove "))
        .collect();
    assert_eq!(bestmoves.len(), 3, "warmup plus one bestmove for stop and the new root");
    assert!(
        bestmoves[2] != "bestmove a0a0",
        "the second real-ONNX search must reach a legal root move"
    );
}
