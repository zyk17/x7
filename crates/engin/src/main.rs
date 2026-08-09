//! stream 搜索的 UCI stdin/stdout 入口。
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use engin::{Engine, StdoutUciResponder, UciLoop};

/// 在 UCI 启动前定位随程序发布的正式 ONNX 权重。
///
/// px0 在 `src/neural/shared_params.cc:43-50` 提供同名 `WeightsFile` option，
/// 并在 `src/engine.cc:153-167` 从 option 创建 backend。本实现没有权重自动发现或
/// backend 注册表；发行布局等价为 `engin.exe` 与同目录 `x7.onnx`。当前目录与编译期
/// 仓库路径仅用于开发期回退，发布布局不依赖它们。
fn default_weights_file() -> PathBuf {
    let mut candidates = Vec::new();
    if let Ok(executable) = std::env::current_exe()
        && let Some(directory) = executable.parent()
    {
        candidates.push(directory.join("x7.onnx"));
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
    let mut responder = StdoutUciResponder::default();
    let mut engine = Engine::new();
    // 首个 `position` 命令初始化 ONNX backend；Engine 本身已经在上一行创建。
    engine
        .set_option("WeightsFile", &default_weights_file().to_string_lossy())
        .expect("default UCI options must be valid");
    let mut uci = UciLoop::new(&mut responder, &mut engine);

    // GUI 等待有限 `go` 的结果且不再输入时，watchdog 的回调仍必须输出到 stdout。
    let (input_tx, input_rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            if input_tx.send(line).is_err() {
                break;
            }
        }
    });
    loop {
        match input_rx.recv_timeout(Duration::from_millis(25)) {
            Ok(Ok(line)) => match uci.process_line(&line, env!("CARGO_PKG_VERSION")) {
                Ok(true) => {}
                Ok(false) => break,
                Err(error) => eprintln!("UCI error: {error}"),
            },
            Ok(Err(_)) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => uci.flush_output(),
        }
    }
}
