//! TensorRT feature：用 nvcc 编译 `cuda/onnx_kernels.cu`。
//! 换机器只改下面两个默认路径，或设 `CUDA_PATH` / `X7_MSVC_BIN`。

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/onnx_kernels.cu");
    println!("cargo:rerun-if-changed=cuda/onnx_kernels.h");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=X7_CUDA_PATH");
    println!("cargo:rerun-if-env-changed=X7_MSVC_BIN");

    if env::var("CARGO_FEATURE_TENSORRT").is_err() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda = env::var("X7_CUDA_PATH")
        .or_else(|_| env::var("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"));
    let nvcc = cuda.join("bin/nvcc.exe");
    assert!(
        nvcc.is_file(),
        "nvcc missing: {} (set CUDA_PATH / X7_CUDA_PATH)",
        nvcc.display()
    );

    let obj = out_dir.join("onnx_kernels.obj");
    let mut cmd = Command::new(&nvcc);
    cmd.args([
        "-c",
        manifest_dir.join("cuda/onnx_kernels.cu").to_str().unwrap(),
        "-o",
        obj.to_str().unwrap(),
        "-O3",
        "--std=c++17",
        "-I",
        manifest_dir.join("cuda").to_str().unwrap(),
        "-Xcompiler",
        "/MD",
    ]);
    // nvcc 需要 cl.exe；本机默认 VS 18 MSVC，可用 X7_MSVC_BIN 覆盖。
    let msvc_bin = env::var("X7_MSVC_BIN").unwrap_or_else(|_| {
        r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64".into()
    });
    let mut path = env::var_os("PATH").unwrap_or_default();
    let mut prefixed = std::ffi::OsString::from(&msvc_bin);
    prefixed.push(";");
    prefixed.push(&path);
    std::mem::swap(&mut path, &mut prefixed);
    cmd.env("PATH", path);

    assert!(cmd.status().expect("spawn nvcc").success(), "nvcc failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda.join("lib/x64").display());
    println!("cargo:rustc-link-arg={}", obj.display());
    println!("cargo:rustc-link-lib=cudart");
}
