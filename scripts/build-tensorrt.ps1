$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "This package script targets Windows TensorRT only."
}

# ---------------------------------------------------------------------------
# 本机固定路径（换机器只改这几行）
# ---------------------------------------------------------------------------
$ortNative = "C:\projects\onnxruntime-cuda13\windows\runtimes\win-x64\native"
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
$cudaBinX64 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
$trtLibs = "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorrt_libs"
# ---------------------------------------------------------------------------

# 打包注意：
# 1) 与 DirectML 互斥，不要混进同一目录。
# 2) ORT TensorRT EP 依赖 nvinfer_10.dll（TRT 10）；不要用 TRT 11。
# 3) 发行物带 ORT：onnxruntime / providers_shared / providers_tensorrt（建议同带 providers_cuda）。
# 4) 包内带完整 TensorRT DLL；CUDA/cuDNN 仍由用户环境提供。
# 5) trt_cache 只带空目录；engine 由用户首跑按本机 GPU 构建，不要跨卡复用。
# 6) 发行前确认 NVIDIA 再分发许可。

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$model = (Resolve-Path -LiteralPath (Join-Path $repoRoot "data\x7.onnx")).Path
$bundle = Join-Path $repoRoot "bundle-tensorrt"
$ortFiles = @(
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_tensorrt.dll",
    "onnxruntime_providers_cuda.dll"
)

foreach ($dir in @($ortNative, $cudaBin, $trtLibs)) {
    if (-not (Test-Path -LiteralPath $dir)) {
        throw "Missing path (edit script header): $dir"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $trtLibs "nvinfer_10.dll"))) {
    throw "Expected nvinfer_10.dll under `$trtLibs (TRT 10 ABI)."
}
$trtFiles = @(Get-ChildItem -LiteralPath $trtLibs -Filter "*.dll" -File)
if ($trtFiles.Count -eq 0) {
    throw "No TensorRT DLLs found under `$trtLibs."
}
foreach ($name in $ortFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $ortNative $name))) {
        throw "ONNX Runtime GPU file is missing: $name"
    }
}

$env:PATH = "$cudaBinX64;$cudaBin;$trtLibs;$ortNative;$env:PATH"
$env:ORT_DYLIB_PATH = Join-Path $ortNative "onnxruntime.dll"

Push-Location $repoRoot
try {
    if (Test-Path -LiteralPath $bundle) {
        Remove-Item -LiteralPath $bundle -Recurse -Force
    }

    cargo build --release -p engin --no-default-features --features tensorrt
    if ($LASTEXITCODE -ne 0) {
        throw "TensorRT engine build failed."
    }

    $releaseDir = Join-Path $repoRoot "target\release"
    $debugDir = Join-Path $repoRoot "target\debug"
    New-Item -ItemType Directory -Path $debugDir -Force | Out-Null
    foreach ($name in $ortFiles) {
        Copy-Item -LiteralPath (Join-Path $ortNative $name) -Destination (Join-Path $releaseDir $name) -Force
        Copy-Item -LiteralPath (Join-Path $ortNative $name) -Destination (Join-Path $debugDir $name) -Force
    }

    New-Item -ItemType Directory -Path $bundle -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $bundle "trt_cache") -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $releaseDir "engin.exe") -Destination (Join-Path $bundle "engin.exe") -Force
    Copy-Item -LiteralPath $model -Destination (Join-Path $bundle "x7.onnx") -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot "TRT-README.txt") -Destination (Join-Path $bundle "TRT-README.txt") -Force
    foreach ($name in $ortFiles) {
        Copy-Item -LiteralPath (Join-Path $releaseDir $name) -Destination $bundle -Force
    }
    foreach ($file in $trtFiles) {
        Copy-Item -LiteralPath $file.FullName -Destination $bundle -Force
    }

    $manifest = [ordered]@{
        engine = "engin.exe"
        model = "x7.onnx"
        readme = "TRT-README.txt"
        execution_provider = "TensorrtExecutionProvider"
        onnx_runtime = "Microsoft.ML.OnnxRuntime.Gpu.Windows (CUDA13 package with TensorRT EP)"
        tensorrt_abi = "nvinfer_10.dll (TensorRT 10.x)"
        runtime_requirement = "CUDA 13 + cuDNN 9 on PATH; TensorRT 10 DLLs are bundled"
        notes = @(
            "Do not mix with DirectML bundle",
            "trt_cache ships empty; end users build engines on first run for their GPU",
            "ORT TensorRT provider ABI is nvinfer_10 (TensorRT 10.x)",
            "Bundle includes all DLLs from the configured tensorrt_libs directory"
        )
        files = @(Get-ChildItem -LiteralPath $bundle -File | ForEach-Object {
                [ordered]@{
                    name = $_.Name
                    bytes = $_.Length
                    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
                }
            })
    }
    $manifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $bundle "manifest.json") -Encoding utf8
    Write-Host "TensorRT bundle created: $bundle"
}
finally {
    Pop-Location
}
