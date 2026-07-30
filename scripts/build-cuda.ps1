$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "This package script targets Windows CUDA only."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$model = (Resolve-Path -LiteralPath (Join-Path $repoRoot "data\x7.onnx")).Path
$bundle = Join-Path $repoRoot "bundle-cuda"
$cudaRuntimeDir = "C:\projects\onnxruntime-cuda13\windows\runtimes\win-x64\native"
# 要更换 CUDA ONNX Runtime，只改上一行。
foreach ($name in "onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll") {
    if (-not (Test-Path -LiteralPath (Join-Path $cudaRuntimeDir $name))) {
        throw "ONNX Runtime CUDA file is missing: $name"
    }
}

Push-Location $repoRoot
try {
    if (Test-Path -LiteralPath $bundle) {
        Remove-Item -LiteralPath $bundle -Recurse -Force
    }

    cargo build --release -p engin --no-default-features --features cuda
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA engine build failed."
    }

    $releaseDir = Join-Path $repoRoot "target\release"
    $debugDir = Join-Path $repoRoot "target\debug"
    # 动态加载的 ONNX Runtime 必须与本机引擎相邻；release 和普通 `cargo run`
    # 都不需要再设置 ORT_DYLIB_PATH。
    New-Item -ItemType Directory -Path $debugDir -Force | Out-Null
    foreach ($name in "onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll") {
        Copy-Item -LiteralPath (Join-Path $cudaRuntimeDir $name) -Destination (Join-Path $releaseDir $name) -Force
        Copy-Item -LiteralPath (Join-Path $cudaRuntimeDir $name) -Destination (Join-Path $debugDir $name) -Force
    }

    New-Item -ItemType Directory -Path $bundle -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $releaseDir "engin.exe") -Destination (Join-Path $bundle "engin.exe") -Force
    Copy-Item -LiteralPath $model -Destination (Join-Path $bundle "x7.onnx") -Force
    foreach ($name in "onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll") {
        Copy-Item -LiteralPath (Join-Path $releaseDir $name) -Destination $bundle -Force
    }

    $cudaRoot = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Where-Object Name -like "v13.*" | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
    if ($cudaRoot) {
        $env:PATH = "$(Join-Path $cudaRoot 'bin\x64');$(Join-Path $cudaRoot 'bin');$env:PATH"
    }
    $env:ORT_DYLIB_PATH = Join-Path $bundle "onnxruntime.dll"

    $smokeOutput = @(
        & {
            "uci"
            "isready"
            "position startpos"
            "go nodes 1"
            Start-Sleep -Seconds 5
            "quit"
        } | & (Join-Path $bundle "engin.exe") 2>&1
    ) | Out-String
    if ($smokeOutput -match "ONNX CUDA unavailable") {
        throw "Bundle smoke fell back to CPU instead of CUDA."
    }
    if ($smokeOutput -notmatch "(?m)^bestmove ") {
        throw "Bundle smoke did not return bestmove."
    }

    $manifest = [ordered]@{
        engine = "engin.exe"
        model = "x7.onnx"
        execution_provider = "CUDAExecutionProvider"
        onnx_runtime = "official Microsoft.ML.OnnxRuntime.Gpu.Windows 1.28.0"
        runtime_requirement = "CUDA 13 and cuDNN 9 available on PATH"
        files = @(Get-ChildItem -LiteralPath $bundle -File | ForEach-Object {
                [ordered]@{
                    name = $_.Name
                    bytes = $_.Length
                    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
                }
            })
    }
    $manifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $bundle "manifest.json") -Encoding utf8
    Write-Host "CUDA bundle created: $bundle"
}
finally {
    Pop-Location
}
