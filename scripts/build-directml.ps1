[CmdletBinding()]
param(
    [string]$ModelPath,
    [string]$BundleDir
)

$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "This package script targets Windows DirectML only."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ModelPath = if ($ModelPath) { $ModelPath } else { Join-Path $repoRoot "data\x7.onnx" }
$BundleDir = if ($BundleDir) { $BundleDir } else { Join-Path $repoRoot "bundle-directml" }
$model = (Resolve-Path -LiteralPath $ModelPath).Path
$bundle = [System.IO.Path]::GetFullPath($BundleDir)

function Copy-ResolvedFile {
    param(
        [Parameter(Mandatory)] [string]$Source,
        [Parameter(Mandatory)] [string]$Destination
    )

    $item = Get-Item -LiteralPath $Source
    $resolvedSource = if ($item.LinkType -and $item.Target) {
        (Resolve-Path -LiteralPath $item.Target).Path
    }
    else {
        $item.FullName
    }
    Copy-Item -LiteralPath $resolvedSource -Destination $Destination -Force
}

Push-Location $repoRoot
try {
    if (Test-Path -LiteralPath $bundle) {
        Remove-Item -LiteralPath $bundle -Recurse -Force
    }

    cargo clean
    cargo build --release -p engin --no-default-features --features directml
    if ($LASTEXITCODE -ne 0) {
        throw "DirectML engine build failed."
    }

    $releaseDir = Join-Path $repoRoot "target\release"
    New-Item -ItemType Directory -Path $bundle -Force | Out-Null
    Copy-ResolvedFile (Join-Path $releaseDir "engin.exe") (Join-Path $bundle "engin.exe")
    Copy-Item -LiteralPath $model -Destination (Join-Path $bundle "x7.onnx") -Force

    # `ort` links ONNX Runtime statically; DirectML is the only separate runtime DLL.
    Copy-ResolvedFile (Join-Path $releaseDir "DirectML.dll") (Join-Path $bundle "DirectML.dll")

    $smokeLog = Join-Path $env:TEMP "x7-directml-package-smoke-$PID.log"
    try {
        $smokeOutput = @(
            & {
                "uci"
                "isready"
                "position startpos"
                "go nodes 1"
                Start-Sleep -Seconds 5
                "quit"
            } | & (Join-Path $bundle "engin.exe") 2>&1 | Tee-Object -FilePath $smokeLog
        )
    }
    finally {
        Remove-Item -LiteralPath $smokeLog -Force -ErrorAction SilentlyContinue
    }

    $smokeText = $smokeOutput | Out-String
    if ($smokeText -match "ONNX DirectML unavailable") {
        throw "Bundle smoke fell back to CPU instead of DirectML."
    }
    if ($smokeText -notmatch "(?m)^bestmove ") {
        throw "Bundle smoke did not return bestmove."
    }

    $manifest = [ordered]@{
        engine = "engin.exe"
        model = "x7.onnx"
        onnx_runtime = "statically linked by ort"
        execution_provider = "DirectML"
        files = @(Get-ChildItem -LiteralPath $bundle -File | ForEach-Object {
                [ordered]@{
                    name = $_.Name
                    bytes = $_.Length
                    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
                }
            })
    }
    $manifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $bundle "manifest.json") -Encoding utf8
    Write-Host "DirectML bundle created: $bundle"
}
finally {
    Pop-Location
}
