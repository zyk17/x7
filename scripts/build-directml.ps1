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
    Copy-ResolvedFile (Join-Path $releaseDir "x7.exe") (Join-Path $bundle "x7.exe")
    Copy-Item -LiteralPath $model -Destination (Join-Path $bundle "x7.onnx") -Force

    # `ort` links ONNX Runtime statically; DirectML is the only separate runtime DLL.
    Copy-ResolvedFile (Join-Path $releaseDir "DirectML.dll") (Join-Path $bundle "DirectML.dll")

    Write-Host "DirectML bundle created: $bundle"
}
finally {
    Pop-Location
}
