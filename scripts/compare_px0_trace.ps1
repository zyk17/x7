<#
  px0 UCI fixed-node trace helper.

  Reference: px0/src/chess/uciloop.cc:178-254 and
  px0/src/search/classic/wrapper.cc:53-141. It deliberately records raw
  transcripts instead of comparing scores: local px0 uses a protobuf weight
  while engin uses ONNX, so numerical equality is not a valid assertion.
#>
[CmdletBinding()]
param(
    [string]$Px0Exe = "C:\Users\Administrator\projects\px0\build_onnx_cpu\x7_bundle\lc0.exe",
    [string]$EnginExe = "C:\projects\77xiangqi_engine\target\release\engin.exe",
    [string]$EnginWeights = "C:\projects\77xiangqi_engine\data\x7.onnx",
    [string]$Fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
    [ValidateRange(1, 1000000000)]
    [int]$Nodes = 10000,
    [string]$OutputDir = "C:\projects\77xiangqi_engine\logs\trace"
)

$ErrorActionPreference = "Stop"

function Invoke-UciTrace {
    param(
        [Parameter(Mandatory)] [string]$Executable,
        [Parameter(Mandatory)] [string[]]$Commands,
        [Parameter(Mandatory)] [string]$OutputPath
    )

    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "UCI executable not found: $Executable"
    }

    $OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
    $previousErrorAction = $ErrorActionPreference
    Push-Location (Split-Path -Parent $Executable)
    try {
        # Native stderr becomes a PowerShell error record under `2>&1`. px0
        # writes its startup banner there, so use the process exit code rather
        # than ErrorActionPreference as the failure signal.
        $ErrorActionPreference = "Continue"
        $Commands | & $Executable 2>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -LiteralPath $OutputPath
        if ($LASTEXITCODE -ne 0) {
            throw "UCI executable failed with exit code ${LASTEXITCODE}: $Executable"
        }
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $EnginWeights -PathType Leaf)) {
    throw "ONNX weight not found: $EnginWeights"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$px0Output = Join-Path $OutputDir "px0_nodes_$Nodes.log"
$enginOutput = Join-Path $OutputDir "engin_nodes_$Nodes.log"
Invoke-UciTrace -Executable $Px0Exe -Commands @(
    "uci", "isready", "position fen $Fen", "go nodes $Nodes", "quit"
) -OutputPath $px0Output
Invoke-UciTrace -Executable $EnginExe -Commands @(
    "uci", "setoption name WeightsFile value $EnginWeights", "isready",
    "position fen $Fen", "go nodes $Nodes", "quit"
) -OutputPath $enginOutput

Write-Output "px0 transcript: $px0Output"
Write-Output "engin transcript: $enginOutput"
