param(
    [string]$Onnx = "",
    [int]$Playouts = 64,
    [int]$Threads = 1,
    [string]$Px0Exe = "",
    [string]$Out = "data\search_regression_latest.ndjson",
    [string]$DiffOut = "data\search_regression_diff.txt",
    [switch]$RequireOnnx
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $Onnx) {
    foreach ($c in @("data\x7.onnx", "data\policy.onnx", "data\checkpoints\baseline_px0_katago_v1.best.onnx")) {
        if (Test-Path $c) { $Onnx = $c; break }
    }
}
if (-not $Px0Exe) {
    foreach ($c in @(
        "C:\Users\Administrator\projects\px0\build\Release\px0.exe",
        "C:\Users\Administrator\projects\px0\build\px0.exe"
    )) {
        if (Test-Path $c) { $Px0Exe = $c; break }
    }
}

$engin = Join-Path $Root "target\release\engin.exe"
if (-not (Test-Path $engin)) { cargo build --release -p engin }

$fenFile = Join-Path $Root "data\search_regression_fens.txt"
$lines = Get-Content $fenFile | Where-Object { $_ -and -not $_.StartsWith("#") }

function Invoke-EnginBench([string]$fen) {
    $args = @("--bench", "--playouts", "$Playouts", "--threads", "$Threads", "--fen", $fen)
    if ($Onnx) { $args += @("--onnx", $Onnx) }
    if ($RequireOnnx) { $args += "--require-onnx" }
    $chunk = & $engin @args 2>&1
    if ($LASTEXITCODE -ne 0) { throw "engin bench failed: $fen" }
    return ($chunk | Where-Object { $_.Trim() } | Select-Object -Last 1)
}

function Invoke-Px0Bench([string]$fen) {
    if (-not $Px0Exe -or -not (Test-Path $Px0Exe)) { return $null }
    $fenArg = if ($fen -match " moves ") { "fen $fen" } else { "fen $fen" }
    $uci = @(
        "uci",
        "isready",
        "position $fenArg",
        "go nodes $Playouts",
        "quit"
    ) -join "`n"
    $out = $uci | & $Px0Exe 2>&1
    $info = $out | Where-Object { $_ -match "^info depth" } | Select-Object -Last 1
    $best = $out | Where-Object { $_ -match "^bestmove " } | Select-Object -Last 1
    if (-not $best) { return $null }
    return [pscustomobject]@{
        info = "$info"
        bestmove = "$best"
    }
}

$all = @()
$diffs = New-Object System.Collections.Generic.List[string]

foreach ($line in $lines) {
    $fen = $line.Trim()
    $enginJson = Invoke-EnginBench $fen | ConvertFrom-Json
    $row = [ordered]@{
        fen = $fen
        engin = $enginJson
    }
    $px0 = Invoke-Px0Bench $fen
    if ($px0) {
        $row.px0 = $px0
        $eb = $enginJson.bestmove
        $pb = if ($px0.bestmove -match "bestmove (\S+)") { $Matches[1] } else { "" }
        $diffs.Add("FEN: $fen")
        $diffs.Add("  engin bestmove=$eb playouts=$($enginJson.playouts) depth=$($enginJson.depth) seldepth=$($enginJson.seldepth)")
        $diffs.Add("  px0   bestmove=$pb info=$($px0.info)")
        if ($enginJson.pv) {
            $diffs.Add("  engin pv=$($enginJson.pv -join ' ')")
        }
        $diffs.Add("")
    }
    $all += ($row | ConvertTo-Json -Compress -Depth 8)
}

$outPath = Join-Path $Root $Out
$all | Set-Content -Encoding utf8 $outPath
$diffPath = Join-Path $Root $DiffOut
$diffs | Set-Content -Encoding utf8 $diffPath
Write-Host "wrote $($all.Count) rows -> $outPath"
Write-Host "diff -> $diffPath"
