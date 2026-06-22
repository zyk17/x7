# nn 训练专用虚拟环境（torch cu128）
# 用法: cd nn; . .\activate.ps1

$VenvRoot = Join-Path $PSScriptRoot ".venv"
$Python = Join-Path $VenvRoot "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "未找到 $Python`n请先在 nn 目录执行: python -m venv .venv && pip install -e .[train]"
    return
}

$env:VIRTUAL_ENV = $VenvRoot
$env:PATH = "$(Join-Path $VenvRoot 'Scripts');$env:PATH"

& $Python -c "import torch; print('nn/.venv | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
