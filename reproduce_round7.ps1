param(
    [ValidateSet('audit', 'replay', 'full')]
    [string]$Mode = 'audit',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -X utf8 scripts/reproduce_round7.py --mode $Mode
    if ($LASTEXITCODE -ne 0) { throw "Round 7 check failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
