param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [ValidateRange(0, 1000)][int]$DemoDonors = 30,
    [string]$StateFile = 'data/runtime/demo_state.json'
)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) {
        throw 'Create .venv and install requirements.txt first. See README.md.'
    }
    Write-Host "Open http://127.0.0.1:$Port (Ctrl+C to stop)"
    & $projectPython -m src.api.server --port $Port --demo-donors $DemoDonors --state-file $StateFile --web-root web
    if ($LASTEXITCODE -ne 0) { throw 'Demo server exited with an error.' }
} finally { Pop-Location }
