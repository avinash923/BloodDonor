param(
    [switch]$Rebuild,
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [ValidateRange(1, 3600)][int]$ResponseWindowSeconds = 60,
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
    if ($Rebuild -or -not (Test-Path -LiteralPath 'mobile\build\web\index.html')) {
        Push-Location (Join-Path $PSScriptRoot 'mobile')
        try {
            & flutter build web --no-web-resources-cdn
            if ($LASTEXITCODE -ne 0) { throw 'Flutter web build failed.' }
        } finally { Pop-Location }
    }
    Write-Host "Open http://127.0.0.1:$Port (Ctrl+C to stop)"
    & $projectPython -m src.api.server --port $Port --response-window-seconds $ResponseWindowSeconds --demo-donors $DemoDonors --state-file $StateFile
    if ($LASTEXITCODE -ne 0) { throw 'Demo server exited with an error.' }
} finally { Pop-Location }
