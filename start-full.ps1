param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$StateFile = 'data/runtime/live_state.json',
    [string]$WebRoot = 'web',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    # Detect Python environment
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) {
        $candidate = 'C:\Users\sanja\AppData\Local\Programs\Python\Python310\python.exe'
        if (Test-Path -LiteralPath $candidate) {
            $projectPython = $candidate
        } else {
            $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
            if ($cmd) {
                $projectPython = $cmd.Source
            } else {
                throw 'Python 3.10+ not found. Please install Python or set up .venv.'
            }
        }
    }

    # Ensure runtime state directory exists
    $runtimeDir = Split-Path $StateFile -Parent
    if ($runtimeDir -and -not (Test-Path -LiteralPath $runtimeDir)) {
        New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    }

    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Red
    Write-Host "  EMERGENCY BLOOD DONOR SYSTEM - FULL LIVE ENGINE & WEB STACK    " -ForegroundColor White
    Write-Host "=================================================================" -ForegroundColor Red
    if (-not $env:ROUTING_PROVIDER) { $env:ROUTING_PROVIDER = 'osrm' }
    Write-Host "  Deployment Mode:    Live (registered donors only, no demo data)" -ForegroundColor Cyan
    Write-Host "  Road Routing:       $($env:ROUTING_PROVIDER) (free OpenStreetMap OSRM by default)" -ForegroundColor Cyan
    Write-Host "  Backend API:        http://127.0.0.1:$Port/api/v1/health" -ForegroundColor Cyan
    Write-Host "  Web Dashboard:      http://127.0.0.1:$Port/" -ForegroundColor Cyan
    Write-Host "  State Storage:      $StateFile" -ForegroundColor Cyan
    Write-Host "  Python Binary:      $projectPython" -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Red
    Write-Host "Open in browser: http://127.0.0.1:$Port (Press Ctrl+C to stop)" -ForegroundColor Yellow
    Write-Host ""

    if (-not $NoBrowser) {
        Start-Process "http://127.0.0.1:$Port"
    }

    & $projectPython -m src.api.server --live --port $Port --state-file $StateFile --web-root $WebRoot
    if ($LASTEXITCODE -ne 0) { throw 'Server exited with an error.' }
} finally {
    Pop-Location
}
