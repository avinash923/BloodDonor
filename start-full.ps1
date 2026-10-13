param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [ValidateRange(1, 3600)][int]$ResponseWindowSeconds = 60,
    [ValidateRange(0, 1000)][int]$DemoDonors = 0,
    [string]$StateFile = 'data/runtime/full_system_state.json',
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
    Write-Host "  Deployment Mode:    Full Live (Real Donors & Live UDI Engine)" -ForegroundColor Cyan
    Write-Host "  Backend API:        http://127.0.0.1:$Port/api/v1/health" -ForegroundColor Cyan
    Write-Host "  Web Dashboard:      http://127.0.0.1:$Port/" -ForegroundColor Cyan
    Write-Host "  Synthetic Donors:   $DemoDonors (Full mode: real accounts & registration)" -ForegroundColor Cyan
    Write-Host "  State Storage:      $StateFile" -ForegroundColor Cyan
    Write-Host "  Wave Response Win:  ${ResponseWindowSeconds}s per wave" -ForegroundColor Cyan
    Write-Host "  Python Binary:      $projectPython" -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Red
    Write-Host "Open in browser: http://127.0.0.1:$Port (Press Ctrl+C to stop)" -ForegroundColor Yellow
    Write-Host ""

    if (-not $NoBrowser) {
        Start-Process "http://127.0.0.1:$Port"
    }

    & $projectPython -m src.api.server --port $Port --response-window-seconds $ResponseWindowSeconds --demo-donors $DemoDonors --state-file $StateFile --web-root $WebRoot
    if ($LASTEXITCODE -ne 0) { throw 'Server exited with an error.' }
} finally {
    Pop-Location
}
