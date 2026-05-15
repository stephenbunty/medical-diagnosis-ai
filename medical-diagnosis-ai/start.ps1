# One-shot local start (no Docker). Run: .\start.ps1
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$Venv = Join-Path $Backend ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Pip = Join-Path $Venv "Scripts\pip.exe"

function Require-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$name' not found. Install it and add to PATH." -ForegroundColor Red
        if ($name -eq "python") { Write-Host "  https://www.python.org/downloads/ (check Add to PATH)" }
        if ($name -eq "node") { Write-Host "  https://nodejs.org/" }
        exit 1
    }
}

Require-Command python
Require-Command node
Require-Command npm

Write-Host ""
Write-Host "=== Medical Diagnosis AI - starting ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $Python)) {
    Write-Host "Creating Python venv..."
    python -m venv $Venv
}

Write-Host "Installing backend dependencies (first run may take several minutes)..."
& $Pip install -q -r (Join-Path $Backend "requirements.txt")

$EnvFile = Join-Path $Backend ".env"
$EnvExample = Join-Path $Backend ".env.example"
if (-not (Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
    Copy-Item $EnvExample $EnvFile
    Write-Host "Created backend\.env from .env.example"
}

if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Push-Location $Frontend
    npm install
    Pop-Location
}

$BackendCmd = "Set-Location -LiteralPath '$Backend'; & '$Python' -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
$FrontendCmd = "Set-Location -LiteralPath '$Frontend'; npm run dev"

Write-Host "Starting API (port 8000) and UI (port 5173) in new windows..." -ForegroundColor Green
Write-Host ""
Start-Process powershell -ArgumentList @("-NoExit", "-Command", $BackendCmd)
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList @("-NoExit", "-Command", $FrontendCmd)

Write-Host "Open:  http://127.0.0.1:5173" -ForegroundColor Yellow
Write-Host "API:   http://127.0.0.1:8000/docs"
Write-Host ""
Write-Host "Close the two PowerShell windows to stop the app."
Write-Host ""
