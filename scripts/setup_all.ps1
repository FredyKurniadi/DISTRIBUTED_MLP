$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPath = Join-Path $projectRoot ".venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment in $venvPath"
    python -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$requirements = Join-Path $projectRoot "requirements.txt"

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r $requirements

Write-Host "Environment ready at $venvPath"
