param(
    [Parameter(Mandatory = $true)]
    [int]$WorkerId,
    [Parameter(Mandatory = $true)]
    [int]$Port,
    [string]$Host = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

Push-Location $projectRoot
try {
    & $pythonExe -m src.distributed.worker --host $Host --port $Port --worker-id $WorkerId --state-dir "runtime/worker_state"
}
finally {
    Pop-Location
}
