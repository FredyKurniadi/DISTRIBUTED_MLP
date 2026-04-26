$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

Push-Location $projectRoot
try {
    & $pythonExe -m src.mlp.train --distributed-config "configs/distributed.yaml" --model-config "configs/model.yaml"
}
finally {
    Pop-Location
}
