$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python venv not found at $pythonExe. Run .\\scripts\\setup_all.ps1 first."
}

$workerJobs = @()
$workers = @(
    @{ WorkerId = 1; Port = 6101 },
    @{ WorkerId = 2; Port = 6102 },
    @{ WorkerId = 3; Port = 6103 }
)

foreach ($worker in $workers) {
    $job = Start-Job -Name ("worker_" + $worker.WorkerId) -ScriptBlock {
        param($root, $py, $wid, $port)
        Set-Location $root
        & $py -m src.distributed.worker --host "127.0.0.1" --port $port --worker-id $wid --state-dir "runtime/worker_state"
    } -ArgumentList $projectRoot, $pythonExe, $worker.WorkerId, $worker.Port

    $workerJobs += $job
}

Push-Location $projectRoot
try {
    & $pythonExe -m src.mlp.train --distributed-config "configs/distributed.yaml" --model-config "configs/model.yaml"
}
finally {
    foreach ($job in $workerJobs) {
        if ($null -ne $job) {
            Stop-Job -Id $job.Id -ErrorAction SilentlyContinue
            Remove-Job -Id $job.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Pop-Location
}
