param(
    [ValidateSet('Start','Status','Final')] [string]$Action = 'Status',
    [int]$Seed = 0
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($Action -eq 'Status') {
    foreach ($label in 'supervisor','final') {
        if (Test-Path "logs/$label.pid") {
            $number = (Get-Content "logs/$label.pid" -Raw).Trim()
            $alive = [bool](Get-Process -Id $number -ErrorAction SilentlyContinue)
            Write-Output "$label PID $number alive=$alive"
            Get-Content "logs/$label.log" -Tail 3
            Get-Content "logs/$label.err" -Tail 3
        }
    }
    $plan = Get-Content results/experiment_plan.json -Raw | ConvertFrom-Json
    foreach ($tree in 'results/quick/runs','results/runs') {
        foreach ($suite in 'arch_ablation','shape_ablation','scaling','tokenizer') {
            $count = @(Get-ChildItem "$tree/$suite/*.json" -ErrorAction SilentlyContinue).Count
            Write-Output "$tree/$suite $count/$($plan.runs.$suite)"
        }
    }
    return
}
$label = if ($Action -eq 'Start') { 'supervisor' } else { 'final' }
$entry = if ($Action -eq 'Start') { 'scripts.phase3_jobs' } else { 'scripts.train_final' }
$extra = if ($Action -eq 'Final') { ' --token-budget 3145728' } else { '' }
$lock = [IO.File]::Open((Join-Path $PWD "logs/$label.launch.lock"), 'OpenOrCreate','Write','None')
try {
    if (Test-Path "logs/$label.pid") {
        $number = (Get-Content "logs/$label.pid" -Raw).Trim()
        if (Get-Process -Id $number -ErrorAction SilentlyContinue) { throw "$label PID $number is alive" }
    }
    $stamp = Get-Date -Format yyyyMMdd-HHmmss
    foreach ($suffix in 'log','err') {
        if (Test-Path "logs/$label.$suffix") {
            Copy-Item "logs/$label.$suffix" "logs/$label.$stamp.$suffix"
        }
    }
    $proc = Start-Process .venv\Scripts\python.exe -ArgumentList "-u -m $entry --seed $Seed$extra" -PassThru -WindowStyle Hidden -RedirectStandardOutput "logs/$label.log" -RedirectStandardError "logs/$label.err"
    $proc.Id | Set-Content "logs/$label.pid"
    Write-Output "Launched $label PID $($proc.Id). Monitor with .\scripts\phase3_jobs.ps1 -Action Status"
} finally {
    $lock.Dispose()
}
