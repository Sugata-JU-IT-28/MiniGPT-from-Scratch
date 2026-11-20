param(
    [Parameter(Mandatory, Position=0)]
    [ValidateSet("install","lint","test","data","train","experiments","benchmark","serve","smoke","docker-build","docker-run")]
    [string]$Task,
    [int]$Seed = 42,
    [string]$Checkpoint = "checkpoints/final.pt",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python = ".\.venv\Scripts\python.exe"
    function Invoke-Python {
        & $python @args
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    switch ($Task) {
        install { Invoke-Python -m pip install -e ".[dev]" }
        lint { Invoke-Python -m ruff check . }
        test { Invoke-Python -m pytest }
        data {
            foreach ($vocab in 512,1024,2048) {
                Invoke-Python -m scripts.prepare_data --vocab-size $vocab --seed $Seed
            }
        }
        train {
            foreach ($preset in "baseline","modern") {
                Invoke-Python -m scripts.train --config configs/cpu_tiny.yaml --preset $preset --seed $Seed
            }
            Invoke-Python -m scripts.phase2_summary --seed $Seed
        }
        experiments {
            if (-not $PSBoundParameters.ContainsKey("Seed")) { $Seed = 0 }
            Invoke-Python -m scripts.run_experiments --seed $Seed
            Invoke-Python -m scripts.aggregate_results --seed $Seed
        }
        benchmark { Invoke-Python -m scripts.benchmark_kv_cache --seed $Seed }
        serve { Invoke-Python -m minigpt.serve --checkpoint $Checkpoint --port $Port --seed $Seed }
        smoke { Invoke-Python scripts/smoke.py --checkpoint $Checkpoint --seed $Seed }
        docker-build { & docker build -t minigpt .; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
        docker-run {
            $root = (Get-Location).Path
            & docker run --rm -p "127.0.0.1:${Port}:8000" -v "${root}/checkpoints:/app/checkpoints:ro" -v "${root}/data:/app/data:ro" -e "MINIGPT_CHECKPOINT=/app/$Checkpoint" minigpt
            if ($LASTEXITCODE) { exit $LASTEXITCODE }
        }
    }
} finally {
    Pop-Location
}
