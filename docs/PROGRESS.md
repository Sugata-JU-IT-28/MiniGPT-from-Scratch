# Project progress

## 2026-11-19 · Tokenizer and data pipeline

Implemented deterministic byte-level BPE and the memory-mapped batch loader. Prepared TinyShakespeare at 512, 1,024, and 2,048 vocabulary entries. All saved splits decoded byte-for-byte to their raw text. The default 1,024-entry tokenizer produced 411,252 training tokens and 49,411 validation tokens, or 2.2573 held-out bytes per token. [Preparation results](../results/prepare_vocab_1024.json)

The resumable vocabulary study measured diminishing token savings per added embedding row. At the default vocabulary, the English probe reached 2.4090 bytes/token. The Bengali and Hindi probes each used one token per UTF-8 byte. [Study and chart](tokenizer_study.md) · [Measurements](../results/tokenizer_study.json)

Ruff passed. Pytest recorded 32 passed and 1 skipped. CUDA batch transfer remains unverified because the installed PyTorch build runs on CPU. [Verification record](../results/verification.json)

## 2026-11-19 · Causal decoder and full training runs

Completed the model, resumable trainer, and matched TinyShakespeare runs. Each preset trained on 3,145,728 tokens with seed 42 and reached its best validation loss at the final update. Modern scored 3.3270 nats/token and baseline scored 3.5770. Both beat the train unigram (5.7085) and uniform reference (6.9314). Reloading both checkpoints reproduced those losses. The modern checkpoint was promoted to `checkpoints/best.pt` and its checksum matched the source. [Results and checkpoint selection](../results/phase2_summary.json)

Validation loss fell at every recorded evaluation for both presets. Modern fitted the training split more closely, with a final validation-minus-training gap of 0.5250 versus 0.3070 for baseline. The six newline-prompted continuations show what the lower loss buys in this run. At temperature 0.5, modern follows a BUCKINGHAM cue with “And I will find the business of my heart.” Baseline also switches speakers and uses “thou wast,” but invents “musicience.” At temperature 1.0, baseline opens with “Bappructionpon” and modern later writes “These duke is bird of cions.” Both learned dialogue layout and archaic diction more reliably than sustained meaning. [Learning curves and rerun commands](training_phase2.md) · [Generated samples](samples_phase2.md)

Ruff passed. The full pytest suite recorded 95 passed and 2 skipped. CUDA batch transfer and AMP remain unverified on this CPU-only build. Baseline took 21.1 minutes and modern took 22.2 minutes including evaluation, both slightly above the laptop timing target. [Verification record](../results/phase2_verification.json) · [Run timings](../results/phase2_summary.json)

## 2026-11-19 · Cached generation and experiment budget

Before starting the experiments, the full suite recorded 108 passing tests and two CUDA skips. Cached continuations matched the corresponding full-forward logits and attention rows, including multi-token continuations. The trained checkpoint produced identical cached and uncached greedy IDs at both tested lengths. Measured cache bytes matched the formula in all 20 benchmark cases. [Test record](../results/phase3_part_a_verification.json) · [Timing and memory measurements](kv_cache_benchmark.md)

The CPU probe selected 212,992 training tokens per experiment and 851,968 for the final run. Its advance estimate for all fits was 2 hours 57 minutes. The shape ablation kept three seeds and shares the central model across its three one-factor grids. Run `python -m scripts.run_experiments --seed 0` to resume the suites. Each run restores its last optimizer checkpoint or skips a published result. [Timing plan](../results/experiment_plan.json)

## 2026-11-20 · Learning-rate decision and Part B resume

Both presets achieved their lowest seed-zero endpoint validation loss at 0.003. The equally weighted mean was 4.4825 nats/token. The runner-up, 0.006, was 0.1572 nats/token higher. Sharing 0.003 costs zero measured loss relative to each preset's own optimum on this grid. The short, single-seed sweep supports a controlled choice for the later suites. [Full grid and decision](../results/lr_selection.json)

From the repository root in PowerShell, inspect `powershell -File scripts/phase3_jobs.ps1 -Action Status`. If the supervisor has exited before aggregation, resume with `powershell -File scripts/phase3_jobs.ps1 -Action Start -Seed 0`. Its first stage fills the isolated quick grids and generates their tables and plots. It then runs the four remaining full suites in order, checks the expected files, and aggregates. Its PID and logs are in `logs/supervisor.pid`, `logs/supervisor.log` and `logs/supervisor.err`. A completed result is reused and an unfinished run restores its optimizer checkpoint. The original final-fit budget in the timing plan has been superseded by the Phase 2 token budget for the requested LR transfer check and final comparison. [Phase 2 budget](../results/phase2_summary.json)

After the grids finish, run `.venv\Scripts\python.exe -m scripts.aggregate_results --seed 0` and inspect the table and figures. Use `.venv\Scripts\python.exe -m scripts.train_final --plan-only --token-budget 3145728 --seed 0` for the shape and time estimate. Launch or resume the two LR probes and final fit with `powershell -File scripts/phase3_jobs.ps1 -Action Final -Seed 0`. The same status command monitors the final PID and log. Review the best checkpoint against Phase 2 and write the analysis from the results. Finish with `.venv\Scripts\python.exe -m scripts.verify_phase1 --seed 0 --output results/phase3_verification.json`. [Preflight test results](../results/phase3_preflight.json)

## 2026-11-20 · Phase 3 Part B complete

The isolated quick report and all 62 full-budget runs finished. No planned variant or seed is missing. Modern led the architecture suite at 4.3831 ± 0.0203 nats/token across three seeds. The default experiment budget covered 0.5179 training-split equivalents. The 15 architecture fits took 29 minutes 50 seconds, against 34 minutes 22 seconds planned. The quick supervisor recovered a Windows CSV replacement error from the affected optimizer checkpoint. The shared atomic writer now retries transient Windows sharing failures. [Completeness](../results/phase3_completeness.json) · [Measured results](../results/experiment_aggregate.json) · [Timing update](../results/phase3_remaining_time.json)

On the final shape, the seed-zero LR probes favored 0.0015 over 0.003. The 5,024,576-parameter modern model trained for 3,145,728 tokens. Its lowest measured validation loss came at the last update. The reloaded checkpoint scored 3.3017 nats/token and 2.1101 bits/byte, below the Phase 2 best of 3.3270 and 2.1262 respectively. Cached and uncached greedy decoding matched for 256 tokens. The checkpoint comparison uses one seed per fit. Ruff passed and the full suite recorded 118 passed and two skips. CUDA batch transfer and CUDA AMP remain unverified on this CPU installation. The resume commands above also reproduce the reports from saved runs. [Transfer probes](../results/final_lr_check.json) · [Final checkpoint measurements](../results/final_summary.json) · [Verification](../results/phase3_verification.json)

### Phase 4 readiness checklist

| Artifact | Status | Location |
|---|---|---|
| Best and last checkpoints | Present | [final.pt](../checkpoints/final.pt) and [final_last.pt](../checkpoints/final_last.pt) have identical SHA-256 values because the last update was best. |
| Matching tokenizer | Present | [Vocabulary 1,024 tokenizer](../data/vocab_1024/tokenizer.json). Its SHA-256 is recorded in the [final summary](../results/final_summary.json). |
| Experiment results | Present | [Aggregated measurements](../results/experiment_aggregate.json), [LR decision](../results/lr_selection.json), [transfer probes](../results/final_lr_check.json) and [per-run JSON and CSV files](../results/runs/). |
| Tables and plots | Present | [Generated tables and six experiment figures](results_tables.md). |
| Written interpretation | Present | [Analysis and limitations](analysis.md). |
| KV benchmark | Present | [Report](kv_cache_benchmark.md), [figure](img/kv_cache_benchmark.png) and [raw measurements](../results/kv_cache_benchmark.json). |
| Generation samples | Present | [Nine continuations](samples_final.md) and [token IDs and greedy check](../results/final_model.json). |

## 2026-11-20 · Phase 4 glass-box API, UI and packaging

Added the checkpoint-backed token inspector, streaming API, study notes, results-backed README, Docker packaging, CI and Windows tasks. The trace exposes raw and filtered probability, entropy, token bytes and causal attention for each draw. Captured both layouts in Chrome. The final trainer can rebuild weights from a checkout containing only published metrics. Existing experiments supplied all model measurements. [README](../README.md) · [Design notes](design_notes.md) · [Desktop](img/glass_box_desktop.png) · [Mobile](img/glass_box_mobile.png)

Ruff passed. The full CPU suite, including slow tests, recorded 176 passed and 2 CUDA skips. The CI test selection also passed in an isolated source tree without local assets. The real-checkpoint smoke test verified all 7 routes and server shutdown. The built wheel served its packaged UI outside the checkout. The checkpoint and tokenizer hashes matched the final summary. At 60,401,259 bytes, final.pt is a candidate for a GitHub release asset. [Test results](../results/phase4_verification.json) · [Smoke and file size](../results/phase4_smoke.json) · [Recorded hashes](../results/final_summary.json)

Open checks are the Docker build and runtime, the hosted CI matrix, and CUDA transfer and AMP. Docker and the command named `make` were unavailable. The smoke script ran directly under the Windows venv, the PowerShell lint target passed, and MinGW Make expanded the recipes in a dry run. Starlette emitted a deprecation warning for the project's allowed httpx test client.

## 2026-11-21 · GitHub publication preparation

Confirmed Git is installed and this is an existing, clean `main` repository with a configured Git identity and no `origin`. Reviewed tracked paths and reachable history: no datasets, checkpoints, virtual environments, or oversized tracked files were found. No common credential or private-key markers were found in tracked files. Extended `.gitignore` for local secrets, checkpoints, datasets, and temporary/runtime files. GitHub CLI is not installed, so creating or checking an account-owned repository remains a manual GitHub.com step. No source files, commits, remotes, or pushes were changed. The GitHub account/repository URL remains unverified.
