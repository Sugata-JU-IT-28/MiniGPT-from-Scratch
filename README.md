# MiniGPT

A GPT-style language model built by hand in PyTorch. The decoder supports learned positions or RoPE, LayerNorm or RMSNorm, and GELU or SwiGLU. The baseline and modern presets train on the same byte-level BPE data.

## Reproduce the tokenizer study

From the repository root in a Python virtual environment, run the following commands.

```sh
python -m pip install -e ".[dev]"
python scripts/prepare_data.py
python scripts/prepare_data.py --vocab-size 512 --out-dir data/vocab_512
python scripts/prepare_data.py --vocab-size 2048 --out-dir data/vocab_2048
python scripts/tokenizer_study.py
python -m ruff check .
python -m pytest
```

The first preparation downloads TinyShakespeare to `data/raw/input.txt`. Both scripts reuse this file offline or accept a local UTF-8 file via `--input`. The study checkpoints completed vocabulary sizes and accepts `--restart` for a fresh timing run. To record lint and test outcomes in `results/verification.json`, run `python scripts/verify_phase1.py`.

[Tokenizer study](docs/tokenizer_study.md) · [Measured results](results/tokenizer_study.json) · [Project progress](docs/PROGRESS.md)

## Train and compare the models

With the default dataset prepared, run these commands from the activated project environment.

```sh
python scripts/train.py --config configs/cpu_tiny.yaml --preset baseline --seed 42
python scripts/train.py --config configs/cpu_tiny.yaml --preset modern --seed 42
python scripts/phase2_samples.py --seed 42
python scripts/phase2_summary.py --seed 42
```

Append `--resume` to a training command to continue from its last checkpoint with the original budget and random states. For a separate experiment, use `--run-name` and dotted overrides such as `--override model.pos_encoding=rope`. The training CSV records full-split loss at each evaluation. Throughput counts training tokens per second spent on optimizer updates, and elapsed time also includes evaluation.

Set `--override train.compile=true` to enable `torch.compile`. On Windows CPU, the trainer selects AOT eager when the configured native C++ compiler is missing from the path. Each compiled run records its backend.

[Training comparison and curves](docs/training_phase2.md) · [Generated samples](docs/samples_phase2.md) · [Checkpoint selection](results/phase2_summary.json)

## Generate and inspect tokens

```sh
python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "KING RICHARD:" --stream
python scripts/benchmark_kv_cache.py --seed 42
```

`minigpt.generate.generate` returns continuation IDs. Its streaming counterpart yields the sampled ID, both vocabulary distributions and their entropies in nats. Use `step.top_candidates(k)` for the filtered probabilities or pass `raw=True` for the model probabilities. A cached call to `next_token_distribution` takes the unprocessed suffix. Supply the full history when using a repetition penalty.

The penalty changes only seen logits. Renormalization also changes unseen probabilities, preserving their relative odds before truncation. The sampler then applies temperature, top-k and nucleus filtering, with lower token IDs winning ties. Greedy decoding uses the highest penalized logit.

The default policy raises when the prompt and requested continuation exceed the context capacity. Use `--window-policy reprefill` for exact sliding-window output or `--no-cache` for a reference decode. [Cache measurements and context policy](docs/kv_cache_benchmark.md)

## Reproduce or resume the controlled experiments

Run these commands from the repository root with the prepared vocabularies.

```sh
python -m scripts.run_experiments --plan-only --seed 0
python -m scripts.run_experiments --seed 0
python -m scripts.aggregate_results --seed 0
python -m scripts.train_final --seed 0
python -m scripts.generate --checkpoint checkpoints/final.pt --prompt "KING RICHARD:" --stream
```

The runner times each distinct model shape, prints its wall-clock estimate, then trains the LR grid before using the selected shared rate for the other suites. The default seed counts and token budget come from the timing plan. Select a suite with `--suite` and set its replicate count with `--seeds`. Each completed run has a JSON record and a CSV in `results/runs/<suite>/`. Repeating the command skips complete results and resumes interrupted runs from `checkpoints/experiments/`. [Plan and timing estimate](results/experiment_plan.json)

For a pipeline check, run `python -m scripts.run_experiments --suite lr_sweep --only baseline_lr0.0003 --quick --seed 0`. Quick records live under `results/quick/`. Aggregation checks the planned configurations, budgets and CSV histories before writing the tables and figures. The final training command reloads the published checkpoint, reevaluates it, and generates the samples.

[Result tables and plots](docs/results_tables.md) · [Analysis](docs/analysis.md) · [Final samples](docs/samples_final.md)
