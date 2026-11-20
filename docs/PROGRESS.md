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
