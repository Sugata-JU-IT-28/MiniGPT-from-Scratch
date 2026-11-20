"""Measure real training updates and evaluation batches before assigning the suite budget."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from minigpt.data import default_device, get_batch, seed_everything
from minigpt.experiments import (
    QUICK_TOKENS,
    SEEDS,
    SUITES,
    TOKENS_PER_UPDATE,
    Variant,
    run_config,
    variant,
)
from minigpt.io import write_json
from minigpt.model import GPTConfig
from minigpt.train import Trainer

PLAN = Path("results/experiment_plan.json")
PROBES = Path("results/experiment_timing.json")
FINAL_CPU = variant("final", "modern", n_embd=256, n_layer=6)
FINAL_GPU = variant("final", "modern", n_embd=384, n_layer=10, n_head=6, block_size=256)


def model_key(model: GPTConfig) -> str:
    """Group timing trials by architecture, context, and vocabulary."""
    return hashlib.sha256(json.dumps(asdict(model), sort_keys=True).encode()).hexdigest()


def probe(spec: Variant, seed: int) -> dict[str, Any]:
    """Time warmed optimizer steps and shifted-target evaluation on real prepared data."""
    cfg = run_config(spec, seed, QUICK_TOKENS, 0.001)
    trainer = Trainer(cfg)
    update_times = []
    for index in range(10):
        start = perf_counter()
        trainer._update(cfg.train.lr)
        if index >= 2:
            update_times.append(perf_counter() - start)
    # The trainer synchronizes each optimization step to read its loss. Also synchronize eval.
    eval_times = []
    trainer.model.eval()
    with torch.inference_mode():
        for index in range(10):
            start = perf_counter()
            x, y = get_batch(
                "val",
                cfg.train.eval_batch_size,
                cfg.model.block_size,
                trainer.device,
                trainer.generator,
                data_dir=trainer.data_dir,
            )
            loss = trainer.model(x, y)[1]
            loss.item()
            if index >= 2:
                eval_times.append(perf_counter() - start)
    meta = json.loads((trainer.data_dir / "meta.json").read_text(encoding="utf-8"))
    return {
        "model": asdict(cfg.model),
        "params": trainer.model.count_params(),
        "non_embedding_params": trainer.model.count_params(True),
        "step_seconds": statistics.median(update_times),
        "step_trials": update_times,
        "eval_batch_seconds": statistics.median(eval_times),
        "eval_batch_trials": eval_times,
        "eval_batch_tokens": cfg.train.eval_batch_size * cfg.model.block_size,
        "split_targets": meta["train_tokens"] + meta["val_tokens"] - 2,
    }


def estimate(
    timings: dict[str, Any], budgets: int, seeds: dict[str, int], final: Variant
) -> dict[str, float]:
    """Include training, five full-split evaluations, checkpoint overhead, and headroom."""

    def one(spec: Variant, tokens: int) -> float:
        timing = timings[model_key(spec.model)]
        updates = tokens // TOKENS_PER_UPDATE
        evaluation = (
            timing["split_targets"] / timing["eval_batch_tokens"] * timing["eval_batch_seconds"]
        )
        return 1.25 * (updates * timing["step_seconds"] + 5 * evaluation + 3.0)

    times = {
        suite: sum(one(v, budgets) for v in specs) * seeds[suite] for suite, specs in SUITES.items()
    }
    times["final"] = one(final, 4 * budgets)
    return times


def make_plan(seed: int) -> dict[str, Any]:
    """Resume per-shape probes and choose the largest candidate under the time ceiling."""
    seed_everything(seed)
    device = default_device()
    final = FINAL_CPU if device.type == "cpu" else FINAL_GPU
    context = {
        "seed": seed,
        "device": str(device),
        "torch": str(torch.__version__),
        "tokens_per_update": TOKENS_PER_UPDATE,
        "cpu_threads": 2,
    }
    readings: dict[str, Any] = {"context": context, "timings": {}}
    if PROBES.exists():
        readings = json.loads(PROBES.read_text(encoding="utf-8"))
        if readings["context"] != context:
            raise ValueError("Saved timing probe settings differ from this machine or seed")
    for spec in [v for specs in SUITES.values() for v in specs] + [final]:
        key = model_key(spec.model)
        if key not in readings["timings"]:
            print(
                f"Probing {spec.name} {spec.model.n_embd} wide, {spec.model.n_layer} layers",
                flush=True,
            )
            readings["timings"][key] = probe(spec, seed)
            write_json(PROBES, readings)
    seeds = SEEDS.copy()
    # A ceiling below one training-split pass leaves room for replicated controlled fits.
    choices = range(262144, 32767, -16384)

    def choose() -> tuple[int, dict[str, float]]:
        for budget in choices:
            times = estimate(readings["timings"], budget, seeds, final)
            if sum(times.values()) <= 3 * 3600:
                return budget, times
        return budget, times

    budget, times = choose()
    if sum(times.values()) > 4 * 3600:
        seeds["shape_ablation"] = 2
        budget, times = choose()
        print("Shape ablation reduced to two seeds after the four-hour timing check.", flush=True)
    plan = {
        **context,
        "token_budget": budget,
        "final_token_budget": 4 * budget,
        "seeds": seeds,
        "estimated_seconds": times,
        "total_estimated_seconds": sum(times.values()),
        "estimate_method": "Median of eight warmed updates and evaluation batches per shape, "
        "five full train/val evaluations, three seconds overhead per run, "
        "and 25% headroom. Final timing uses the modern largest model proxy.",
        "runs": {s: len(v) * seeds[s] for s, v in SUITES.items()},
    }
    write_json(PLAN, plan)
    return plan
