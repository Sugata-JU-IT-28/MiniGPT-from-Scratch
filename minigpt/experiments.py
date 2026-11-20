"""Declarative matched-budget TinyShakespeare suites and reproducible model selection."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from typing import Any

from minigpt.config import RunConfig, TrainConfig
from minigpt.model import PRESETS, GPTConfig

BASE = GPTConfig(vocab_size=1024, block_size=128, n_layer=4, n_head=4, n_embd=96, dropout=0.1)
TOKENS_PER_UPDATE = 512
QUICK_TOKENS = 50 * TOKENS_PER_UPDATE
LRS = (3e-4, 1e-3, 3e-3, 6e-3)
SEEDS = {"lr_sweep": 1, "arch_ablation": 3, "shape_ablation": 3, "scaling": 2, "tokenizer": 2}


@dataclass(frozen=True)
class Variant:
    """One model specification; only the LR sweep fixes its own peak LR."""

    name: str
    model: GPTConfig
    preset: str
    lr: float | None = None


def variant(
    name: str, preset: str = "baseline", *, lr: float | None = None, **model_fields: Any
) -> Variant:
    """Apply one declared change to the shared base decoder."""
    return Variant(name, replace(BASE, **(PRESETS[preset] | model_fields)), preset, lr)


SUITES: dict[str, tuple[Variant, ...]] = {
    "lr_sweep": tuple(
        variant(f"{preset}_lr{lr:g}", preset, lr=lr) for preset in PRESETS for lr in LRS
    ),
    "arch_ablation": (
        variant("baseline"),
        variant("rmsnorm", norm_type="rmsnorm"),
        variant("rope", pos_encoding="rope"),
        variant("swiglu", mlp_type="swiglu"),
        variant("modern", "modern"),
    ),
    # The center of all three one-factor grids is one shared control per seed.
    "shape_ablation": (
        variant("base"),
        variant("layers_2", n_layer=2),
        variant("layers_6", n_layer=6),
        variant("heads_2", n_head=2),
        variant("heads_8", n_head=8),
        variant("context_64", block_size=64),
        variant("context_256", block_size=256),
    ),
    "scaling": tuple(
        variant(f"L{depth}_D{width}", n_layer=depth, n_embd=width)
        for depth, width in ((1, 32), (2, 48), (3, 80), (4, 128), (5, 176), (6, 256))
    ),
    "tokenizer": tuple(variant(f"vocab_{size}", vocab_size=size) for size in (512, 1024, 2048)),
}


def run_config(
    spec: Variant, seed: int, tokens: int, lr: float, *, quick: bool = False
) -> RunConfig:
    """Keep optimizer updates and token exposure matched across context lengths."""
    block = spec.model.block_size
    if tokens % TOKENS_PER_UPDATE or TOKENS_PER_UPDATE % block:
        raise ValueError("Budget and context must allow complete matched-token updates")
    steps = tokens // TOKENS_PER_UPDATE
    peak = spec.lr if spec.lr is not None else lr
    opts = TrainConfig(
        token_budget=tokens,
        batch_size=TOKENS_PER_UPDATE // block,
        grad_accum=1,
        lr=peak,
        min_lr=peak / 10,
        warmup_steps=min(steps - 1, max(1, steps // 20)),
        eval_interval=max(1, steps // (2 if quick else 4)),
        eval_batch_size=16,
        checkpoint_interval=max(1, steps // 8),
        log_interval=max(1, steps // 4),
        cpu_threads=2,
    )
    return RunConfig(
        f"{spec.name}_s{seed}",
        spec.preset,
        seed,
        f"data/vocab_{spec.model.vocab_size}",
        spec.model,
        opts,
    )


def select_lr(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one shared LR by averaging endpoint nats/token equally over the two presets."""
    scores: dict[str, dict[float, float]] = {}
    for preset in PRESETS:
        scores[preset] = {}
        for lr in LRS:
            values = [
                r["final"]["val_loss"]
                for r in records
                if r["config"]["preset"] == preset and r["config"]["train"]["lr"] == lr
            ]
            if not values:
                raise ValueError(f"Missing LR result for {preset} at {lr}")
            scores[preset][lr] = statistics.mean(values)
    common_scores = {lr: statistics.mean(scores[p][lr] for p in PRESETS) for lr in LRS}
    best, second, *_ = sorted(LRS, key=lambda lr: (common_scores[lr], lr))
    optima = {p: min(LRS, key=lambda lr: (scores[p][lr], lr)) for p in PRESETS}
    return {
        "shared_lr": best,
        "runner_up": second,
        "runner_up_gap": common_scores[second] - common_scores[best],
        "by_preset": optima,
        "shared_minus_optimum": {p: scores[p][best] - scores[p][optima[p]] for p in PRESETS},
        "endpoint_loss_by_preset": scores,
        "mean_endpoint_loss": common_scores,
        "reason": "Lowest mean endpoint validation loss with equal weight per preset. "
        "One common LR keeps later architecture comparisons controlled.",
    }
