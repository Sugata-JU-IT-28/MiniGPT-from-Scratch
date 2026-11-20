"""Load a training experiment from YAML, a preset, and explicit dotted overrides."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from minigpt.model import GPTConfig


@dataclass(frozen=True)
class TrainConfig:
    """Budget, optimizer settings, and reporting intervals in optimizer updates."""

    token_budget: int
    batch_size: int
    lr: float
    min_lr: float
    warmup_steps: int
    eval_interval: int
    checkpoint_interval: int
    grad_accum: int = 1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_batch_size: int = 16
    log_interval: int = 25
    device: Literal["auto", "cpu", "cuda"] = "auto"
    cpu_threads: int | None = None
    compile: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.token_budget,
            self.batch_size,
            self.grad_accum,
            self.eval_interval,
            self.checkpoint_interval,
            self.eval_batch_size,
            self.log_interval,
        )
        if any(type(n) is not int or n <= 0 for n in positive):
            raise ValueError("Budgets, batches and intervals must be positive integers")
        if self.cpu_threads is not None and self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if not 0 <= self.min_lr <= self.lr or self.warmup_steps < 0:
            raise ValueError("Require 0 <= min_lr <= lr and nonnegative warmup_steps")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Require nonnegative weight decay and positive grad_clip")


@dataclass(frozen=True)
class RunConfig:
    """Fully resolved configuration stored in every checkpoint."""

    name: str
    preset: str
    seed: int
    data_dir: str
    model: GPTConfig
    train: TrainConfig

    @property
    def tokens_per_step(self) -> int:
        """Count training targets per optimizer update, including accumulation."""
        return self.train.batch_size * self.train.grad_accum * self.model.block_size

    @property
    def steps(self) -> int:
        """Fit as many complete optimizer updates as the token budget allows."""
        return self.train.token_budget // self.tokens_per_step

    def __post_init__(self) -> None:
        if not self.name or Path(self.name).name != self.name or self.name in (".", ".."):
            raise ValueError("run name must be a single directory name")
        if self.steps < 1 or self.train.warmup_steps >= self.steps:
            raise ValueError("The token budget must allow at least one update after warmup")


def _value(raw: str) -> Any:
    # JSON accepts scientific notation such as 1e-3, which YAML 1.1 reads as text.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return yaml.safe_load(raw)


def load_config(
    path: Path,
    overrides: Sequence[str] = (),
    *,
    preset: str | None = None,
    seed: int | None = None,
    run_name: str | None = None,
) -> RunConfig:
    """Resolve the YAML preset, explicit model fields, and CLI overrides."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for argument in overrides:
        key, equals, raw = argument.partition("=")
        group, dot, field = key.partition(".")
        if not equals or not dot or group not in ("model", "train") or not field:
            raise ValueError(f"Expected model.field=value or train.field=value, got {argument!r}")
        data[group][field] = _value(raw)
    yaml_preset = data["model"].pop("preset", "baseline")
    selected = preset or yaml_preset
    return RunConfig(
        name=run_name or data.get("run_name", f"{path.stem}_{selected}"),
        preset=selected,
        seed=data.get("seed", 42) if seed is None else seed,
        data_dir=data.get("data_dir", "data/vocab_1024"),
        model=GPTConfig.from_preset(selected, **data["model"]),
        train=TrainConfig(**data["train"]),
    )
