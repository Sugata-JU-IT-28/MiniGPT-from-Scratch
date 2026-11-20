"""Token-budget training with deterministic evaluation and resumable optimizer state."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn

from minigpt.checkpoint import (
    copy_checkpoint,
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
)
from minigpt.config import RunConfig
from minigpt.data import default_device, get_batch, seed_everything
from minigpt.io import file_sha256, write_json
from minigpt.metrics import MetricRow, evaluate, plot_losses, write_metrics
from minigpt.model import GPT


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Decay matrices only. PyTorch's parameter iterator deduplicates tied weights."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def learning_rate(step: int, steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    """Zero-indexed update rate, with linear warmup and the cosine floor on the final update."""
    if warmup_steps and step < warmup_steps:
        return lr * step / warmup_steps
    if step >= steps - 1:
        return min_lr
    fraction = (step - warmup_steps) / (steps - 1 - warmup_steps)
    return min_lr + (lr - min_lr) * (1 + math.cos(math.pi * fraction)) / 2


def accumulate_gradients(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    grad_accum: int,
    *,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> Tensor:
    """Backpropagate the mean loss of equally sized microbatches; return its detached value."""
    total = None
    consumed = 0
    for x, y in batches:
        context = (
            torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else nullcontext()
        )
        with context:
            _, loss = model(x, y)
            scaled = loss / grad_accum
        (scaler.scale(scaled) if scaler is not None else scaled).backward()
        total = scaled.detach() if total is None else total + scaled.detach()
        consumed += 1
    if total is None or consumed != grad_accum:
        raise ValueError("Expected exactly grad_accum equally sized microbatches")
    return total


def _check_data(path: Path, config: RunConfig) -> dict[str, str]:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta["vocab_size"] != config.model.vocab_size or meta["dtype"] != "<u2":
        raise ValueError("Model vocabulary and uint16 dataset metadata must agree")
    for split in ("train", "val"):
        if (path / f"{split}.bin").stat().st_size != 2 * meta[f"{split}_tokens"]:
            raise ValueError(f"Unexpected length of {split}.bin")
    if meta["train_tokens"] <= config.model.block_size:
        raise ValueError("Training split is shorter than one shifted training window")
    return {
        name: file_sha256(path / name)
        for name in ("meta.json", "tokenizer.json", "train.bin", "val.bin")
    }


class Trainer:
    """Own the model, optimizer, sample stream, and durable progress of a single run."""

    def __init__(
        self,
        config: RunConfig,
        root: Path = Path("."),
        *,
        metrics_path: Path | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.root = root
        self.data_dir = root / config.data_dir
        self.data_hashes = _check_data(self.data_dir, config)
        self.run_dir = root / "results/runs" / config.name
        self.metrics_path = metrics_path or self.run_dir / "metrics.csv"
        self.checkpoint_dir = checkpoint_dir or root / "checkpoints" / config.name
        self.last = self.checkpoint_dir / "last.pt"
        self.best = self.checkpoint_dir / "best.pt"
        train = config.train
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.device = default_device() if train.device == "auto" else torch.device(train.device)
        if self.device.type == "cpu" and train.cpu_threads is not None:
            torch.set_num_threads(train.cpu_threads)
        torch.use_deterministic_algorithms(True)
        seed_everything(config.seed)
        # The two presets see identical random windows despite different model RNG consumption.
        self.generator = torch.Generator().manual_seed(config.seed)
        self.model = GPT(config.model).to(self.device)
        self.forward_model = self.model
        self.optimizer = build_optimizer(self.model, train.lr, train.weight_decay)
        self.amp_dtype = None
        self.scaler = None
        if self.device.type == "cuda":
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if self.amp_dtype == torch.float16:
                self.scaler = torch.amp.GradScaler("cuda")
        self.runtime = {
            "device": str(self.device),
            "torch": str(torch.__version__),
            "amp": str(self.amp_dtype),
            "cpu_threads": torch.get_num_threads(),
        }
        if train.compile:
            backend = "inductor"
            if os.name == "nt" and self.device.type == "cpu":
                if shutil.which(os.environ.get("CXX", "cl")) is None:
                    backend = "aot_eager"
            self.forward_model = torch.compile(self.model, backend=backend)
            self.runtime["compile_backend"] = backend
            print(f"torch.compile backend={backend}", flush=True)
        self.step = 0
        self.best_val = math.inf
        self.best_step = -1
        self.rows: list[MetricRow] = []
        self.train_seconds = 0.0
        self.elapsed_before = 0.0
        self.started = perf_counter()

    @property
    def elapsed(self) -> float:
        """Wall seconds spent in active sessions, excluding downtime between resumes."""
        return self.elapsed_before + perf_counter() - self.started

    def resume(self) -> None:
        """Restore a matching run and repair the log from the authoritative last checkpoint."""
        state = load_checkpoint(self.last)
        if state["config"] != asdict(self.config) or state["data_sha256"] != self.data_hashes:
            raise ValueError("Resume requires the original configuration, seed, and dataset")
        if state["runtime"] != self.runtime:
            raise ValueError("Resume requires the original device, AMP mode, threads, and PyTorch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        self.best_val, self.best_step = state["best_val"], state["best_step"]
        self.rows = state["history"]
        self.train_seconds = state["train_seconds"]
        self.elapsed_before = state["elapsed"]
        self.started = perf_counter()
        restore_rng(state["rng"], self.generator)
        if self.best_step == self.step:
            copy_checkpoint(self.last, self.best)
        elif not self.best.is_file():
            raise FileNotFoundError(f"Missing historical best checkpoint {self.best}")
        write_metrics(self.metrics_path, self.rows)

    def _save(self) -> None:
        state = {
            "config": asdict(self.config),
            "data_sha256": self.data_hashes,
            "runtime": self.runtime,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "step": self.step,
            "best_step": self.best_step,
            "best_val": self.best_val,
            "rng": rng_state(self.generator, self.device),
            "history": self.rows,
            "train_seconds": self.train_seconds,
            "elapsed": self.elapsed,
        }
        save_checkpoint(self.last, state)
        if self.best_step == self.step:
            copy_checkpoint(self.last, self.best)
        write_metrics(self.metrics_path, self.rows)

    def _evaluate(self, lr: float) -> None:
        opts = self.config.train
        train = evaluate(
            self.model, "train", data_dir=self.data_dir, batch_size=opts.eval_batch_size
        )
        val = evaluate(self.model, "val", data_dir=self.data_dir, batch_size=opts.eval_batch_size)
        tokens = self.step * self.config.tokens_per_step
        row = MetricRow(
            step=self.step,
            tokens_seen=tokens,
            train_loss=train.loss,
            val_loss=val.loss,
            val_ppl=val.ppl,
            val_bpb=val.bpb,
            lr=lr,
            tokens_per_sec=tokens / self.train_seconds if self.train_seconds else 0.0,
            elapsed=self.elapsed,
        )
        self.rows.append(row)
        if val.loss < self.best_val:
            self.best_val, self.best_step = val.loss, self.step
        print(
            f"eval step={self.step} train={train.loss:.6f} val={val.loss:.6f} "
            f"bpb={val.bpb:.6f} elapsed={self.elapsed:.1f}s",
            flush=True,
        )

    def _update(self, lr: float) -> float:
        train = self.config.train
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        batches = (
            get_batch(
                "train",
                train.batch_size,
                self.config.model.block_size,
                self.device,
                self.generator,
                data_dir=self.data_dir,
            )
            for _ in range(train.grad_accum)
        )
        loss = accumulate_gradients(
            self.forward_model,
            batches,
            train.grad_accum,
            amp_dtype=self.amp_dtype,
            scaler=self.scaler,
        )
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), train.grad_clip)
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return loss.item()

    def run(self, stop_after: int | None = None, *, export_report: bool = True) -> dict[str, Any]:
        """Run through a fixed optimizer step without changing the planned cosine schedule."""
        cfg = self.config
        opts = cfg.train
        end = cfg.steps if stop_after is None else min(stop_after, cfg.steps)
        if end < self.step or end < 0:
            raise ValueError("stop_after precedes the current checkpoint")
        self.model.train()
        print(
            f"{cfg.name} device={self.device} params={self.model.count_params()} "
            f"planned_steps={cfg.steps} tokens_per_step={cfg.tokens_per_step} "
            f"starting_step={self.step}",
            flush=True,
        )
        if not self.rows:
            self._evaluate(learning_rate(0, cfg.steps, opts.warmup_steps, opts.lr, opts.min_lr))
            self._save()
        while self.step < end:
            lr = learning_rate(self.step, cfg.steps, opts.warmup_steps, opts.lr, opts.min_lr)
            started = perf_counter()
            loss = self._update(lr)
            self.train_seconds += perf_counter() - started
            self.step += 1
            if self.step % opts.log_interval == 0:
                rate = self.step * cfg.tokens_per_step / self.train_seconds
                print(
                    f"step={self.step}/{cfg.steps} batch_loss={loss:.5f} "
                    f"lr={lr:.7g} tok/s={rate:.0f}",
                    flush=True,
                )
            measured = self.step % opts.eval_interval == 0 or self.step == cfg.steps
            if measured:
                self._evaluate(lr)
            if measured or self.step % opts.checkpoint_interval == 0 or self.step == end:
                self._save()
        summary = {
            "config": asdict(cfg),
            "data_sha256": self.data_hashes,
            "runtime": self.runtime,
            "params": self.model.count_params(),
            "non_embedding_params": self.model.count_params(True),
            "steps_planned": cfg.steps,
            "step": self.step,
            "tokens_seen": self.step * cfg.tokens_per_step,
            "best_step": self.best_step,
            "best_val": self.best_val,
            "elapsed": self.elapsed,
            "train_seconds": self.train_seconds,
        }
        if export_report:
            plot_losses(self.root / "docs/img" / f"{cfg.name}_loss.png", cfg.name, self.rows)
            write_json(self.run_dir / "run.json", summary)
        return summary
