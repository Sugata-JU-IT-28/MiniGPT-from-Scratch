"""Whole-split metrics, configuration resolution, exact replay, and training entry points."""

import csv
import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from minigpt.checkpoint import load_checkpoint, restore_rng, rng_state, run_writer
from minigpt.config import RunConfig, TrainConfig, load_config
from minigpt.data import prepare_dataset, seed_everything
from minigpt.model import GPT, GPTConfig
from minigpt.train import Trainer, evaluate, learning_rate
from scripts import train as train_cli

ROOT = Path(__file__).resolve().parents[1]


def tiny_run(name: str, dropout: float = 0.0) -> RunConfig:
    model = GPTConfig.from_preset(
        "modern", vocab_size=257, n_embd=24, n_head=3, block_size=8, n_layer=1, dropout=dropout
    )
    opts = TrainConfig(
        token_budget=2560,
        batch_size=4,
        grad_accum=2,
        lr=0.002,
        min_lr=0.0002,
        warmup_steps=4,
        eval_interval=13,
        checkpoint_interval=9,
        cpu_threads=1,
        device="cpu",
    )
    return RunConfig(
        name=name, preset="modern", data_dir="corpus", seed=116, model=model, train=opts
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A real UTF-8 preparation with a non-block-aligned validation suffix."""
    path = tmp_path / "corpus"
    prepare_dataset("A raven stirred its wings and flew.\n" * 9 + "مেঘ終", 257, path)
    return path


def test_dotted_overrides_and_budget() -> None:
    cfg = load_config(
        ROOT / "configs/cpu_tiny.yaml",
        [
            "model.pos_encoding=rope",
            "train.lr=1e-3",
            "train.token_budget=103001",
            "train.warmup_steps=4",
        ],
        seed=13,
    )
    assert cfg.model.pos_encoding == "rope" and cfg.model.norm_type == "layernorm"
    assert cfg.model.mlp_type == "gelu" and cfg.train.lr == 0.001 and cfg.seed == 13
    assert cfg.steps == 103001 // cfg.tokens_per_step
    assert cfg.steps * cfg.tokens_per_step <= cfg.train.token_budget
    modern = load_config(ROOT / "configs/gpu_small.yaml", preset="modern")
    assert (modern.model.norm_type, modern.model.pos_encoding, modern.model.mlp_type) == (
        "rmsnorm",
        "rope",
        "swiglu",
    )
    assert modern.model.block_size == 256 and modern.train.compile is False
    with pytest.raises(ValueError, match="Expected model.field"):
        load_config(ROOT / "configs/cpu_tiny.yaml", ["foo=2"])


def test_lr_endpoints_and_monotonic_sections() -> None:
    schedule = [learning_rate(i, 21, 4, 0.001, 0.0001) for i in range(21)]
    assert schedule[0] == 0
    assert schedule[2] == 0.0005
    assert schedule[4] == max(schedule) == 0.001
    assert schedule[-1] == 0.0001
    assert schedule[:5] == sorted(schedule[:5])
    assert schedule[4:] == sorted(schedule[4:], reverse=True)
    assert learning_rate(0, 21, 0, 0.001, 0.0001) == 0.001


def test_cli_allows_one_writer_per_run(tmp_path: Path) -> None:
    with run_writer(tmp_path / "same"):
        with pytest.raises(RuntimeError, match="writer lock"), run_writer(tmp_path / "same"):
            pytest.fail("A second writer acquired the run")
        with run_writer(tmp_path / "other"):
            pass
    with run_writer(tmp_path / "same"):
        pass


@pytest.mark.parametrize("split", ["train", "val"])
def test_evaluation_counts_every_target_including_tail(corpus: Path, split: str) -> None:
    torch.manual_seed(73)
    cfg = tiny_run("eval", dropout=0.5)
    model = GPT(cfg.model)
    raw = torch.from_numpy(np.fromfile(corpus / f"{split}.bin", dtype="<u2").astype(np.int64))
    assert (raw.numel() - 1) % cfg.model.block_size
    meta = json.loads((corpus / "meta.json").read_text())
    # Deliberately slow one-window reference, independent of the batched sweep.
    model.eval()
    nll = 0.0
    with torch.no_grad():
        for start in range(0, len(raw) - 1, cfg.model.block_size):
            window = raw[start : start + cfg.model.block_size + 1]
            logits, _ = model(window[:-1][None])
            nll += F.cross_entropy(logits[0], window[1:], reduction="sum").item()
    model.train()
    state = torch.get_rng_state()
    metrics = evaluate(model, split, data_dir=corpus, batch_size=5)
    assert model.training and torch.equal(state, torch.get_rng_state())
    assert metrics.tokens == raw.numel() - 1
    assert metrics.loss == pytest.approx(nll / metrics.tokens, abs=1e-6)
    assert metrics.ppl == pytest.approx(math.exp(metrics.loss))
    ratio = (
        meta["bytes_per_token"] if split == "val" else meta["train_bytes"] / meta["train_tokens"]
    )
    assert metrics.bpb == pytest.approx(metrics.loss / math.log(2) / ratio)
    assert evaluate(model, split, data_dir=corpus, batch_size=5) == metrics


@pytest.mark.slow
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_resume_exactly_matches_forty_updates(tmp_path: Path, corpus: Path, dropout: float) -> None:
    straight = Trainer(tiny_run("straight", dropout), tmp_path)
    straight.run()
    cfg = tiny_run("resumed", dropout)
    interrupted = Trainer(cfg, tmp_path)
    interrupted.run(stop_after=20)
    state = load_checkpoint(interrupted.last)
    assert state["step"] == 20 and state["optimizer"]["state"]
    csv_path = interrupted.run_dir / "metrics.csv"
    with csv_path.open("a") as stream:
        stream.write("partial row beyond checkpoint\n")
    seed_everything(999)
    resumed = Trainer(cfg, tmp_path)
    resumed.resume()
    resumed.run()
    for name, weight in straight.model.state_dict().items():
        diff = (weight - resumed.model.state_dict()[name]).abs().max().item()
        assert diff < 1e-6, (name, diff)
    assert torch.equal(straight.generator.get_state(), resumed.generator.get_state())
    assert resumed.step == straight.step == 40
    assert resumed.best_val == straight.best_val
    assert resumed.best_step == straight.best_step
    for a, b in zip(straight.rows, resumed.rows, strict=True):
        for field in ("step", "tokens_seen", "lr", "val_loss", "train_loss", "val_bpb"):
            assert a[field] == b[field]
    with csv_path.open() as stream:
        assert [int(row["step"]) for row in csv.DictReader(stream)] == [0, 13, 26, 39, 40]
    assert load_checkpoint(resumed.best)["step"] == resumed.best_step
    changed = Trainer(replace(cfg, seed=51), tmp_path)
    with pytest.raises(ValueError, match="Resume requires"):
        changed.resume()


def test_rng_checkpoint_restores_all_cpu_streams() -> None:
    seed_everything(9)
    generator = torch.Generator().manual_seed(113)
    saved = rng_state(generator, torch.device("cpu"))

    def draws() -> tuple[float, float, float, float]:
        return (
            random.random(),
            float(np.random.random()),
            torch.rand(()).item(),
            torch.rand((), generator=generator).item(),
        )

    expected = draws()
    restore_rng(saved, generator)
    assert draws() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA AMP")
def test_cuda_amp_training_and_checkpoint(tmp_path: Path, corpus: Path) -> None:
    cfg = tiny_run("amp")
    opts = replace(cfg.train, device="cuda", token_budget=128, warmup_steps=0)
    trainer = Trainer(replace(cfg, train=opts), tmp_path)
    expected = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    assert trainer.amp_dtype == expected
    assert (trainer.scaler is None) == (expected == torch.bfloat16)
    trainer.run()
    state = load_checkpoint(trainer.last)
    assert state["step"] == 2 and math.isfinite(state["best_val"])
    assert bool(state["rng"]["cuda"])
    assert (state["scaler"] is None) == (expected == torch.bfloat16)


@pytest.mark.parametrize("compiled", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_train_cli_end_to_end(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch, compiled: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tiny_run("cli")
    path = tmp_path / "tiny.yaml"
    path.write_text(
        json.dumps(
            {"data_dir": str(corpus), "model": asdict(cfg.model), "train": asdict(cfg.train)}
        )
    )
    args = [
        "--config",
        str(path),
        "--preset",
        "modern",
        "--seed",
        "123",
        "--run-name",
        "cli",
        "--override",
        "train.token_budget=128",
        "train.warmup_steps=0",
        f"train.compile={str(compiled).lower()}",
    ]
    train_cli.main([*args, "--stop-after", "1"])
    train_cli.main([*args, "--resume"])
    result = json.loads((tmp_path / "results/runs/cli/run.json").read_text())
    assert result["tokens_seen"] == 128 and result["step"] == 2
    assert result["config"]["seed"] == 123 and result["config"]["preset"] == "modern"
    assert ("compile_backend" in result["runtime"]) == compiled
    assert (tmp_path / "docs/img/cli_loss.png").stat().st_size > 0
