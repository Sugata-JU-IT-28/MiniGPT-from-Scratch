"""Exercise sampling and checkpoint selection against independent small training runs."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from minigpt.checkpoint import load_checkpoint
from minigpt.data import prepare_dataset
from minigpt.model import GPT, GPTConfig
from minigpt.tokenizer import BPETokenizer
from scripts import phase2_summary, sample_quick
from scripts import train as train_cli

ROOT = Path(__file__).resolve().parents[1]


def test_empirical_unigram_is_unsmoothed_and_scores_matching_targets(tmp_path: Path) -> None:
    np.array([1, 1, 1, 2], dtype="<u2").tofile(tmp_path / "train.bin")
    np.array([2, 2, 1], dtype="<u2").tofile(tmp_path / "val.bin")
    result = phase2_summary.sanity_baselines(tmp_path, 257)
    assert result["val_targets"] == 2
    assert result["uniform_loss"] == math.log(257)
    assert result["unigram_loss"] == pytest.approx(-(math.log(0.25) + math.log(0.75)) / 2)
    assert result["unigram_all_val_ids_loss"] == pytest.approx(
        -(2 * math.log(0.25) + math.log(0.75)) / 3
    )
    np.array([1, 103], dtype="<u2").tofile(tmp_path / "val.bin")
    result = phase2_summary.sanity_baselines(tmp_path, 257)
    assert result["unigram_loss"] is None and result["unseen_val_targets"] == 1


@pytest.mark.parametrize("preset", ["baseline", "modern"])
def test_sampling_recomputes_a_cropped_context(preset: str) -> None:
    torch.manual_seed(45)
    model = GPT(
        GPTConfig.from_preset(preset, block_size=4, n_layer=1, n_head=2, n_embd=16, vocab_size=23)
    )
    generated = sample_quick.sample(model, [1, 2, 3], 0.8, 9, 3)
    assert len(generated) == 9 and max(generated) < model.config.vocab_size
    assert generated == sample_quick.sample(model, [1, 2, 3], 0.8, 9, 3)
    # An independent one-draw reference for every prefix, including context eviction.
    g = torch.Generator().manual_seed(3)
    prefix = [1, 2, 3]
    with torch.inference_mode():
        for token in generated:
            logits, _ = model(torch.tensor([prefix[-4:]]))
            assert (
                torch.multinomial((logits[0, -1] / 0.8).softmax(0), 1, generator=g).item() == token
            )
            prefix.append(token)


def test_counts_use_both_real_configs() -> None:
    small = phase2_summary.parameter_counts(ROOT / "configs/gpu_small.yaml")
    tiny = phase2_summary.parameter_counts(ROOT / "configs/cpu_tiny.yaml")
    for preset in ("baseline", "modern"):
        assert 15_000_000 < small[preset]["total"] < 25_000_000
        assert 1_800_000 < tiny[preset]["total"] < 2_200_000
        assert 0 < tiny[preset]["non_embedding"] < tiny[preset]["total"]


@pytest.mark.slow
def test_full_reporting_and_promotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "corpus"
    text = "abcdefghijklmnopqrstuvwxyz" * 20
    prepare_dataset(text, 257, data)
    cfg = {
        "data_dir": str(data),
        "model": {
            "vocab_size": 257,
            "block_size": 8,
            "n_embd": 32,
            "n_head": 4,
            "n_layer": 1,
            "dropout": 0.0,
        },
        "train": {
            "token_budget": 4096,
            "batch_size": 8,
            "lr": 0.008,
            "min_lr": 0.0008,
            "warmup_steps": 2,
            "eval_interval": 32,
            "checkpoint_interval": 32,
            "cpu_threads": 1,
            "device": "cpu",
        },
    }
    path = tmp_path / "fixture.yaml"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    names = ["fixture_baseline", "fixture_modern"]
    for preset in ("baseline", "modern"):
        train_cli.main(["--config", str(path), "--preset", preset, "--seed", "73"])
    phase2_summary.main(
        [
            "--cpu-config",
            str(path),
            "--gpu-config",
            str(ROOT / "configs/gpu_small.yaml"),
            "--seed",
            "73",
        ]
    )
    summary = json.loads(Path("results/phase2_summary.json").read_text(encoding="utf-8"))
    selected = summary["selected"]
    best = min(run["best"]["val_loss"] for run in summary["runs"])
    assert selected["val_loss"] == load_checkpoint(Path(selected["path"]))["best_val"] == best
    assert Path(selected["path"]).read_bytes() == Path(selected["source"]).read_bytes()
    assert hashlib.sha256(Path(selected["path"]).read_bytes()).hexdigest() == selected["sha256"]
    assert Path("docs/training_phase2.md").is_file()
    sample_quick.main(["--seed", "73", "--new-tokens", "12", "--runs", *names])
    samples = json.loads(Path("results/samples_phase2.json").read_text(encoding="utf-8"))
    assert len(samples["samples"]) == 6 and samples["prompt"] == "\n"
    codec = BPETokenizer.load(data / "tokenizer.json")
    report = Path("docs/samples_phase2.md").read_text(encoding="utf-8")
    for row in samples["samples"]:
        assert len(row["token_ids"]) <= 12
        assert codec.decode(row["token_ids"]) == row["text"]
        assert row["text"] in report
