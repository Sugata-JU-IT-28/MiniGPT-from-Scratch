"""Exercise batched memory measurements and both command-line output modes."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from minigpt.checkpoint import save_checkpoint
from minigpt.io import file_sha256
from minigpt.model import GPT, GPTConfig
from minigpt.tokenizer import BPETokenizer
from scripts import generate as cli
from scripts.benchmark_kv_cache import measure


@pytest.mark.parametrize("preset", ["baseline", "modern"])
def test_batched_cache_storage_and_benchmark(preset: str) -> None:
    cfg = GPTConfig.from_preset(
        preset, vocab_size=23, block_size=24, n_embd=16, n_head=2, n_layer=2, dropout=0
    )
    model = GPT(cfg).eval()
    row = measure(model, torch.tensor([[2, 1, 3], [0, 1, 2]]), 7, 3)
    assert row["cached_positions"] == 9
    assert row["cache_bytes"] == row["formula_bytes"] == 2 * 2 * 2 * 9 * 16 * 4
    assert row["difference_bytes"] == 0
    assert len(row["cached_trials_seconds"]) == len(row["uncached_trials_seconds"]) == 3
    assert row["speedup"] == row["uncached_seconds"] / row["cached_seconds"]


def test_generation_cli_stream_and_plain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    torch.manual_seed(29)
    cfg = GPTConfig(vocab_size=257, block_size=16, n_embd=16, n_head=2, n_layer=1, dropout=0)
    vocab = tmp_path / "tokenizer.json"
    BPETokenizer().save(vocab)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        {
            "model": GPT(cfg).state_dict(),
            "config": {
                "model": asdict(cfg),
                "data_dir": str(tmp_path),
                "train": {"cpu_threads": 1},
            },
            "data_sha256": {"tokenizer.json": file_sha256(vocab)},
        },
    )
    args = [
        "--checkpoint",
        str(path),
        "--tokenizer",
        str(vocab),
        "--prompt",
        "tea",
        "--temperature",
        "0.8",
        "--top-k",
        "9",
        "--top-p",
        "0.9",
        "--max-new-tokens",
        "10",
        "--seed",
        "5",
    ]
    cli.main(args)
    plain = capsys.readouterr().out
    cli.main([*args, "--stream", "--no-cache"])
    assert capsys.readouterr().out == plain
    assert plain.startswith("tea")
