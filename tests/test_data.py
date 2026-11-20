"""Exercise real files, held-out boundaries, seeding, and batch offsets."""

import io
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from minigpt import corpus
from minigpt.data import TOKEN_DTYPE, default_device, get_batch, prepare_dataset, seed_everything
from minigpt.tokenizer import BPETokenizer
from scripts import prepare_data


def test_prepare_cli_has_no_validation_leakage(tmp_path: Path) -> None:
    train, val = "a " * 44 + "bc", "z" * 10
    raw = tmp_path / "raw.txt"
    raw.write_bytes((train + val).encode("utf-8"))
    out, results = tmp_path / "data", tmp_path / "results"
    prepare_data.main(
        [
            "--input",
            str(raw),
            "--vocab-size",
            "259",
            "--out-dir",
            str(out),
            "--results-dir",
            str(results),
            "--seed",
            "17",
        ]
    )
    tokenizer = BPETokenizer.load(out / "tokenizer.json")
    assert tokenizer.merges == BPETokenizer().train(train, 259).merges
    assert tokenizer.merges != BPETokenizer().train(train + val, 259).merges
    meta = json.loads((out / "meta.json").read_text())
    for split, text in (("train", train), ("val", val)):
        ids = np.fromfile(out / f"{split}.bin", dtype="<u2").tolist()
        assert ids == tokenizer.encode(text)
        assert meta[f"{split}_tokens"] == len(ids)
        assert meta[f"{split}_bytes"] == len(text.encode())
        assert max(ids) < meta["vocab_size"] == 259
    assert meta["bytes_per_token"] == 1.0
    result = json.loads((results / "prepare_vocab_259.json").read_text())
    assert result == {"seed": 17, **meta}


def test_split_code_points_preserve_raw_utf8_and_crlf(tmp_path: Path) -> None:
    text = "🧶দ" * 7 + "\r\n\t e\u0301"
    path = tmp_path / "raw.txt"
    path.write_bytes(text.encode())
    assert corpus.read_corpus(path) == text
    train, val = corpus.split_text(text)
    assert train == text[:18] and val == text[18:]
    meta = prepare_dataset(text, 257, tmp_path)
    assert meta["train_bytes"] == len(train.encode())
    assert meta["val_bytes"] == len(val.encode())


def test_download_once_and_error_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = tmp_path / "data/raw/input.txt"
    monkeypatch.setattr(corpus, "RAW_PATH", raw)

    def fail(url: str, *, timeout: int) -> io.BytesIO:
        raise OSError("simulated offline")

    monkeypatch.setattr(corpus, "urlopen", fail)
    with pytest.raises(SystemExit, match=r"Place the file at .*data/raw/input.txt"):
        corpus.read_corpus()
    assert not raw.exists()

    def download(url: str, *, timeout: int) -> io.BytesIO:
        assert url == corpus.DATASET_URL
        return io.BytesIO(b"real test fixture\r\n")

    monkeypatch.setattr(corpus, "urlopen", download)
    assert corpus.read_corpus() == "real test fixture\r\n"
    monkeypatch.setattr(corpus, "urlopen", fail)
    assert corpus.read_corpus() == "real test fixture\r\n"


@pytest.mark.parametrize("size", [256, 65536])
def test_uint16_guard(tmp_path: Path, size: int) -> None:
    with pytest.raises(ValueError, match="uint16"):
        prepare_dataset("abc", size, tmp_path)


def test_batch_memmap_offsets_and_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for split, start in (("train", 0), ("val", 80)):
        np.arange(start, start + 20, dtype=TOKEN_DTYPE).tofile(tmp_path / f"{split}.bin")
    original = np.memmap
    opened: list[Path] = []

    def tracked(path: Path, *, mode: str, dtype: np.dtype) -> np.memmap:
        opened.append(path)
        assert mode == "r" and dtype == TOKEN_DTYPE
        return original(path, dtype=dtype, mode=mode)

    monkeypatch.setattr(np, "memmap", tracked)
    for split, low in (("train", 0), ("val", 80)):
        for seed in (3, 17):
            generator = torch.Generator().manual_seed(seed)
            x, y = get_batch(split, 64, 4, "cpu", generator, data_dir=tmp_path)
            generator.manual_seed(seed)
            x2, y2 = get_batch(split, 64, 4, "cpu", generator, data_dir=tmp_path)
            assert torch.equal(x, x2) and torch.equal(y, y2)
            assert x.shape == y.shape == (64, 4)
            assert x.dtype == y.dtype == torch.long
            assert torch.equal(x[:, 1:], y[:, :-1])
            assert torch.equal(x + 1, y)
            assert x.min() >= low and y.max() < low + 20
            assert x[:, 0].unique().numel() > 1
        # Exactly one legal window exercises the very last token in the split.
        x, y = get_batch(split, 1, 19, "cpu", data_dir=tmp_path)
        assert x[0, 0].item() == low and y[0, -1].item() == low + 19
        with pytest.raises(ValueError, match="at least"):
            get_batch(split, 1, 20, "cpu", data_dir=tmp_path)
    assert set(opened) == {tmp_path / "train.bin", tmp_path / "val.bin"}


def test_seed_and_device() -> None:
    def draws() -> tuple[float, float, float]:
        return random.random(), float(np.random.random()), torch.rand(()).item()

    seed_everything(76)
    first = draws()
    seed_everything(76)
    assert draws() == first
    assert default_device().type == ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_batch(tmp_path: Path) -> None:
    np.arange(16, dtype=TOKEN_DTYPE).tofile(tmp_path / "train.bin")
    x, y = get_batch("train", 3, 4, data_dir=tmp_path)
    assert x.is_cuda and y.is_cuda
    assert torch.equal(x + 1, y)
