"""Prepare contiguous splits and sample shifted language-model batches from disk."""

import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypedDict

import numpy as np
import torch

from minigpt.corpus import split_text, text_digest
from minigpt.io import write_json
from minigpt.tokenizer import BPETokenizer
from minigpt.tokenizer.bpe import BASE_SIZE

DEFAULT_DIR = Path("data/vocab_1024")
TOKEN_DTYPE = np.dtype("<u2")


class DatasetMeta(TypedDict):
    """Describe the corpus, binary layout, and validation compression."""

    corpus_sha256: str
    vocab_size: int
    dtype: str
    train_tokens: int
    val_tokens: int
    train_bytes: int
    val_bytes: int
    bytes_per_token: float


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def default_device() -> torch.device:
    """Choose CUDA when available and CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset(text: str, vocab_size: int, out_dir: Path) -> DatasetMeta:
    """Train only on the raw training prefix and write little-endian uint16 token IDs."""
    if not BASE_SIZE <= vocab_size <= 65535:
        raise ValueError(f"uint16 dataset vocabulary must be between {BASE_SIZE} and 65535")
    train, val = split_text(text)
    if not train or not val:
        raise ValueError("The corpus needs non-empty training and validation splits")
    tokenizer = BPETokenizer().train(train, vocab_size)
    train_ids = tokenizer.encode(train)
    val_ids = tokenizer.encode(val)
    val_bytes = len(val.encode("utf-8"))
    meta = DatasetMeta(
        corpus_sha256=text_digest(text),
        vocab_size=tokenizer.vocab_size,
        dtype=TOKEN_DTYPE.str,
        train_tokens=len(train_ids),
        val_tokens=len(val_ids),
        train_bytes=len(train.encode("utf-8")),
        val_bytes=val_bytes,
        bytes_per_token=val_bytes / len(val_ids),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out_dir) as temp:
        staging = Path(temp)
        for name, ids in (("train", train_ids), ("val", val_ids)):
            np.asarray(ids, dtype=TOKEN_DTYPE).tofile(staging / f"{name}.bin")
        tokenizer.save(staging / "tokenizer.json")
        write_json(staging / "meta.json", meta)
        for name in ("train.bin", "val.bin", "tokenizer.json", "meta.json"):
            (staging / name).replace(out_dir / name)
    return meta


def get_batch(
    split: Literal["train", "val"],
    batch_size: int,
    block_size: int,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    *,
    data_dir: str | Path = DEFAULT_DIR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample uniform valid starts and return contiguous int64 inputs and next-token targets."""
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    if batch_size <= 0 or block_size <= 0:
        raise ValueError("batch_size and block_size must be positive")
    mapped = np.memmap(Path(data_dir) / f"{split}.bin", mode="r", dtype=TOKEN_DTYPE)
    num_starts = len(mapped) - block_size
    if num_starts <= 0:
        raise ValueError(f"{split} needs at least {block_size + 1} tokens")
    offsets = (
        torch.randint(
            num_starts,
            (batch_size,),
            generator=generator,
            device=generator.device if generator is not None else "cpu",
        )
        .cpu()
        .numpy()
    )
    indices = offsets[:, None] + np.arange(block_size + 1)
    windows = torch.from_numpy(mapped[indices].astype(np.int64))
    destination = default_device() if device is None else torch.device(device)
    if destination.type == "cuda":
        windows = windows.pin_memory()
    windows = windows.to(destination, non_blocking=destination.type == "cuda")
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()
