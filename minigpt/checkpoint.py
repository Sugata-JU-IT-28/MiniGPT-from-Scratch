"""Atomically persist training state and the random streams needed for replay."""

import os
import random
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch

from minigpt.io import replace_file


@contextmanager
def run_writer(directory: Path) -> Iterator[None]:
    """Keep one CLI writer per run. The OS releases the lock even on process termination."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"Could not acquire the training writer lock for {directory}"
            ) from exc
        yield


def rng_state(generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    """Capture global RNGs and the independent generator used for training offsets."""
    algorithm, keys, offset, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (algorithm, keys.tolist(), offset, has_gauss, cached),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "batch_generator": generator.get_state(),
    }


def restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    """Restore states after model construction and optimizer loading."""
    random.setstate(state["python"])
    alg, keys, offset, has_gauss, cached = state["numpy"]
    np.random.set_state((alg, np.asarray(keys, dtype=np.uint32), offset, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["batch_generator"])


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Publish a complete checkpoint with a same-volume atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            torch.save(state, stream)
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_checkpoint(source: Path, destination: Path) -> None:
    """Atomically copy a verified candidate to its best-checkpoint path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(dir=destination.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream, source.open("rb") as src:
            shutil.copyfileobj(src, stream)
        replace_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load tensors and basic Python data onto CPU, including CPU RNG states."""
    return torch.load(path, map_location="cpu", weights_only=True)
