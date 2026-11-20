"""Token-weighted evaluation and durable learning curves."""

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from minigpt.data import DEFAULT_DIR, TOKEN_DTYPE
from minigpt.io import write_bytes
from minigpt.model import GPT


@dataclass(frozen=True)
class Evaluation:
    """Mean cross-entropy in nats, perplexity, bits per byte, and scored targets."""

    loss: float
    ppl: float
    bpb: float
    tokens: int


@torch.inference_mode()
def evaluate(
    model: GPT,
    split: Literal["train", "val"],
    *,
    data_dir: Path = DEFAULT_DIR,
    batch_size: int = 16,
) -> Evaluation:
    """Score every predictable token once, including the final short window, in eval mode."""
    if split not in ("train", "val") or batch_size < 1:
        raise ValueError("Expected a train or val split and a positive batch size")
    ids = np.memmap(data_dir / f"{split}.bin", mode="r", dtype=TOKEN_DTYPE)
    count = len(ids) - 1
    if count < 1:
        raise ValueError("Evaluation requires at least two tokens")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    bytes_per_token = (
        meta["bytes_per_token"] if split == "val" else meta["train_bytes"] / meta["train_tokens"]
    )
    block = model.config.block_size
    device = model.token_embedding.weight.device
    nll = torch.zeros((), dtype=torch.float64, device=device)
    training = model.training
    model.eval()
    try:
        start = 0
        while start < count:
            # Batch only equal-length windows; the tail runs as one shorter window.
            width = min(block, count - start)
            windows = min(batch_size, (count - start) // width)
            size = windows * width
            stream = torch.from_numpy(ids[start : start + size + 1].astype(np.int64)).to(device)
            x, y = stream[:-1].view(windows, width), stream[1:].view(windows, width)
            _, loss = model(x, y)
            assert loss is not None
            nll += loss.double() * size
            start += size
    finally:
        model.train(training)
    mean = nll.item() / count
    return Evaluation(mean, math.exp(mean), mean / math.log(2) / bytes_per_token, count)


class MetricRow(TypedDict):
    """Synchronized full-split measurements at an optimizer update boundary."""

    step: int
    tokens_seen: int
    train_loss: float
    val_loss: float
    val_ppl: float
    val_bpb: float
    lr: float
    tokens_per_sec: float
    elapsed: float


def write_metrics(path: Path, rows: list[MetricRow]) -> None:
    """Regenerate the small CSV from checkpointed history, avoiding duplicate resume rows."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MetricRow.__annotations__, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_bytes(path, output.getvalue().encode("utf-8"))


def plot_losses(path: Path, name: str, rows: list[MetricRow]) -> None:
    """Plot train and validation cross-entropy against the actual token budget consumed."""
    figure = Figure(figsize=(7.2, 4.2), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    x = [r["tokens_seen"] for r in rows]
    for key, label, color in (
        ("train_loss", "Train", "#167260"),
        ("val_loss", "Validation", "#aa4e30"),
    ):
        ax.plot(x, [r[key] for r in rows], "o-", label=label, color=color)
    ax.set(title=name, xlabel="Training tokens seen", ylabel="Cross-entropy (nats/token)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
