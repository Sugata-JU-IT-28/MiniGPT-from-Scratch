"""Verify two completed fits against held-out baselines and promote the best checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minigpt.checkpoint import copy_checkpoint, load_checkpoint
from minigpt.config import load_config
from minigpt.data import TOKEN_DTYPE, default_device, seed_everything
from minigpt.io import file_sha256, write_bytes, write_json
from minigpt.metrics import evaluate
from minigpt.model import GPT, GPTConfig


def sanity_baselines(path: Path, vocab_size: int) -> dict[str, Any]:
    """Fit the unsmoothed unigram on train, then score all val IDs and the aligned LM targets."""
    train = np.memmap(path / "train.bin", dtype=TOKEN_DTYPE, mode="r")
    val = np.memmap(path / "val.bin", dtype=TOKEN_DTYPE, mode="r")
    counts = np.bincount(train, minlength=vocab_size)

    def unigram(ids: np.ndarray) -> float | None:
        frequencies = counts[ids]
        if np.any(frequencies == 0):
            return None  # A zero-probability target gives infinite cross-entropy.
        return float(-np.log(frequencies / train.size).mean())

    return {
        "uniform_loss": math.log(vocab_size),
        "unigram_loss": unigram(val[1:]),
        "unigram_all_val_ids_loss": unigram(val),
        "unseen_val_targets": int(np.count_nonzero(counts[val[1:]] == 0)),
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "val_targets": int(val.size - 1),
    }


def parameter_counts(config_path: Path) -> dict[str, dict[str, int]]:
    """Count actual model parameters without allocating weight storage."""
    result = {}
    for preset in ("baseline", "modern"):
        cfg = load_config(config_path, preset=preset)
        with torch.device("meta"):
            model = GPT(cfg.model)
        result[preset] = {"total": model.count_params(), "non_embedding": model.count_params(True)}
    return result


def read_run(name: str) -> dict[str, Any]:
    """Cross-check the complete CSV against its best weights and a new validation sweep."""
    run_dir = Path("results/runs") / name
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = [
            {key: int(v) if key in ("step", "tokens_seen") else float(v) for key, v in row.items()}
            for row in csv.DictReader(stream)
        ]
    last = rows[-1]
    if run["step"] != run["steps_planned"] or last["step"] != run["step"]:
        raise ValueError(f"{name} has not completed its full budget")
    best = min(rows, key=lambda r: r["val_loss"])
    path = Path("checkpoints") / name / "best.pt"
    state = load_checkpoint(path)
    cfg = run["config"]
    if (
        state["config"] != cfg
        or state["step"] != best["step"]
        or state["best_val"] != best["val_loss"]
    ):
        raise ValueError(f"{name} checkpoint and measured minimum disagree")
    device = default_device()
    if device.type == "cpu" and cfg["train"]["cpu_threads"] is not None:
        torch.set_num_threads(cfg["train"]["cpu_threads"])
    model = GPT(GPTConfig(**cfg["model"])).to(device)
    model.load_state_dict(state["model"])
    val = evaluate(
        model, "val", data_dir=Path(cfg["data_dir"]), batch_size=cfg["train"]["eval_batch_size"]
    )
    if not math.isclose(val.loss, best["val_loss"], abs_tol=1e-6, rel_tol=0.0):
        raise ValueError(f"{name} loaded weights did not reproduce the recorded validation loss")
    divergences = [
        [left["step"], right["step"]]
        for left, right in zip(rows[:-1], rows[1:], strict=True)
        if right["train_loss"] < left["train_loss"] and right["val_loss"] > left["val_loss"]
    ]
    return {
        "name": name,
        "preset": cfg["preset"],
        "config": cfg,
        "data_sha256": run["data_sha256"],
        "runtime": run["runtime"],
        "params": model.count_params(),
        "tokens_seen": run["tokens_seen"],
        "elapsed": run["elapsed"],
        "best": best,
        "final": last,
        "best_gap": best["val_loss"] - best["train_loss"],
        "final_gap": last["val_loss"] - last["train_loss"],
        "rechecked_val": asdict(val),
        "falling_train_rising_val_intervals": divergences,
        "best_checkpoint": path.as_posix(),
    }


def _display(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def write_comparison(summary: dict[str, Any]) -> None:
    """Render the quantitative report directly from recorded results."""
    sanity = summary["sanity_baselines"]
    unigram = "infinite" if sanity["unigram_loss"] is None else _display(sanity["unigram_loss"])
    lines = [
        "# Baseline and modern MiniGPT on TinyShakespeare",
        "",
        f"Both fits used seed {summary['training_seed']} and scored "
        f"{summary['tokens_per_run']:,} training targets in "
        f"{summary['runs'][0]['final']['step']:,} optimizer updates each. "
        "Both curves use full train and validation sweeps "
        "with dropout disabled. "
        "Each next-token target is scored once, including the final partial window.",
        "",
        f"The uniform reference scored {_display(sanity['uniform_loss'])} nats/token. "
        f"The empirical train unigram scored {unigram} on the same {sanity['val_targets']:,} "
        "validation targets. Gaps below are validation loss minus training loss. "
        "Elapsed minutes include the evaluations. "
        "[Machine-readable comparison](../results/phase2_summary.json)",
        "",
        "| Preset | Best step | Train loss there | Best val loss | Val PPL | Val BPB "
        "| Final gap | Elapsed min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        best = run["best"]
        lines.append(
            f"| {run['preset']} | {best['step']:,} | {_display(best['train_loss'])} "
            f"| {_display(best['val_loss'])} | {_display(best['val_ppl'])} "
            f"| {_display(best['val_bpb'])} | {_display(run['final_gap'])} "
            f"| {_display(run['elapsed'] / 60)} |"
        )
    winner = summary["selected"]
    lines.extend(
        [
            "",
            f"The {winner['preset']} checkpoint had the lower measured validation loss "
            f"by {_display(summary['best_loss_difference'])} nats/token. "
            "Selection used the minimum across recorded evaluations for each run. "
            "[Matched temperature samples](samples_phase2.md)",
            "",
            "## Learning curves",
            "",
        ]
    )
    for run in summary["runs"]:
        name, preset = run["name"], run["preset"]
        lines.extend(
            [
                f"### {preset.title()}",
                "",
                f"At the final step, training loss was {_display(run['final']['train_loss'])} and "
                f"validation loss was {_display(run['final']['val_loss'])}. "
                f"Training loss fell while validation loss rose in "
                f"{len(run['falling_train_rising_val_intervals'])} measured intervals. "
                f"[Full curve data](../results/runs/{name}/metrics.csv)",
                "",
                f"![{preset.title()} train and validation cross-entropy](img/{name}_loss.png)",
                "",
            ]
        )
    lines += ["## Parameter counts", "", "| Configuration | Baseline | Modern |", "|---|---:|---:|"]
    for config, counts in summary["parameter_counts"].items():
        lines.append(
            f"| {config} | {counts['baseline']['total']:,} | {counts['modern']['total']:,} |"
        )
    lines += [
        "",
        "## Reproduce from the project environment",
        "",
        "```sh",
        *summary["commands"].values(),
        "```",
        "",
        "Add `--resume` to continue the matching run from its last saved update.",
        "",
    ]
    write_bytes(Path("docs/training_phase2.md"), "\n".join(lines).encode())


def main(argv: Sequence[str] | None = None) -> None:
    """Require complete matched budgets and both baseline wins before selecting weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-config", type=Path, default=Path("configs/cpu_tiny.yaml"))
    parser.add_argument("--gpu-config", type=Path, default=Path("configs/gpu_small.yaml"))
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    presets = ("baseline", "modern")
    configs = [load_config(args.cpu_config, preset=p, seed=args.seed) for p in presets]
    runs = [read_run(c.name) for c in configs]
    for cfg, run in zip(configs, runs, strict=True):
        if asdict(cfg) != run["config"] or run["tokens_seen"] != cfg.steps * cfg.tokens_per_step:
            raise ValueError("Run must match the resolved config, seed, and full token budget")
    if runs[0]["data_sha256"] != runs[1]["data_sha256"]:
        raise ValueError("The presets must use the same prepared dataset")
    data_dir = Path(configs[0].data_dir)
    for name, expected in runs[0]["data_sha256"].items():
        if file_sha256(data_dir / name) != expected:
            raise ValueError(f"Prepared {name} differs from the training input")
    sanity = sanity_baselines(data_dir, configs[0].model.vocab_size)
    limit = min(
        sanity["uniform_loss"],
        sanity["unigram_loss"] if sanity["unigram_loss"] is not None else math.inf,
    )
    if any(r["best"]["val_loss"] >= limit or r["final"]["val_loss"] >= limit for r in runs):
        raise ValueError("A trained model did not beat both validation sanity baselines")
    winner = min(runs, key=lambda r: r["best"]["val_loss"])
    source = Path(winner["best_checkpoint"])
    destination = Path("checkpoints/best.pt")
    copy_checkpoint(source, destination)
    checksum = file_sha256(source)
    if file_sha256(destination) != checksum:
        raise OSError("Promoted checkpoint checksum mismatch")
    summary = {
        "date": date.today().isoformat(),
        "training_seed": args.seed,
        "tokens_per_run": runs[0]["tokens_seen"],
        "sanity_baselines": sanity,
        "runs": runs,
        "parameter_counts": {
            p.stem: parameter_counts(p) for p in (args.cpu_config, args.gpu_config)
        },
        "selected": {
            "preset": winner["preset"],
            "step": winner["best"]["step"],
            "val_loss": winner["best"]["val_loss"],
            "reason": "Lowest full validation loss",
            "source": source.as_posix(),
            "path": destination.as_posix(),
            "sha256": checksum,
        },
        "best_loss_difference": abs(runs[0]["best"]["val_loss"] - runs[1]["best"]["val_loss"]),
        "commands": {
            p: f"python scripts/train.py --config {args.cpu_config.as_posix()} "
            f"--preset {p} --seed {args.seed}"
            for p in presets
        },
    }
    write_json(Path("results/phase2_summary.json"), summary)
    write_comparison(summary)
    print(json.dumps(summary["selected"], indent=2))


if __name__ == "__main__":
    main()
