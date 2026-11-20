"""Aggregate completed, matched-budget runs and draw the Phase 3 experiment figures."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from minigpt.data import seed_everything
from minigpt.experiment_runner import read_json
from minigpt.experiments import QUICK_TOKENS, SEEDS, SUITES, Variant, run_config
from minigpt.io import display_metric, write_bytes, write_json

TITLES = {
    "lr_sweep": "Learning rate",
    "arch_ablation": "Architecture",
    "shape_ablation": "Shape",
    "scaling": "Parameter scaling",
    "tokenizer": "Tokenizer",
}
COLORS = ("#154b74", "#ba5327", "#167764", "#76569b", "#98566c", "#817322")


def measured(values: Sequence[float]) -> dict[str, float | None]:
    """Use sample SD for replication and leave single-seed variation unknown."""
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def group_runs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-seed endpoints and matched evaluation boundaries."""
    first = records[0]
    steps = [row["step"] for row in first["history"]]
    for record in records:
        if (
            [r["step"] for r in record["history"]] != steps
            or record["params"] != first["params"]
            or record["data_sha256"] != first["data_sha256"]
        ):
            raise ValueError("Replicates disagree on evaluation steps, parameters or input data")
    metrics = {
        key: measured([r["final"][key] for r in records])
        for key in ("train_loss", "val_loss", "val_bpb", "val_ppl")
    }
    gaps = measured([r["final"]["val_loss"] - r["final"]["train_loss"] for r in records])
    curve = [
        {
            "step": step,
            "tokens_seen": first["history"][i]["tokens_seen"],
            **{
                key: measured([r["history"][i][key] for r in records])
                for key in ("train_loss", "val_loss", "val_bpb")
            },
        }
        for i, step in enumerate(steps)
    ]
    return {
        "name": first["name"],
        "preset": first["config"]["preset"],
        "model": first["config"]["model"],
        "lr": first["config"]["train"]["lr"],
        "seeds": [r["seed"] for r in records],
        "params": first["params"],
        "non_embedding_params": first["non_embedding_params"],
        "endpoints": {**metrics, "gap": gaps},
        "endpoint_by_seed": {str(r["seed"]): r["final"] for r in records},
        "lowest_validation_step_by_seed": {
            str(r["seed"]): min(r["history"], key=lambda row: row["val_loss"])["step"]
            for r in records
        },
        "curve": curve,
        "elapsed_seconds": sum(r["elapsed"] for r in records),
    }


def contrast(reference: dict[str, Any], variant: dict[str, Any], key: str) -> dict[str, Any]:
    """Compare an endpoint difference against the larger of the two seed SDs."""
    left, right = (g["endpoints"][key] for g in (reference, variant))
    delta = left["mean"] - right["mean"]
    spread = max(left["std"], right["std"]) if None not in (left["std"], right["std"]) else None
    verdict = "inconclusive"
    if spread is not None and abs(delta) > spread:
        verdict = "lower" if delta > 0 else "higher"
    return {
        "reference": reference["name"],
        "variant": variant["name"],
        "metric": key,
        "reference_minus_variant": delta,
        "larger_sample_std": spread,
        "verdict": verdict,
    }


def collect(root: Path, first_seed: int, *, quick: bool = False) -> dict[str, Any]:
    """Refuse incomplete grids or unmatched configurations before publishing tables."""
    base = root / ("results/quick" if quick else "results")
    plan = (
        {"token_budget": QUICK_TOKENS, "seeds": SEEDS}
        if quick
        else read_json(base / "experiment_plan.json")
    )
    selection = read_json(root / "results/lr_selection.json")
    budget = plan["token_budget"]
    if not quick and selection["token_budget"] != budget:
        raise ValueError("The LR selection and experiment plan have different token budgets")
    suites = {}
    total_seconds = 0.0
    count = 0
    epochs = {}
    for suite, specs in SUITES.items():
        groups = {}
        for spec in specs:
            records = []
            for seed in range(first_seed, first_seed + plan["seeds"][suite]):
                cfg = run_config(spec, seed, budget, selection["shared_lr"], quick=quick)
                if cfg.data_dir not in epochs:
                    train_tokens = read_json(root / cfg.data_dir / "meta.json")["train_tokens"]
                    epochs[cfg.data_dir] = {
                        "train_tokens": train_tokens,
                        "epochs": budget / train_tokens,
                    }
                path = base / "runs" / suite / f"{cfg.name}.json"
                record = read_json(path)
                with path.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
                    csv_rows = list(csv.DictReader(stream))
                if (
                    record["mode"] != ("quick" if quick else "full")
                    or record["tokens_seen"] != budget
                    or record["final"]["tokens_seen"] != budget
                    or record["step"] != cfg.steps
                    or record["config"] != asdict(cfg)
                    or csv_rows
                    != [{k: str(v) for k, v in row.items()} for row in record["history"]]
                    or record["final"] != record["history"][-1]
                    or (record["suite"], record["name"], record["seed"]) != (suite, spec.name, seed)
                ):
                    raise ValueError(f"Inconsistent endpoint or run identity in {path}")
                records.append(record)
            group = group_runs(records)
            groups[spec.name] = group
            count += len(records)
            total_seconds += group["elapsed_seconds"]
        suites[suite] = groups
    comparisons = {
        s: [contrast(suites[s][ref], g, "val_loss") for n, g in suites[s].items() if n != ref]
        for s, ref in (("arch_ablation", "baseline"), ("shape_ablation", "base"))
    }
    for suite, key in (("scaling", "val_loss"), ("tokenizer", "val_bpb")):
        groups_in_order = list(suites[suite].values())
        comparisons[suite] = [
            contrast(a, b, key)
            for a, b in zip(groups_in_order[:-1], groups_in_order[1:], strict=True)
        ]
    return {
        "seed": first_seed,
        "run_count": count,
        "train_split_equivalents": epochs,
        "token_budget": budget,
        "lr_selection": selection,
        "seeds": plan["seeds"],
        "elapsed_seconds": total_seconds,
        "suites": suites,
        "comparisons": comparisons,
    }


def label(spec: Variant) -> str:
    """Use familiar names for single-factor model changes."""
    return {
        "baseline": "Baseline",
        "rmsnorm": "+ RMSNorm",
        "rope": "+ RoPE",
        "swiglu": "+ SwiGLU",
        "modern": "Modern",
        "base": "Base",
    }.get(spec.name, spec.name.replace("_", " "))


def show(stat: dict[str, float | None]) -> str:
    """Print mean and sample SD without implying precision from a single seed."""
    avg = display_metric(stat["mean"])
    return avg if stat["std"] is None else f"{avg} ± {display_metric(stat['std'])}"


def tables(summary: dict[str, Any]) -> str:
    """Render only measurements present in the machine-readable summary."""
    rows = [
        "# TinyShakespeare experiment results",
        "",
        "<!-- Generated by scripts/aggregate_results.py. Do not edit by hand. -->",
        "",
        f"Each of the {summary['run_count']} runs used {summary['token_budget']:,} training "
        "tokens. Entries show endpoint means and sample standard deviations across seeds. "
        "A single seed has no sample standard deviation. Displayed decimals are truncated. "
        "[Full measurements](../results/experiment_aggregate.json)",
        "",
        "[Learning-rate sweep](img/lr_sweep.png) · [Architecture](img/arch_ablation.png) · "
        "[Shape](img/shape_ablation.png) · [Scaling](img/scaling.png) · "
        "[Tokenizer](img/tokenizer_bpb.png) · "
        "[Learning curves](img/experiment_learning_curves.png)",
        "",
    ]
    for suite, groups in summary["suites"].items():
        rows += [
            f"## {TITLES[suite]}",
            "",
            "| Variant | Seeds | Total params | Non-embedding params | Validation bits/byte |"
            if suite == "tokenizer"
            else "| Variant | Seeds | Total params | Non-embedding params | Train loss | "
            "Validation loss | Validation − train |",
            "|---|---:|---:|---:|---:|"
            if suite == "tokenizer"
            else "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for spec in SUITES[suite]:
            g = groups[spec.name]
            keys = ("val_bpb",) if suite == "tokenizer" else ("train_loss", "val_loss", "gap")
            numbers = " | ".join(show(g["endpoints"][k]) for k in keys)
            rows.append(
                f"| {label(spec)} | {len(g['seeds'])} | {g['params']:,} | "
                f"{g['non_embedding_params']:,} | {numbers} |"
            )
        rows.append("")
    rows += [
        "Losses are in nats per predicted token. Tokenizers use validation bits per source "
        "byte. The shape grids share their base observation at each seed. Curves use all "
        "recorded full-split evaluations. [Interpretation](analysis.md)",
        "",
    ]
    return "\n".join(rows)


def figure(size: tuple[float, float]) -> Figure:
    """Use a display-independent plotting canvas."""
    fig = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def style(ax: Axes, ylabel: str, xlabel: str = "") -> None:
    """Keep labels and measurement units visible in each exported panel."""
    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)


def bars(ax: Axes, groups: list[dict[str, Any]], names: list[str], key: str, color: str) -> None:
    """Show means with one sample SD as the error-bar length."""
    data = [g["endpoints"][key] for g in groups]
    errors = [m["std"] for m in data]
    ax.bar(
        names,
        [m["mean"] for m in data],
        yerr=errors if all(e is not None for e in errors) else None,
        capsize=4,
        color=color,
        width=0.65,
    )
    ax.tick_params(axis="x", labelsize=9)


def figures(summary: dict[str, Any], directory: Path) -> None:
    """Write architecture, shape, LR, scaling, tokenizer, and training curves."""
    directory.mkdir(parents=True, exist_ok=True)
    groups = summary["suites"]
    arch = groups["arch_ablation"]
    fig = figure((7.8, 4.1))
    ax = fig.subplots()
    arch_specs = SUITES["arch_ablation"]
    bars(
        ax,
        [arch[s.name] for s in arch_specs],
        [label(s) for s in arch_specs],
        "val_loss",
        COLORS[0],
    )
    style(ax, "Validation loss (nats/token)")
    ax.set_title("Architecture at equal training tokens · mean ± sample SD")
    fig.savefig(directory / "arch_ablation.png", dpi=150)
    shape = groups["shape_ablation"]
    fig = figure((9.3, 4.3))
    axes = fig.subplots(1, 3, sharey=True)
    for ax, (title, names, values) in zip(
        axes,
        [
            ("Layers", ["layers_2", "base", "layers_6"], ["2", "4", "6"]),
            ("Attention heads", ["heads_2", "base", "heads_8"], ["2", "4", "8"]),
            ("Context tokens", ["context_64", "base", "context_256"], ["64", "128", "256"]),
        ],
        strict=True,
    ):
        bars(ax, [shape[n] for n in names], values, "val_loss", COLORS[2])
        style(ax, "Validation loss (nats/token)" if ax == axes[0] else "", title)
    fig.suptitle("One-factor shape ablations · mean ± sample SD")
    fig.savefig(directory / "shape_ablation.png", dpi=150)
    fig = figure((10.5, 4.2))
    ax, trace = fig.subplots(1, 2)
    for i, preset in enumerate(("baseline", "modern")):
        points = sorted(
            (g for g in groups["lr_sweep"].values() if g["preset"] == preset), key=lambda g: g["lr"]
        )
        ax.plot(
            [g["lr"] for g in points],
            [g["endpoints"]["val_loss"]["mean"] for g in points],
            "o-",
            color=COLORS[i],
            label=preset.capitalize(),
        )
        for j, g in enumerate(points):
            trace.plot(
                [s["tokens_seen"] for s in g["curve"]],
                [s["val_loss"]["mean"] for s in g["curve"]],
                "--" if i == 0 else "-",
                color=COLORS[j],
                label=f"{preset} {g['lr']:g}",
            )
    ax.set_xscale("log")
    ax.set_xticks([g["lr"] for g in points], labels=[f"{g['lr']:g}" for g in points])
    ax.minorticks_off()
    style(ax, "Endpoint validation loss (nats/token)", "Peak learning rate")
    style(trace, "Validation loss (nats/token)", "Training tokens")
    ax.legend(frameon=False)
    trace.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Learning-rate sweep")
    fig.savefig(directory / "lr_sweep.png", dpi=150)
    scale = list(groups["scaling"].values())
    fig = figure((9.0, 4.1))
    for ax, (field, title) in zip(
        fig.subplots(1, 2, sharey=True),
        [
            ("params", "All trainable parameters"),
            ("non_embedding_params", "Non-embedding parameters"),
        ],
        strict=True,
    ):
        ax.errorbar(
            [g[field] for g in scale],
            [g["endpoints"]["val_loss"]["mean"] for g in scale],
            yerr=[g["endpoints"]["val_loss"]["std"] for g in scale],
            fmt="o-",
            capsize=4,
            color=COLORS[0],
        )
        ax.set_xscale("log")
        style(ax, "Validation loss (nats/token)" if field == "params" else "", title)
    fig.suptitle("Model size at equal training tokens · mean ± sample SD")
    fig.savefig(directory / "scaling.png", dpi=150)
    fig = figure((6.0, 4.0))
    ax = fig.subplots()
    vocabs = list(groups["tokenizer"].values())
    bars(ax, vocabs, [str(g["model"]["vocab_size"]) for g in vocabs], "val_bpb", COLORS[2])
    style(ax, "Validation bits per byte", "BPE vocabulary size")
    ax.set_title("Held-out text cost · mean ± sample SD")
    fig.savefig(directory / "tokenizer_bpb.png", dpi=150)
    fig = figure((9.0, 4.4))
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for ax, (key, title) in zip(
        axes, [("train_loss", "Train"), ("val_loss", "Validation")], strict=True
    ):
        for i, spec in enumerate(arch_specs):
            curve = arch[spec.name]["curve"]
            ax.plot(
                [r["tokens_seen"] for r in curve],
                [r[key]["mean"] for r in curve],
                "o-",
                markersize=3,
                label=label(spec),
                color=COLORS[i],
            )
        ax.set_title(title)
        style(ax, "Loss (nats/token)" if ax == axes[0] else "", "Training tokens")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Architecture learning curves · mean across seeds")
    fig.savefig(directory / "experiment_learning_curves.png", dpi=150)


def main(argv: Sequence[str] | None = None) -> None:
    """Write tables and figures from the full published experiment matrix."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0, help="First seed in each planned suite")
    p.add_argument("--quick", action="store_true", help="Read and publish only in results/quick")
    args = p.parse_args(argv)
    seed_everything(args.seed)
    root = Path(".")
    summary = collect(root, args.seed, quick=args.quick)
    base = root / ("results/quick" if args.quick else "results")
    docs = base / "report" if args.quick else root / "docs"
    write_json(base / "experiment_aggregate.json", summary)
    page = tables(summary)
    if args.quick:
        page = page.replace("../results/experiment_aggregate.json", "../experiment_aggregate.json")
        page = page.replace("[Interpretation](analysis.md)", "Quick pipeline validation")
        page = page.replace("# TinyShakespeare", "# Quick TinyShakespeare")
    write_bytes(docs / "results_tables.md", page.encode("utf-8"))
    figures(summary, docs / "img")
    print(f"Aggregated {summary['run_count']} runs.", flush=True)


if __name__ == "__main__":
    main()
