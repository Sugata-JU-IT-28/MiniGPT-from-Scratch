"""Run, resume and publish matched-budget experiment records."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minigpt.checkpoint import run_writer
from minigpt.config import RunConfig
from minigpt.experiments import QUICK_TOKENS, SEEDS, SUITES, run_config, select_lr
from minigpt.io import write_json
from minigpt.metrics import write_metrics
from minigpt.train import Trainer


def read_json(path: Path) -> dict[str, Any]:
    """Read one published UTF-8 result."""
    return json.loads(path.read_text(encoding="utf-8"))


def result_dir(root: Path, quick: bool) -> Path:
    """Keep pipeline-validation runs out of the experiment tables."""
    return root / ("results/quick" if quick else "results")


def check_dir(root: Path, suite: str, cfg: RunConfig, quick: bool) -> Path:
    """Address one resumable optimizer and its historical best weights."""
    base = root / "checkpoints/experiments"
    return (base / "quick" if quick else base) / suite / cfg.name


def run_one(
    root: Path, suite: str, name: str, cfg: RunConfig, *, quick: bool = False
) -> dict[str, Any]:
    """Restore in-flight state, or reuse a fully published run and its exact CSV history."""
    mode = "quick" if quick else "full"
    directory = result_dir(root, quick) / "runs" / suite
    record = directory / f"{cfg.name}.json"
    log = directory / f"{cfg.name}.csv"
    checkpoints = check_dir(root, suite, cfg, quick)
    with run_writer(checkpoints):
        if record.exists():
            result = read_json(record)
            if (
                result["config"] != asdict(cfg)
                or result["tokens_seen"] != cfg.train.token_budget
                or (result["suite"], result["name"], result["mode"]) != (suite, name, mode)
            ):
                raise ValueError(f"Existing record does not match the requested run: {record}")
            if not log.is_file():
                write_metrics(log, result["history"])
            print(f"SKIP {record}", flush=True)
            return result
        trainer = Trainer(cfg, root, metrics_path=log, checkpoint_dir=checkpoints)
        if trainer.last.is_file():
            trainer.resume()
        summary = trainer.run(export_report=False)
        if summary["tokens_seen"] != cfg.train.token_budget:
            raise ValueError(
                "Completed optimizer updates did not consume the declared token budget"
            )
        result = {
            **summary,
            "suite": suite,
            "name": name,
            "seed": cfg.seed,
            "mode": mode,
            "history": trainer.rows,
            "final": trainer.rows[-1],
        }
        write_json(record, result)
        print(f"DONE {record}", flush=True)
        return result


def execute(
    suites: list[str],
    plan: dict[str, Any],
    first_seed: int,
    seed_count: int | None,
    *,
    quick: bool,
    only: str | None = None,
    root: Path = Path("."),
) -> None:
    """Execute declarative suites and freeze the LR decision before dependent fits."""
    directory = result_dir(root, quick)
    selection_path = directory / "lr_selection.json"
    budget = QUICK_TOKENS if quick else plan["token_budget"]
    for suite in suites:
        specs = [s for s in SUITES[suite] if only is None or s.name == only]
        if not specs:
            raise ValueError(f"No variant named {only!r} in {suite}")
        if suite == "lr_sweep":
            lr = 0.001  # Each sweep variant supplies its own rate.
        elif selection_path.is_file():
            selection = read_json(selection_path)
            if selection["token_budget"] != budget:
                raise ValueError("The saved LR sweep used a different token budget")
            lr = selection["shared_lr"]
        elif quick:
            lr = 0.001
            print(
                "Quick pipeline check using the default LR without a completed quick sweep.",
                flush=True,
            )
        else:
            raise ValueError("Complete --suite lr_sweep before starting dependent suites")
        count = seed_count if seed_count is not None else plan.get("seeds", SEEDS)[suite]
        seeds = list(range(first_seed, first_seed + count))
        for spec in specs:
            for seed in seeds:
                cfg = run_config(spec, seed, budget, lr, quick=quick)
                run_one(root, suite, spec.name, cfg, quick=quick)
        if suite == "lr_sweep":
            paths = [
                directory / "runs" / suite / f"{s.name}_s{seed}.json"
                for s in SUITES[suite]
                for seed in seeds
            ]
            if all(p.is_file() for p in paths):
                records = [read_json(p) for p in paths]
                selection = {**select_lr(records), "seeds": seeds, "token_budget": budget}
                if (
                    selection_path.exists()
                    and read_json(selection_path)["shared_lr"] != selection["shared_lr"]
                ):
                    if any((directory / "runs" / s).is_dir() for s in SUITES if s != "lr_sweep"):
                        raise ValueError("New LR decision conflicts with published downstream runs")
                write_json(selection_path, selection)
                print(
                    f"Chosen LR {selection['shared_lr']}; per preset {selection['by_preset']}",
                    flush=True,
                )
