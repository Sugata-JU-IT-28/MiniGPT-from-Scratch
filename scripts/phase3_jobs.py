"""Run quick validation, then the four measured suites in sequence, with durable progress."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from minigpt.checkpoint import run_writer
from minigpt.data import seed_everything
from minigpt.experiment_runner import execute, read_json
from minigpt.experiments import QUICK_TOKENS, SEEDS, SUITES
from minigpt.io import write_json
from scripts import aggregate_results


def completeness(seed: int, plan: dict, quick: bool) -> dict[str, list[str]]:
    """Enumerate missing result files from the experiment design."""
    root = Path("results/quick/runs" if quick else "results/runs")
    return {
        suite: [
            f"{spec.name}_s{i}.json"
            for spec in specs
            for i in range(seed, seed + plan["seeds"][suite])
            if not (root / suite / f"{spec.name}_s{i}.json").is_file()
        ]
        for suite, specs in SUITES.items()
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Validate quick artifacts before starting any full downstream run."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    seed_everything(args.seed)
    plan = read_json(Path("results/experiment_plan.json"))
    suites = [s for s in SUITES if s != "lr_sweep"]
    with run_writer(Path("logs/phase3_supervisor")):
        quick_plan = {"token_budget": QUICK_TOKENS, "seeds": SEEDS}
        execute(list(SUITES), quick_plan, args.seed, None, quick=True)
        if any(completeness(args.seed, quick_plan, True).values()):
            raise RuntimeError("Quick results are incomplete")
        aggregate_results.main(["--quick", "--seed", str(args.seed)])
        quick_report = Path("results/quick/report")
        if (
            not (quick_report / "results_tables.md").is_file()
            or len(list((quick_report / "img").glob("*.png"))) != 6
        ):
            raise RuntimeError("Quick report failed to produce its table and six plots")
        print("Quick validation complete. Beginning the measured suites.", flush=True)
        for suite in suites:
            execute([suite], plan, args.seed, None, quick=False)
            if suite == "arch_ablation":
                elapsed = sum(
                    read_json(p)["elapsed"]
                    for p in Path("results/runs/arch_ablation").glob("*.json")
                )
                reference = plan["estimated_seconds"][suite]
                remaining = sum(plan["estimated_seconds"][s] for s in suites[1:])
                estimate = {
                    "observed_arch_seconds": elapsed,
                    "planned_arch_seconds": reference,
                    "remaining_suites_seconds": remaining * elapsed / reference,
                }
                write_json(Path("results/phase3_remaining_time.json"), estimate)
                print(f"Architecture timing update {estimate}", flush=True)
        missing = completeness(args.seed, plan, False)
        write_json(
            Path("results/phase3_completeness.json"), {"expected": plan["runs"], "missing": missing}
        )
        if any(missing.values()):
            raise RuntimeError(f"Missing required results {missing}")
        aggregate_results.main(["--seed", str(args.seed)])


if __name__ == "__main__":
    main()
