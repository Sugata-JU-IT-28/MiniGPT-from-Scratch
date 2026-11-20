"""Time, train and resume controlled TinyShakespeare experiment suites."""

import argparse
from collections.abc import Sequence

from minigpt.data import seed_everything
from minigpt.experiment_runner import execute
from minigpt.experiment_timing import make_plan
from minigpt.experiments import QUICK_TOKENS, SEEDS, SUITES
from minigpt.io import display_metric


def main(argv: Sequence[str] | None = None) -> None:
    """Print the timing-based budget before the first training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=tuple(SUITES), help="Default: all suites, LR sweep first"
    )
    parser.add_argument(
        "--seeds", type=int, help="Number of successive seeds for the selected suite"
    )
    parser.add_argument("--seed", type=int, default=0, help="First training seed")
    parser.add_argument(
        "--quick", action="store_true", help="50 updates; separate results/quick tree"
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="Measure costs and print the budget"
    )
    parser.add_argument("--only", help="Run one named variant in the selected suite")
    args = parser.parse_args(argv)
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.only and not args.suite:
        parser.error("--only requires --suite")
    seed_everything(args.seed)
    if args.quick and not args.plan_only:
        plan = {"token_budget": QUICK_TOKENS, "seeds": SEEDS}
        print(f"Quick mode, {QUICK_TOKENS:,} tokens per run.", flush=True)
    else:
        # Timing uses a fixed seed so selecting additional training seeds reuses the same plan.
        plan = make_plan(0)
        hours = display_metric(plan["total_estimated_seconds"] / 3600, 2)
        print(
            f"Measured timing estimate {hours} hours including the final fit. "
            f"{plan['token_budget']:,} tokens/run, seeds {plan['seeds']}.",
            flush=True,
        )
    if not args.plan_only:
        execute(
            [args.suite] if args.suite else list(SUITES),
            plan,
            args.seed,
            args.seeds,
            quick=args.quick,
            only=args.only,
        )


if __name__ == "__main__":
    main()
