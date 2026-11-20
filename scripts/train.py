"""Train or resume one MiniGPT experiment using an explicit token budget."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from minigpt.checkpoint import run_writer
from minigpt.config import load_config
from minigpt.train import Trainer


def main(argv: Sequence[str] | None = None) -> None:
    """Resolve YAML and dotted overrides before creating the trainer and its seeded generators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preset", choices=("baseline", "modern"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--override", nargs="+", action="extend", default=[])
    parser.add_argument("--resume", action="store_true", help="Continue this run from last.pt")
    parser.add_argument("--stop-after", type=int, help="Pause after this optimizer step")
    args = parser.parse_args(argv)
    cfg = load_config(
        args.config, args.override, preset=args.preset, seed=args.seed, run_name=args.run_name
    )
    with run_writer(Path("checkpoints") / cfg.name):
        trainer = Trainer(cfg)
        if args.resume:
            trainer.resume()
        trainer.run(args.stop_after)


if __name__ == "__main__":
    main()
