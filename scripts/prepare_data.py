"""Prepare data files and record measured split statistics."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from minigpt.corpus import read_corpus
from minigpt.data import prepare_dataset, seed_everything
from minigpt.io import write_json


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare one vocabulary and copy its measurements to the results directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    out_dir = args.out_dir or Path(f"data/vocab_{args.vocab_size}")
    meta = prepare_dataset(read_corpus(args.input), args.vocab_size, out_dir)
    write_json(
        args.results_dir / f"prepare_vocab_{args.vocab_size}.json",
        {
            "seed": args.seed,
            **meta,
        },
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
