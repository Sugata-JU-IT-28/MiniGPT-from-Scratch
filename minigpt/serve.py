"""Serve the local MiniGPT glass-box interface."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from minigpt.api import create_app
from minigpt.data import default_device, seed_everything
from minigpt.generate import load_model


def main(argv: Sequence[str] | None = None) -> None:
    """Restore the checkpoint's own architecture and matching tokenizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=os.getenv("MINIGPT_CHECKPOINT"))
    parser.add_argument("--tokenizer", type=Path, help="Override the tokenizer path in the config")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    model, tokenizer = (
        load_model(args.checkpoint, args.tokenizer) if args.checkpoint else (None, None)
    )
    app = create_app(model, tokenizer, default_device(), checkpoint=args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
