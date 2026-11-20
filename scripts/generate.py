"""Generate text from a trained MiniGPT checkpoint, optionally streaming UTF-8 fragments."""

import argparse
import codecs
import sys
from collections.abc import Sequence
from pathlib import Path

from minigpt.data import seed_everything
from minigpt.generate import generate_stream, load_model
from minigpt.tokenizer.bpe import EOT_ID


def main(argv: Sequence[str] | None = None) -> None:
    """Print a prompt followed by its continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--eos-id", type=int, default=EOT_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--window-policy", choices=("error", "reprefill"), default="error")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    model, tokenizer = load_model(args.checkpoint, args.tokenizer)
    steps = generate_stream(
        model,
        tokenizer.encode(args.prompt),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_id=args.eos_id,
        seed=args.seed,
        use_cache=not args.no_cache,
        window_policy=args.window_policy,
    )
    if args.stream:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        print(args.prompt, end="", flush=True)
        for step in steps:
            fragment = decoder.decode(tokenizer.decode_bytes([step.token_id]))
            print(fragment, end="", flush=True)
        print(decoder.decode(b"", final=True))
    else:
        sys.stdout.write(args.prompt + tokenizer.decode(step.token_id for step in steps) + "\n")


if __name__ == "__main__":
    main()
