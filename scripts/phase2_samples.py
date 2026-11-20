"""Reproduce the historical Phase 2 sample report using the shared sampler."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from minigpt.checkpoint import load_checkpoint
from minigpt.data import default_device, seed_everything
from minigpt.generate import generate
from minigpt.io import file_sha256, write_bytes, write_json
from minigpt.model import GPT, GPTConfig
from minigpt.tokenizer import BPETokenizer
from minigpt.tokenizer.bpe import EOT_ID

TEMPERATURES = (0.5, 0.8, 1.0)


@torch.inference_mode()
def sample(
    model: GPT, prompt: list[int], temperature: float, new_tokens: int, seed: int
) -> list[int]:
    """Recompute the current context for each draw and return only generated IDs."""
    model.eval()
    return generate(
        model,
        prompt,
        max_new_tokens=new_tokens,
        temperature=temperature,
        seed=seed,
        eos_id=EOT_ID,
        use_cache=False,
        window_policy="reprefill",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Save generated text, exact token IDs, and the settings shared between presets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--new-tokens", type=int, default=160)
    parser.add_argument("--runs", nargs=2, default=["cpu_tiny_baseline", "cpu_tiny_modern"])
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    prompt = "\n"
    samples: list[dict[str, Any]] = []
    for name in args.runs:
        path = Path("checkpoints") / name / "best.pt"
        state = load_checkpoint(path)
        config = state["config"]
        device = default_device()
        if device.type == "cpu" and config["train"]["cpu_threads"] is not None:
            torch.set_num_threads(config["train"]["cpu_threads"])
        model = GPT(GPTConfig(**config["model"])).to(device)
        model.load_state_dict(state["model"])
        vocab_path = Path(config["data_dir"]) / "tokenizer.json"
        if file_sha256(vocab_path) != state["data_sha256"]["tokenizer.json"]:
            raise ValueError(f"Saved tokenizer differs from the training tokenizer for {name}")
        tokenizer = BPETokenizer.load(vocab_path)
        for temp in TEMPERATURES:
            ids = sample(model, tokenizer.encode(prompt), temp, args.new_tokens, args.seed)
            samples.append(
                {
                    "run": name,
                    "preset": config["preset"],
                    "step": state["step"],
                    "temperature": temp,
                    "checkpoint": path.as_posix(),
                    "token_ids": ids,
                    "text": tokenizer.decode(ids),
                }
            )
    payload = {
        "seed": args.seed,
        "prompt": prompt,
        "max_new_tokens": args.new_tokens,
        "temperatures": TEMPERATURES,
        "samples": samples,
    }
    write_json(Path("results/samples_phase2.json"), payload)
    lines = [
        "# TinyShakespeare samples",
        "",
        f"Each continuation starts from a single newline and uses seed {args.seed} with up to "
        f"{args.new_tokens} new tokens. The sampler draws from the full temperature-scaled "
        "distribution and stops at end-of-text or the token limit. "
        "The prompt is omitted below. [Exact token IDs and text](../results/samples_phase2.json)",
    ]
    for row in samples:
        lines.extend(
            [
                "",
                f"## {row['preset'].title()} · temperature {row['temperature']}",
                "",
                f"Checkpoint at step {row['step']}.",
                "",
                "```text",
                row["text"],
                "```",
            ]
        )
    write_bytes(Path("docs/samples_phase2.md"), ("\n".join(lines) + "\n").encode())
    print(json.dumps({"samples": len(samples), "path": "docs/samples_phase2.md"}))


if __name__ == "__main__":
    main()
