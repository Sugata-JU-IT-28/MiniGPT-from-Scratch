"""Measure end-to-end greedy decode latency and allocated KV tensor storage."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import Tensor

from minigpt.config import load_config
from minigpt.data import default_device, seed_everything
from minigpt.generate import generate, load_model
from minigpt.io import display_metric, file_sha256, write_bytes, write_json
from minigpt.model import GPT, KVCache

SIZES = (32, 64, 128, 256, 512)


def cache_bytes(cache: KVCache) -> int:
    """Count actual bytes of distinct live backing storages, including any retained views."""
    storages = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for pair in cache for t in pair
    }
    return sum(storages.values())


@torch.inference_mode()
def decode(model: GPT, prompt: Tensor, length: int, cached: bool) -> tuple[Tensor, KVCache | None]:
    """Time the same batched greedy workload in both modes, including prompt prefill."""
    tokens = prompt
    cache = None
    for _ in range(length):
        result = model(tokens if cache is None else tokens[:, -1:], past_kv=cache, use_cache=cached)
        if cached:
            cache = result[-1]
        token = result[0][:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, token), dim=-1)
    return tokens, cache


def measure(model: GPT, prompt: Tensor, length: int, repeats: int) -> dict[str, Any]:
    """Warm both paths and alternate measurement order, synchronizing device work at the edges."""
    device = prompt.device

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for cached in (False, True):
        decode(model, prompt, min(8, length), cached)
    trials: dict[bool, list[float]] = {False: [], True: []}
    storage = element_bytes = positions = 0
    for trial in range(repeats):
        outputs = {}
        for cached in (False, True) if trial % 2 == 0 else (True, False):
            sync()
            start = time.perf_counter()
            tokens, cache = decode(model, prompt, length, cached)
            sync()
            trials[cached].append(time.perf_counter() - start)
            outputs[cached] = tokens
            if cache is not None:
                storage = cache_bytes(cache)
                positions = cache[0][0].size(2)
                element_bytes = cache[0][0].element_size()
            del cache
        if not torch.equal(outputs[False], outputs[True]):
            raise RuntimeError("Greedy cached and uncached benchmark tokens diverged")
    batch = prompt.size(0)
    uncached, cached_time = (statistics.median(trials[key]) for key in (False, True))
    theoretical = 2 * model.config.n_layer * positions * model.config.n_embd * element_bytes * batch
    return {
        "batch": batch,
        "length": length,
        "cached_positions": positions,
        "uncached_seconds": uncached,
        "cached_seconds": cached_time,
        "uncached_tokens_per_sec": batch * length / uncached,
        "cached_tokens_per_sec": batch * length / cached_time,
        "speedup": uncached / cached_time,
        "cache_bytes": storage,
        "formula_bytes": theoretical,
        "element_bytes": element_bytes,
        "difference_bytes": storage - theoretical,
        "uncached_trials_seconds": trials[False],
        "cached_trials_seconds": trials[True],
    }


def write_report(report: dict[str, Any], docs: Path) -> None:
    """Render timing and memory measurements from the resumable JSON record."""
    rows = report["rows"]
    fig = Figure(figsize=(9, 4), layout="constrained")
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)
    lines = [
        "# KV-cache benchmark",
        "",
        f"Random cpu_tiny weights, context capacity {report['capacity']}, "
        f"{report['prompt_tokens']} prompt tokens, device {report['device']}. "
        f"Each median covers {report['repeats']} timed greedy generations after warmup. "
        "Latency includes prefill and all draws. Throughput counts new tokens across the batch. "
        "[Trials and machine details](../results/kv_cache_benchmark.json)",
        "",
        "| Preset | Batch | New tokens | Plain s | Cached s | Plain tok/s | Cached tok/s "
        "| Speedup | Cached positions | Storage bytes | Formula bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for preset in ("baseline", "modern"):
        for batch in report["batches"]:
            group = sorted(
                (r for r in rows if r["preset"] == preset and r["batch"] == batch),
                key=lambda r: r["length"],
            )
            label = f"{preset}, B={batch}"
            left.plot(
                [r["length"] for r in group], [r["speedup"] for r in group], "o-", label=label
            )
            if preset == "baseline":
                right.plot(
                    [r["cached_positions"] for r in group],
                    [r["cache_bytes"] / 2**20 for r in group],
                    "o-",
                    label=f"B={batch}",
                )
            for r in group:
                lines.append(
                    f"| {preset} | {batch} | {r['length']} "
                    f"| {display_metric(r['uncached_seconds'])} "
                    f"| {display_metric(r['cached_seconds'])} "
                    f"| {display_metric(r['uncached_tokens_per_sec'], 1)} "
                    f"| {display_metric(r['cached_tokens_per_sec'], 1)} "
                    f"| {display_metric(r['speedup'], 3)}× "
                    f"| {r['cached_positions']} | {r['cache_bytes']} | {r['formula_bytes']} |"
                )
    smallest = min(rows, key=lambda r: r["speedup"])
    largest = max(rows, key=lambda r: r["speedup"])
    difference = max(abs(r["difference_bytes"]) for r in rows)
    lines += [
        "",
        f"The smallest observed speedup was {display_metric(smallest['speedup'], 3)}× "
        f"({smallest['preset']}, batch {smallest['batch']}, {smallest['length']} new tokens). "
        f"The largest was {display_metric(largest['speedup'], 3)}× ({largest['preset']}, batch "
        f"{largest['batch']}, {largest['length']} new tokens). Short continuations "
        "amortize prefill and per-step cache handling over fewer saved prefix computations.",
        "",
        "Storage counts unique live K and V tensor buffers after the last draw. "
        "The final sampled token has yet to enter the model. The prediction in bytes is "
        "`2 × layers × batch × cached positions × embedding width × bytes per element`. "
        f"The largest absolute discrepancy was {difference} bytes. "
        "Temporary attention matrices and allocator reserves are outside this measurement.",
        "",
        "![Measured speedup and cache storage](img/kv_cache_benchmark.png)",
        "",
        "## Context limits",
        "",
        "The default policy checks prompt plus requested continuation against block_size. "
        "With reprefill, a full cache is discarded. The last block_size−1 preceding tokens "
        "and the newest token are evaluated together again, matching uncached sliding windows. "
        "For learned position embeddings, surviving tokens need new positions starting at zero. "
        "Deeper cached states also contain information from evicted history, including with "
        "RoPE. Exact re-prefill costs quadratic attention work per step once full.",
        "",
        "The trained checkpoint also produced identical cached and uncached greedy continuations "
        "at the lengths recorded in the benchmark JSON.",
        "",
    ]
    left.axhline(1, color="#999999", linewidth=0.7)
    left.set(
        xlabel="New tokens per sequence", ylabel="Uncached / cached latency", title="Decode speedup"
    )
    right.set(xlabel="Cached positions", ylabel="KV storage (MiB)", title="Live cache tensors")
    for ax in (left, right):
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)
    target = docs / "img/kv_cache_benchmark.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    write_bytes(docs / "kv_cache_benchmark.md", "\n".join(lines).encode())


def main(argv: Sequence[str] | None = None) -> None:
    """Resume by (preset, batch, length); refresh only with --restart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 2 or min(args.batches) < 1:
        raise ValueError("Use positive batches and at least two repetitions")
    seed_everything(args.seed)
    device = default_device()
    if device.type == "cpu":
        torch.set_num_threads(2)
    path = Path("results/kv_cache_benchmark.json")
    spec = {
        "seed": args.seed,
        "repeats": args.repeats,
        "batches": args.batches,
        "lengths": list(SIZES),
        "capacity": 1024,
        "prompt_tokens": 8,
        "device": str(device),
        "torch": str(torch.__version__),
        "processor": platform.processor(),
        "threads": torch.get_num_threads(),
        "checkpoint_sha256": file_sha256(Path("checkpoints/best.pt")),
    }
    report: dict[str, Any] = dict(spec, rows=[])
    if path.exists() and not args.restart:
        report = json.loads(path.read_text(encoding="utf-8"))
        if any(report[k] != v for k, v in spec.items()):
            raise ValueError("Existing benchmark settings differ. Use --restart for new timings.")
    for preset in ("baseline", "modern"):
        seed_everything(args.seed)
        cfg = replace(
            load_config(Path("configs/cpu_tiny.yaml"), preset=preset).model, block_size=1024
        )
        model = GPT(cfg).to(device).eval()
        models = report.setdefault("models", {})
        if preset in models and models[preset] != asdict(cfg):
            raise ValueError("Existing benchmark model differs. Use --restart for new timings.")
        models[preset] = asdict(cfg)
        for batch in args.batches:
            gen = torch.Generator(device=device).manual_seed(args.seed)
            prompt = torch.randint(
                cfg.vocab_size, (batch, report["prompt_tokens"]), device=device, generator=gen
            )
            for length in SIZES:
                if any(
                    r["preset"] == preset and r["batch"] == batch and r["length"] == length
                    for r in report["rows"]
                ):
                    continue
                print(f"Timing {preset} B={batch} new_tokens={length}", flush=True)
                row = measure(model, prompt, length, args.repeats)
                report["rows"].append(dict(row, preset=preset))
                write_json(path, report)
    model, codec = load_model(Path("checkpoints/best.pt"))
    ids = codec.encode("\n")
    sanity = []
    for length in (16, 64):
        plain = generate(
            model, ids, max_new_tokens=length, temperature=0, use_cache=False, seed=args.seed
        )
        cached = generate(model, ids, max_new_tokens=length, temperature=0, seed=args.seed)
        if plain != cached:
            raise ValueError("Trained checkpoint greedy decode mismatch")
        sanity.append(
            {"length": length, "token_ids": cached, "text": codec.decode(cached), "matched": True}
        )
    report["trained_sanity"] = sanity
    write_json(path, report)
    write_report(report, Path("docs"))


if __name__ == "__main__":
    main()
