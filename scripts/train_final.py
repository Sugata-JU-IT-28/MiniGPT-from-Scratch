"""Train the selected architecture at the largest profiled device scale and publish its samples."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from minigpt.checkpoint import copy_checkpoint
from minigpt.data import default_device, seed_everything
from minigpt.experiment_runner import check_dir, read_json, run_one
from minigpt.experiment_timing import FINAL_CPU, FINAL_GPU, model_key, probe
from minigpt.experiments import SUITES, TOKENS_PER_UPDATE, Variant, run_config, variant
from minigpt.generate import generate, load_model
from minigpt.io import display_metric, file_sha256, write_bytes, write_json
from minigpt.metrics import evaluate
from minigpt.tokenizer.bpe import EOT_ID


def final_choice(summary: dict[str, Any], device_type: str) -> tuple[Variant, dict[str, Any]]:
    """Select by replicated endpoint loss; use the available LR evidence for that architecture."""
    scores = summary["suites"]["arch_ablation"]
    name = min(
        scores, key=lambda n: (scores[n]["endpoints"]["val_loss"]["mean"], scores[n]["params"], n)
    )
    leader = scores[name]
    runner_up = min(
        (n for n in scores if n != name),
        key=lambda n: (scores[n]["endpoints"]["val_loss"]["mean"], scores[n]["params"], n),
    )
    first = leader["endpoints"]["val_loss"]
    second = scores[runner_up]["endpoints"]["val_loss"]
    separation = second["mean"] - first["mean"]
    selected = next(s for s in SUITES["arch_ablation"] if s.name == name)
    sizes = FINAL_CPU if device_type == "cpu" else FINAL_GPU
    model = selected.model
    spec = variant(
        "final",
        selected.preset,
        n_embd=sizes.model.n_embd,
        n_layer=sizes.model.n_layer,
        n_head=sizes.model.n_head,
        block_size=sizes.model.block_size,
        norm_type=model.norm_type,
        pos_encoding=model.pos_encoding,
        mlp_type=model.mlp_type,
    )
    decision = summary["lr_selection"]
    lr = decision["shared_lr"]
    return spec, {
        "architecture": name,
        "selection_metric": leader["endpoints"]["val_loss"],
        "selection_seeds": leader["seeds"],
        "runner_up_architecture": runner_up,
        "runner_up_loss_difference": separation,
        "exceeds_larger_seed_std": separation > max(first["std"], second["std"]),
        "lr": lr,
        "architecture_rule": "Lowest mean endpoint validation loss over architecture seeds; "
        "ties use fewer parameters, then name.",
        "shared_lr": lr,
        "lr_rule": "Choose the lower endpoint validation loss on the final shape "
        "from the shared LR and half the shared LR.",
        "size_rule": "Largest CPU shape in the measured scaling grid, or gpu_small on CUDA.",
    }


def publish(
    root: Path, source: Path, training: dict[str, Any], decision: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Verify the promoted checkpoint by identity and held-out loss before sampling."""
    artifact = root / "checkpoints/final.pt"
    sha = file_sha256(source)
    if not artifact.is_file() or file_sha256(artifact) != sha:
        copy_checkpoint(source, artifact)
    if file_sha256(artifact) != sha:
        raise ValueError("Promoted checkpoint checksum differs from the selected checkpoint")
    cfg = training["config"]
    data = root / cfg["data_dir"]
    model, tokenizer = load_model(artifact, data / "tokenizer.json")
    validation = evaluate(model, "val", data_dir=data, batch_size=cfg["train"]["eval_batch_size"])
    if abs(validation.loss - training["best_val"]) > 1e-5:
        raise ValueError("Reloaded checkpoint did not reproduce the selected validation loss")
    prompt = "\n"
    prompt_ids = tokenizer.encode(prompt)
    count = min(96, model.config.block_size - len(prompt_ids))
    greedy_args = dict(max_new_tokens=256, temperature=0, seed=seed, window_policy="reprefill")
    cached = generate(model, prompt_ids, use_cache=True, **greedy_args)
    plain = generate(model, prompt_ids, use_cache=False, **greedy_args)
    if len(cached) != 256 or cached != plain:
        raise ValueError("Cached and uncached greedy IDs differ on the promoted checkpoint")
    groups = []
    for cue in (prompt, "KING RICHARD:\n", "What say you,"):
        ids = tokenizer.encode(cue)
        samples = []
        for temp in (0.5, 0.8, 1.0):
            tokens = generate(
                model,
                ids,
                max_new_tokens=count,
                temperature=temp,
                seed=seed,
                eos_id=EOT_ID,
                use_cache=True,
                window_policy="reprefill",
            )
            samples.append({"temperature": temp, "ids": tokens, "text": tokenizer.decode(tokens)})
        groups.append({"prompt": cue, "prompt_ids": ids, "samples": samples})
    report = {
        "selection": decision,
        "config": cfg,
        "seed": seed,
        "step": training["step"],
        "best_step": training["best_step"],
        "tokens_seen": training["tokens_seen"],
        "params": training["params"],
        "non_embedding_params": training["non_embedding_params"],
        "elapsed_seconds": training["elapsed"],
        "checkpoint_sha256": sha,
        "source": source.relative_to(root).as_posix(),
        "validation": asdict(validation),
        "endpoint": training["final"],
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "max_new_tokens": count,
        "eos_id": EOT_ID,
        "samples": groups[0]["samples"],
        "sample_sets": groups,
        "greedy_cached_equals_uncached": True,
        "greedy_ids": cached,
    }
    write_json(root / "results/final_model.json", report)
    lines = [
        "# Samples from the final MiniGPT model",
        "",
        f"The {decision['architecture']} architecture trained on {training['tokens_seen']:,} "
        f"tokens at peak learning rate {decision['lr']:g}. Its selected checkpoint scored "
        f"{display_metric(validation.loss)} nats/token and {display_metric(validation.bpb)} "
        "bits/byte on validation after reloading. [Training record and token IDs]"
        "(../results/final_model.json)",
        "",
        f"Each prompt uses seed {seed} and allows up to {count} new tokens. Within a prompt, "
        "temperature is the only sampling setting varied. Samples are unedited.",
        "",
    ]
    for group in groups:
        lines.extend([f"## Prompt {group['prompt']!r}", ""])
        for sample in group["samples"]:
            lines.extend(
                [
                    f"### Temperature {sample['temperature']}",
                    "",
                    "```text",
                    sample["text"],
                    "```",
                    "",
                ]
            )
    write_bytes(root / "docs/samples_final.md", "\n".join(lines).encode("utf-8"))
    return report


def plan_final(root: Path, spec: Variant, budget: int, seed: int) -> tuple[Variant, dict[str, Any]]:
    """Select the largest eligible shape that fits the time limit on this device."""
    timings = read_json(root / "results/experiment_timing.json")["timings"]
    candidates = [spec] + [
        replace(
            spec,
            model=replace(
                spec.model, n_layer=v.model.n_layer, n_embd=v.model.n_embd, n_head=v.model.n_head
            ),
        )
        for v in reversed(SUITES["scaling"])
        if v.model.n_embd < spec.model.n_embd
    ]
    attempts = []
    for candidate in candidates:
        key = model_key(candidate.model)
        timing = timings.get(key) or probe(candidate, seed)
        secs = 1.25 * (
            budget / TOKENS_PER_UPDATE * timing["step_seconds"]
            + 5
            * timing["split_targets"]
            / timing["eval_batch_tokens"]
            * timing["eval_batch_seconds"]
            + 3
        )
        attempts.append(
            {"model": asdict(candidate.model), "timing": timing, "estimated_seconds": secs}
        )
        if secs <= 6000:
            return candidate, {
                "token_budget": budget,
                "estimated_seconds": secs,
                "candidates": attempts,
                "seed": seed,
                "selected_model": asdict(candidate.model),
            }
    raise ValueError("Even the smallest profiled shape exceeds the final time limit")


def transfer(root: Path, spec: Variant, budget: int, seed: int, lr: float) -> dict[str, Any]:
    """Compare two independent schedules with identical initialization and sample order."""
    tokens = max(TOKENS_PER_UPDATE, (budget // 10 // TOKENS_PER_UPDATE) * TOKENS_PER_UPDATE)
    records = []
    for rate in (lr, lr / 2):
        trial = replace(spec, name=f"final_lr{rate:g}")
        cfg = run_config(trial, seed, tokens, rate)
        outcome = run_one(root, "final_lr_check", trial.name, cfg)
        records.append(
            {
                "lr": rate,
                "val_loss": outcome["final"]["val_loss"],
                "run": cfg.name,
                "tokens_seen": outcome["tokens_seen"],
            }
        )
    winner = min(records, key=lambda r: r["val_loss"])
    result = {
        "shared_lr": lr,
        "selected_lr": winner["lr"],
        "seed": seed,
        "model": asdict(spec.model),
        "token_budget": tokens,
        "probes": records,
        "reason": "Lowest endpoint validation loss",
    }
    write_json(root / "results/final_lr_check.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Resume training and reproducibly export the selected checkpoint."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--token-budget", type=int, default=3145728)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args(argv)
    seed_everything(args.seed)
    root = Path(".")
    phase2 = read_json(root / "results/phase2_summary.json")
    if args.token_budget < phase2["tokens_per_run"] or args.token_budget % TOKENS_PER_UPDATE:
        p.error("Budget must cover Phase 2 and contain whole optimizer updates")
    summary = read_json(root / "results/experiment_aggregate.json")
    summary["lr_selection"] = read_json(root / "results/lr_selection.json")
    spec, decision = final_choice(summary, default_device().type)
    path = root / "results/final_plan.json"
    plan = read_json(path) if path.is_file() else {}
    if (
        plan.get("token_budget") == args.token_budget
        and plan.get("seed") == args.seed
        and plan.get("requested_model") == asdict(spec.model)
    ):
        spec = replace(spec, model=replace(spec.model, **plan["selected_model"]))
    else:
        requested = asdict(spec.model)
        spec, plan = plan_final(root, spec, args.token_budget, args.seed)
        write_json(path, {**plan, "requested_model": requested})
    print(
        f"Estimated final fit {display_metric(plan['estimated_seconds'] / 60, 1)} minutes, "
        f"{spec.model.n_layer} layers, width {spec.model.n_embd}, {args.token_budget} tokens.",
        flush=True,
    )
    if args.plan_only:
        return
    check = transfer(root, spec, args.token_budget, args.seed, decision["shared_lr"])
    decision["lr"] = check["selected_lr"]
    print(f"Transfer probes {check['probes']}. Selected {decision['lr']}.", flush=True)
    cfg = run_config(spec, args.seed, args.token_budget, decision["lr"])
    result = run_one(root, "final", spec.name, cfg, require_weights=True)
    source = check_dir(root, "final", cfg, False) / "best.pt"
    report = publish(root, source, result, decision, args.seed)
    last_source = source.with_name("last.pt")
    last = root / "checkpoints/final_last.pt"
    copy_checkpoint(last_source, last)
    if file_sha256(last) != file_sha256(last_source):
        raise ValueError("Final last checkpoint checksum mismatch")
    best_row = next(row for row in result["history"] if row["step"] == result["best_step"])
    previous = next(r for r in phase2["runs"] if r["preset"] == phase2["selected"]["preset"])[
        "best"
    ]
    tokenizer = Path(cfg.data_dir) / "tokenizer.json"
    write_json(
        root / "results/final_summary.json",
        {
            "architecture": decision["architecture"],
            "preset": cfg.preset,
            "model": asdict(cfg.model),
            "step": result["best_step"],
            "val_loss": report["validation"]["loss"],
            "val_ppl": report["validation"]["ppl"],
            "val_bpb": report["validation"]["bpb"],
            "train_loss": best_row["train_loss"],
            "params": result["params"],
            "non_embedding_params": result["non_embedding_params"],
            "token_budget": args.token_budget,
            "lr": decision["lr"],
            "tokenizer": tokenizer.as_posix(),
            "tokenizer_sha256": file_sha256(tokenizer),
            "best_sha256": report["checkpoint_sha256"],
            "last_sha256": file_sha256(last),
            "phase2_best": previous,
            "beats_phase2": report["validation"]["loss"] < previous["val_loss"],
            "greedy_match_tokens": len(report["greedy_ids"]),
            "seed": args.seed,
        },
    )
    print(
        f"Published checkpoints/final.pt, reloaded validation loss {report['validation']['loss']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
