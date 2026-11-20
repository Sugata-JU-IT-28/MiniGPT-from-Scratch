"""Train the selected architecture at the largest profiled device scale and publish its samples."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minigpt.checkpoint import copy_checkpoint
from minigpt.data import default_device, seed_everything
from minigpt.experiment_runner import check_dir, read_json, run_one
from minigpt.experiment_timing import FINAL_CPU, FINAL_GPU
from minigpt.experiments import SUITES, Variant, run_config, variant
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
    lr = decision["by_preset"].get(name, decision["shared_lr"])
    return spec, {
        "architecture": name,
        "selection_metric": leader["endpoints"]["val_loss"],
        "selection_seeds": leader["seeds"],
        "lr": lr,
        "architecture_rule": "Lowest mean endpoint validation loss over architecture seeds; "
        "ties use fewer parameters, then name.",
        "lr_rule": "Use the winning preset's individual sweep optimum when available, "
        "otherwise use the common LR tested on the winning toggle combination.",
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
    samples = []
    for temp in (0.5, 0.8, 1.0):
        tokens = generate(
            model,
            prompt_ids,
            max_new_tokens=count,
            temperature=temp,
            seed=seed,
            eos_id=EOT_ID,
            use_cache=True,
        )
        samples.append({"temperature": temp, "ids": tokens, "text": tokenizer.decode(tokens)})
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
        "samples": samples,
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
        f"Each continuation starts with a newline, uses seed {seed}, and allows up to {count} "
        "new tokens. Temperature is the only sampling setting varied. Samples are unedited.",
        "",
    ]
    for sample in samples:
        lines.extend(
            [f"## Temperature {sample['temperature']}", "", "```text", sample["text"], "```", ""]
        )
    write_bytes(root / "docs/samples_final.md", "\n".join(lines).encode("utf-8"))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Resume training and reproducibly export the selected checkpoint."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    seed_everything(args.seed)
    root = Path(".")
    plan = read_json(root / "results/experiment_plan.json")
    summary = read_json(root / "results/experiment_aggregate.json")
    spec, decision = final_choice(summary, default_device().type)
    cfg = run_config(spec, args.seed, plan["final_token_budget"], decision["lr"])
    result = run_one(root, "final", spec.name, cfg)
    source = check_dir(root, "final", cfg, False) / "best.pt"
    report = publish(root, source, result, decision, args.seed)
    print(
        f"Published checkpoints/final.pt, reloaded validation loss {report['validation']['loss']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
