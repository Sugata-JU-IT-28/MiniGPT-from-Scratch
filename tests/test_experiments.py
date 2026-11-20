"""Exercise fixed budgets, replicated model selection, recovery, reporting and promotion."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from minigpt import experiment_timing as timing
from minigpt.config import RunConfig
from minigpt.data import prepare_dataset
from minigpt.experiment_runner import check_dir, execute, read_json, run_one
from minigpt.experiments import QUICK_TOKENS, SEEDS, SUITES, run_config, select_lr, variant
from minigpt.io import file_sha256, write_json
from minigpt.metrics import write_metrics
from minigpt.train import Trainer
from scripts import aggregate_results as report
from scripts import run_experiments as runner
from scripts import train_final as final


def test_declarative_grids_are_matched() -> None:
    for entries in SUITES.values():
        assert len({s.name for s in entries}) == len(entries)
        for entry in entries:
            cfg = run_config(entry, 1, QUICK_TOKENS, 0.001, quick=True)
            assert cfg.steps == 50
            assert (
                cfg.steps * cfg.model.block_size * cfg.train.grad_accum * cfg.train.batch_size
                == QUICK_TOKENS
            )
            assert cfg.data_dir.endswith(str(cfg.model.vocab_size))
    shapes = {s.name: s.model for s in SUITES["shape_ablation"]}
    base = shapes.pop("base")
    for model in shapes.values():
        assert (
            sum(a != b for a, b in zip(asdict(base).values(), asdict(model).values(), strict=True))
            == 1
        )
    assert {m.n_head for m in [base, *shapes.values()]} == {2, 4, 8}
    assert {m.block_size for m in [base, *shapes.values()]} == {64, 128, 256}
    with pytest.raises(ValueError, match="complete matched-token"):
        run_config(SUITES["arch_ablation"][0], 0, 1234, 0.001)


def test_lr_decision_handles_different_winners() -> None:
    records = []
    for spec in SUITES["lr_sweep"]:
        ideal = 0.001 if spec.preset == "baseline" else 0.003
        loss = abs(math.log(spec.lr / ideal)) * (2 if spec.preset == "baseline" else 1)
        records.append(
            {
                "config": asdict(run_config(spec, 0, QUICK_TOKENS, 0.001)),
                "final": {"val_loss": loss},
            }
        )
    decision = select_lr(records)
    assert decision["by_preset"] == {"baseline": 0.001, "modern": 0.003}
    assert decision["shared_lr"] == 0.001
    with pytest.raises(ValueError, match="Missing LR"):
        select_lr(records[1:])


@pytest.fixture
def small_fit(tmp_path: Path) -> RunConfig:
    """Real, small training corpus for recovery and promotion checks."""
    spec = variant("small", "modern", vocab_size=257, n_embd=16, n_layer=1, n_head=2, block_size=8)
    cfg = run_config(spec, 0, 2048, 0.002)
    prepare_dataset("I saw the moon above the hill.\n" * 22, 257, tmp_path / cfg.data_dir)
    return replace(cfg, train=replace(cfg.train, device="cpu", cpu_threads=1))


def test_run_resume_skip_repair_and_promote(tmp_path: Path, small_fit: RunConfig) -> None:
    cfg = small_fit
    checks = check_dir(tmp_path, "final", cfg, False)
    log = tmp_path / "results/runs/final/small_s0.csv"
    trainer = Trainer(cfg, tmp_path, metrics_path=log, checkpoint_dir=checks)
    with_signals = trainer.run(2, export_report=False)
    assert with_signals["step"] < cfg.steps
    result = run_one(tmp_path, "final", "small", cfg)
    assert [r["step"] for r in result["history"]] == list(range(cfg.steps + 1))
    assert result["params"] > result["non_embedding_params"] and result["step"] == cfg.steps
    original = log.read_bytes()
    log.unlink()
    assert run_one(tmp_path, "final", "small", cfg) == result and log.read_bytes() == original
    with pytest.raises(ValueError, match="does not match"):
        run_one(tmp_path, "final", "small", replace(cfg, train=replace(cfg.train, lr=0.003)))
    decision = {"architecture": "modern", "lr": cfg.train.lr}
    published = final.publish(tmp_path, checks / "best.pt", result, decision, 0)
    assert published["validation"]["loss"] == pytest.approx(result["best_val"], abs=1e-6)
    assert file_sha256(tmp_path / "checkpoints/final.pt") == published["checkpoint_sha256"]
    assert len(published["samples"]) == 3 and all(s["ids"] for s in published["samples"])
    assert (tmp_path / "docs/samples_final.md").is_file()


def test_real_probe_and_persisted_plan(
    tmp_path: Path, small_fit: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    spec = variant("small", "modern", **asdict(small_fit.model))
    result = timing.probe(spec, 0)
    assert len(result["step_trials"]) == len(result["eval_batch_trials"]) == 8
    assert result["step_seconds"] > 0 and result["eval_batch_seconds"] > 0

    def fake_probe(spec: Any, seed: int) -> dict[str, Any]:
        return result

    monkeypatch.setattr(timing, "probe", fake_probe)
    plan = timing.make_plan(0)
    assert plan["total_estimated_seconds"] <= 3 * 3600 and plan["seeds"] == SEEDS
    assert plan["final_token_budget"] == 4 * plan["token_budget"]
    monkeypatch.setattr(timing, "probe", lambda *a: pytest.fail("probe repeated"))
    assert timing.make_plan(0) == plan


def test_quick_cli_and_missing_sweep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = []
    monkeypatch.setattr(runner, "execute", lambda *args, **kwargs: seen.append((args, kwargs)))
    runner.main(
        ["--suite", "shape_ablation", "--seed", "3", "--seeds", "2", "--quick", "--only", "base"]
    )
    assert seen[0][0][0] == ["shape_ablation"] and seen[0][0][2:] == (3, 2)
    assert seen[0][1] == {"quick": True, "only": "base"}
    with pytest.raises(ValueError, match="lr_sweep"):
        execute(
            ["scaling"],
            {"token_budget": QUICK_TOKENS, "seeds": SEEDS},
            0,
            None,
            quick=False,
            root=tmp_path,
        )
    with pytest.raises(SystemExit):
        runner.main(["--seeds", "0"])


def test_aggregation_tables_figures_and_final_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    budget = 2048
    write_json(Path("results/experiment_plan.json"), {"token_budget": budget, "seeds": SEEDS})
    choice = {
        "token_budget": budget,
        "shared_lr": 0.003,
        "by_preset": {"baseline": 0.001, "modern": 0.003},
    }
    write_json(Path("results/lr_selection.json"), choice)
    for suite, specs in SUITES.items():
        for index, spec in enumerate(specs):
            for seed in range(SEEDS[suite]):
                cfg = run_config(spec, seed, budget, 0.003)
                val = 4 + seed * 0.02 - index * 0.01
                row = {
                    "step": cfg.steps,
                    "tokens_seen": budget,
                    "train_loss": val - 0.1,
                    "val_loss": val,
                    "val_bpb": val / 2,
                    "val_ppl": math.exp(val),
                    "lr": cfg.train.lr,
                    "tokens_per_sec": 1.0,
                    "elapsed": 1.0,
                }
                rec = {
                    "name": spec.name,
                    "suite": suite,
                    "config": asdict(cfg),
                    "mode": "full",
                    "seed": seed,
                    "step": cfg.steps,
                    "tokens_seen": budget,
                    "params": 4000,
                    "non_embedding_params": 3000,
                    "final": row,
                    "history": [row],
                    "data_sha256": {"train.bin": "fixture"},
                    "elapsed": 1.0,
                }
                path = Path("results/runs") / suite / f"{cfg.name}.json"
                write_json(path, rec)
                write_metrics(path.with_suffix(".csv"), [row])
    report.main(["--seed", "0"])
    summary = read_json(Path("results/experiment_aggregate.json"))
    assert summary["run_count"] == 62
    comparisons = summary["comparisons"]["arch_ablation"]
    assert comparisons[0]["verdict"] == "inconclusive"
    assert comparisons[-1]["verdict"] == "lower"
    assert report.measured([2.0, 4.0]) == {"mean": 3.0, "std": pytest.approx(math.sqrt(2))}
    assert summary["suites"]["lr_sweep"]["modern_lr0.003"]["endpoints"]["val_loss"]["std"] is None
    assert "bits/byte" in Path("docs/results_tables.md").read_text(encoding="utf-8")
    assert len(list(Path("docs/img").glob("*.png"))) == 6
    selected, why = final.final_choice(summary, "cpu")
    assert (
        why["architecture"] == "modern" and selected.model.n_embd == timing.FINAL_CPU.model.n_embd
    )
    assert why["lr"] == 0.003
    summary["suites"]["arch_ablation"]["rmsnorm"]["endpoints"]["val_loss"]["mean"] = 0.5
    hybrid, why = final.final_choice(summary, "cuda")
    assert hybrid.model.norm_type == "rmsnorm" and hybrid.model.pos_encoding == "learned"
    assert hybrid.model.n_embd == timing.FINAL_GPU.model.n_embd and why["lr"] == choice["shared_lr"]
    path.with_suffix(".csv").write_text("step\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent"):
        report.collect(Path("."), 0)
