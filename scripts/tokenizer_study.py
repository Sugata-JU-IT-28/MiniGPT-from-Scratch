"""Measure independent BPE fits and resume completed vocabulary sizes from JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from time import perf_counter_ns
from typing import TypedDict

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from minigpt import corpus
from minigpt.corpus import read_corpus, split_text, text_digest
from minigpt.data import seed_everything
from minigpt.io import write_json
from minigpt.tokenizer import BPETokenizer, bpe

SIZES = (257, 512, 1024, 2048, 4096)
PROBE_SIZE = 1024
PROBES = {
    "English": (
        "Rain tapped on the kitchen window. I put the kettle on and watched a sparrow "
        "shelter under the sill. When the clouds cleared, it shook its wings and flew away."
    ),
    "Bengali": (
        "রান্নাঘরের জানালায় বৃষ্টির ফোঁটা পড়ছিল। আমি চায়ের জল বসিয়ে দেখলাম, "
        "একটি চড়ুই জানালার নিচে আশ্রয় নিয়েছে। মেঘ কেটে গেলে পাখিটি ডানা ঝেড়ে উড়ে গেল।"
    ),
    "Hindi": (
        "रसोई की खिड़की पर बारिश की बूँदें गिर रही थीं। मैंने चाय का पानी चढ़ाया और देखा कि "
        "एक गौरैया खिड़की के नीचे दुबकी हुई थी। बादल छँटे तो वह पंख झाड़कर उड़ गई।"
    ),
}


class Measurement(TypedDict):
    """Measured validation compression and wall time for one independent fit."""

    vocab_size: int
    val_tokens: int
    bytes_per_token: float
    compression_vs_baseline: float
    train_time_ns: int


class Probe(TypedDict):
    """UTF-8 bytes and Unicode code points per probe paragraph."""

    text: str
    bytes: int
    characters: int
    tokens: int
    bytes_per_token: float
    tokens_per_character: float


class Study(TypedDict):
    """Persist enough context to reuse only compatible completed measurements."""

    context: dict[str, object]
    measurements: list[Measurement]
    probes: dict[str, Probe]


def measure_probes(tokenizer: BPETokenizer) -> dict[str, Probe]:
    """Count tokens in parallel descriptions of the same scene."""
    probes: dict[str, Probe] = {}
    for language, text in PROBES.items():
        tokens = len(tokenizer.encode(text))
        nbytes = len(text.encode("utf-8"))
        probes[language] = Probe(
            text=text,
            bytes=nbytes,
            characters=len(text),
            tokens=tokens,
            bytes_per_token=nbytes / tokens,
            tokens_per_character=tokens / len(text),
        )
    return probes


def fit_study(text: str, seed: int, path: Path, restart: bool) -> Study:
    """Fit each missing size and checkpoint only after its measurements are complete."""
    train, val = split_text(text)
    if not train or not val:
        raise ValueError("The study needs non-empty training and validation splits")
    val_bytes = len(val.encode("utf-8"))
    sources = (Path(__file__), Path(bpe.__file__), Path(corpus.__file__))
    context: dict[str, object] = {
        "seed": seed,
        "corpus_sha256": text_digest(text),
        "implementation_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "train_characters": len(train),
        "val_characters": len(val),
        "train_bytes": len(train.encode("utf-8")),
        "val_bytes": val_bytes,
        "vocab_sizes": list(SIZES),
        "probe_vocab_size": PROBE_SIZE,
    }
    study = Study(context=context, measurements=[], probes={})
    if path.exists() and not restart:
        saved: Study = json.loads(path.read_text(encoding="utf-8"))
        if saved["context"] == context:
            study = saved
    rows = study["measurements"]
    for size in SIZES:
        if any(row["vocab_size"] == size for row in rows):
            print(f"Reusing measured vocabulary {size}", flush=True)
            continue
        print(f"Training vocabulary {size}", flush=True)
        started = perf_counter_ns()
        tokenizer = BPETokenizer().train(train, size)
        elapsed = perf_counter_ns() - started
        val_tokens = len(tokenizer.encode(val))
        baseline_tokens = rows[0]["val_tokens"] if rows else val_tokens
        rows.append(
            Measurement(
                vocab_size=size,
                val_tokens=val_tokens,
                bytes_per_token=val_bytes / val_tokens,
                compression_vs_baseline=baseline_tokens / val_tokens,
                train_time_ns=elapsed,
            )
        )
        if size == PROBE_SIZE:
            study["probes"] = measure_probes(tokenizer)
        write_json(path, study)
    return study


def display(value: float | Decimal) -> str:
    """Truncate display metrics to four decimal places, retaining full precision in JSON."""
    return str(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def plot_study(study: Study, path: Path) -> None:
    """Plot held-out compression and token cost per Unicode code point."""
    figure = Figure(figsize=(10, 4.2), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    rows = study["measurements"]
    sizes = [r["vocab_size"] for r in rows]
    left.plot(sizes, [r["bytes_per_token"] for r in rows], "o-", color="#15675b")
    left.set_xscale("log", base=2)
    left.set_xticks(sizes, labels=[str(size) for size in sizes])
    left.set(
        xlabel="Total vocabulary entries", ylabel="UTF-8 bytes per token", title="Held-out corpus"
    )
    probes = study["probes"]
    right.bar(
        list(probes),
        [p["tokens_per_character"] for p in probes.values()],
        color=["#15675b", "#486a98", "#99572e"],
    )
    right.set(ylabel="Tokens per Unicode code point", title=f"Probes at {PROBE_SIZE} entries")
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)


def write_report(study: Study, docs_dir: Path) -> None:
    """Generate the Markdown interpretation and chart from the same checkpoint."""
    rows = study["measurements"]
    context = study["context"]
    lines = [
        "# Byte-level BPE on held-out text",
        "",
        f"Each vocabulary was fitted independently on the first {context['train_bytes']:,} "
        f"UTF-8 bytes of the corpus. The contiguous suffix contains {context['val_bytes']:,} "
        "bytes. Training time measures only the tokenizer fit on this machine.",
        "",
        "Compression is the baseline token count divided by the fitted tokenizer’s count. "
        "It measures sequence length. Ratios are truncated for display. "
        "Elapsed seconds are converted directly from the recorded nanoseconds.",
        "",
        "| Vocabulary | Validation tokens | Bytes/token | Compression vs. bytes | Fit time (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seconds = Decimal(row["train_time_ns"]) / Decimal(10**9)
        lines.append(
            f"| {row['vocab_size']} | {row['val_tokens']:,} | {display(row['bytes_per_token'])} "
            f"| {display(row['compression_vs_baseline'])}× | {seconds:.9f} |"
        )
    lines += [
        "",
        "## Multilingual probe",
        "",
        f"The {PROBE_SIZE}-entry tokenizer encoded short descriptions of the same scene. "
        "Characters are Unicode code points, including spaces, punctuation and combining marks.",
        "",
        "| Language | Bytes | Characters | Tokens | Bytes/token | Tokens/character |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language, probe in study["probes"].items():
        lines.append(
            f"| {language} | {probe['bytes']} | {probe['characters']} | {probe['tokens']} "
            f"| {display(probe['bytes_per_token'])} | {display(probe['tokens_per_character'])} |"
        )
    first, second, before_last, last = rows[0], rows[1], rows[-2], rows[-1]

    def savings(a: Measurement, b: Measurement) -> float:
        return (a["val_tokens"] - b["val_tokens"]) / (b["vocab_size"] - a["vocab_size"])

    early, late = savings(first, second), savings(before_last, last)
    trend = "fell" if late < early else "changed"
    probes = study["probes"]
    english = probes["English"]["tokens_per_character"]
    bengali_ratio = probes["Bengali"]["tokens_per_character"] / english
    hindi_ratio = probes["Hindi"]["tokens_per_character"] / english
    lines += [
        "",
        "## Interpretation",
        "",
        f"Validation tokens saved per added vocabulary entry {trend} from {display(early)} "
        f"over the first interval to {display(late)} over the last. "
        "Each added entry costs another embedding row at fixed model width. "
        "On this small corpus, the compression gain must justify that additional capacity. "
        f"Per code point, the Bengali passage used {display(bengali_ratio)}× as many tokens "
        f"as the English passage and the Hindi passage used {display(hindi_ratio)}×. "
        "These ratios describe the three passages at the measured vocabulary size.",
        "",
        "![Held-out compression and multilingual token costs](img/tokenizer_study.png)",
        "",
        "### Probe texts",
        "",
    ]
    for language, probe in study["probes"].items():
        lines.extend([f"**{language}**  ", probe["text"], ""])
    plot_study(study, docs_dir / "img/tokenizer_study.png")
    (docs_dir / "tokenizer_study.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Run or resume the study and regenerate both presentation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    path = args.results_dir / "tokenizer_study.json"
    study = fit_study(read_corpus(args.input), args.seed, path, args.restart)
    write_report(study, args.docs_dir)


if __name__ == "__main__":
    main()
