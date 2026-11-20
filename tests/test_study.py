"""Exercise measured output, offline resumption, and corpus-sensitive checkpoints."""

import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from minigpt.tokenizer import BPETokenizer
from scripts import tokenizer_study as study


def test_interrupted_study_resumes_and_regenerates_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "SIZES", (257, 259, 262))
    monkeypatch.setattr(study, "PROBE_SIZE", 259)
    raw = tmp_path / "corpus.txt"
    raw.write_bytes(("lowering slowly, waiting for lower winds.\r\n" * 12).encode())
    args = [
        "--input",
        str(raw),
        "--results-dir",
        str(tmp_path / "results"),
        "--docs-dir",
        str(tmp_path / "docs"),
        "--seed",
        "17",
    ]
    path = tmp_path / "results/tokenizer_study.json"
    train = BPETokenizer.train
    calls: list[int] = []

    def interrupted(self: BPETokenizer, text: str, size: int) -> BPETokenizer:
        calls.append(size)
        if size == 262:
            raise RuntimeError("simulated interruption")
        return train(self, text, size)

    monkeypatch.setattr(BPETokenizer, "train", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        study.main(args)
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert [r["vocab_size"] for r in checkpoint["measurements"]] == [257, 259]
    assert calls == [257, 259, 262]

    def counted(self: BPETokenizer, text: str, size: int) -> BPETokenizer:
        calls.append(size)
        return train(self, text, size)

    calls.clear()
    monkeypatch.setattr(BPETokenizer, "train", counted)
    study.main(args)
    assert calls == [262]
    result = json.loads(path.read_text(encoding="utf-8"))
    rows = result["measurements"]
    assert rows[:2] == checkpoint["measurements"]
    val_bytes = result["context"]["val_bytes"]
    assert rows[0]["val_tokens"] == val_bytes
    for row in rows:
        assert row["bytes_per_token"] == val_bytes / row["val_tokens"]
        assert row["compression_vs_baseline"] == rows[0]["val_tokens"] / row["val_tokens"]
        assert row["train_time_ns"] > 0
    assert set(result["probes"]) == {"English", "Bengali", "Hindi"}
    for probe in result["probes"].values():
        assert probe["bytes_per_token"] == len(probe["text"].encode()) / probe["tokens"]
        assert probe["tokens_per_character"] == probe["tokens"] / len(probe["text"])
    report = tmp_path / "docs/tokenizer_study.md"
    assert "Interpretation" in report.read_text(encoding="utf-8")
    assert mpimg.imread(tmp_path / "docs/img/tokenizer_study.png").std() > 0
    report.unlink()
    calls.clear()
    study.main(args)
    assert not calls and report.exists()
    raw.write_bytes(raw.read_bytes() + b"changed")
    study.main(args)
    assert calls == [257, 259, 262]
