"""Exercise both decoder presets through HTTP without repository data or checkpoints."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Event
from typing import Any

import pytest
import torch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from torch import Tensor

from minigpt import serve
from minigpt.api import MAX_NEW_TOKENS, create_app
from minigpt.checkpoint import save_checkpoint
from minigpt.generate import distributions
from minigpt.io import file_sha256
from minigpt.model import GPT, GPTConfig
from minigpt.tokenizer import BPETokenizer
from minigpt.tokenizer.bpe import EOT_ID
from scripts import smoke


@pytest.fixture(params=["baseline", "modern"])
def net(request: pytest.FixtureRequest) -> GPT:
    """A fresh CPU model with a byte-only vocabulary and deterministic dropout-free inference."""
    torch.manual_seed(17)
    return GPT(
        GPTConfig.from_preset(
            request.param, vocab_size=257, block_size=16, n_layer=2, n_head=2, n_embd=16
        )
    )


@pytest.fixture
def client(net: GPT) -> Iterator[TestClient]:
    with TestClient(create_app(net, BPETokenizer(), "cpu")) as client:
        yield client


def frames(client: TestClient, body: dict[str, Any]) -> list[dict[str, Any]]:
    with client.stream("POST", "/generate/stream", json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = list(response.iter_lines())
    assert lines[::3] == ["event: token"] * (len(lines) // 3)
    assert lines[2::3] == [""] * (len(lines) // 3)
    return [json.loads(line.removeprefix("data: ")) for line in lines[1::3]]


def test_static_and_metadata(client: TestClient, net: GPT) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "Token inspector" in page.text and re.search(r"https?://", page.text, re.I) is None
    assert "innerHTML" not in page.text
    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 204 and not favicon.content
    assert client.get("/model").json() == {
        "config": asdict(net.config),
        "params": net.count_params(),
        "non_embedding_params": net.count_params(True),
        "device": "cpu",
        "vocab_size": 257,
        "checkpoint": None,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def test_generate_stream_trace_agree(client: TestClient, net: GPT) -> None:
    body = {
        "prompt": "To ",
        "top_k": 4,
        "top_p": 0.85,
        "temperature": 0.8,
        "max_new_tokens": 6,
        "seed": 123,
    }
    plain = client.post("/generate", json=body).json()
    ids = plain["token_ids"]
    assert plain["tokens_per_second"] > 0 and plain["text"] == BPETokenizer().decode(ids)
    assert len(ids) == 6 or ids[-1] == EOT_ID
    for use_cache in (True, False):
        params = dict(body, use_cache=use_cache)
        assert client.post("/generate", json=params).json()["token_ids"] == ids
        inspected = dict(params, layer=1, head=1)
        steps = client.post("/trace", json=inspected).json()["steps"]
        streamed = frames(client, inspected)
        assert [s["chosen"]["id"] for s in streamed] == ids == [s["chosen"]["id"] for s in steps]
        assert [s["finished"] for s in streamed] == [False] * (len(ids) - 1) + [True]
        context = BPETokenizer().encode(body["prompt"])
        for s, event in zip(steps, streamed, strict=True):
            assert {k: v for k, v in s.items() if k != "tokens_per_second"} == {
                k: v for k, v in event.items() if k != "tokens_per_second"
            }
            assert [t["id"] for t in s["context_tokens"]] == context
            with torch.no_grad():
                logits, _, maps = net(torch.tensor([context]), return_attn=True)
                raw, final = distributions(logits[0, -1], temperature=0.8, top_k=4, top_p=0.85)
            assert s["attention"] == pytest.approx(maps[1][0, 1, -1].tolist(), abs=1e-6)
            candidates = s["candidates"]
            assert [c["id"] for c in candidates] == raw.argsort(descending=True, stable=True)[
                :10
            ].tolist()
            for c in [*candidates, s["chosen"]]:
                assert c["bytes_hex"] == BPETokenizer().decode_bytes([c["id"]]).hex()
                assert isinstance(c["display"], str)
                assert c["raw_probability"] == pytest.approx(float(raw[c["id"]]), abs=1e-7)
                assert c["final_probability"] == pytest.approx(float(final[c["id"]]), abs=1e-6)
            assert s["raw_entropy"] == pytest.approx(float(torch.special.entr(raw).sum()))
            assert s["final_entropy"] == pytest.approx(float(torch.special.entr(final).sum()))
            for name in ("raw", "final"):
                assert sum(c[name + "_probability"] for c in candidates) + s[
                    name + "_other"
                ] == pytest.approx(1)
            context.append(s["chosen"]["id"])


@pytest.mark.parametrize("head", [0, 1, "mean"])
def test_attention(client: TestClient, net: GPT, head: int | str) -> None:
    body = {"text": "Rose", "layer": 1, "head": head}
    response = client.post("/attention", json=body).json()
    ids = [t["id"] for t in response["tokens"]]
    assert ids == list(b"Rose")
    matrix = torch.tensor(response["matrix"])
    with torch.no_grad():
        maps = net(torch.tensor([ids]), return_attn=True)[2][1][0]
    torch.testing.assert_close(matrix, maps.mean(0) if head == "mean" else maps[head])
    assert matrix.shape == (4, 4) and matrix.triu(1).count_nonzero() == 0
    torch.testing.assert_close(matrix.sum(1), torch.ones(4))
    step = client.post(
        "/trace", json={"prompt": "Rose", "max_new_tokens": 1, "layer": 1, "head": head}
    ).json()["steps"][0]
    assert step["attention"] == pytest.approx(matrix[-1].tolist())


def test_utf8_across_tokens_and_terminal_flush(client: TestClient, net: GPT) -> None:
    sequence = {ord("A"): 0xE2, 0xE2: 0x82, 0x82: 0xAC, 0xAC: EOT_ID}

    def force_bytes(
        module: GPT, inputs: tuple[Tensor, ...], result: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        logits = torch.full_like(result[0], -100)
        logits[0, -1, sequence[int(inputs[0][0, -1])]] = 100
        return (logits, *result[1:])

    with net.register_forward_hook(force_bytes):
        body = {"prompt": "A", "temperature": 0, "max_new_tokens": 5}
        parts = frames(client, body)
        assert [p["text"] for p in parts] == ["", "", "€", "<|endoftext|>"]
        assert [p["chosen"]["bytes_hex"] for p in parts[:3]] == ["e2", "82", "ac"]
        assert parts[0]["chosen"]["display"] == r"\xe2"
        assert (
            "".join(p["text"] for p in parts) == client.post("/generate", json=body).json()["text"]
        )
        cut = frames(client, dict(body, max_new_tokens=2))
        assert [p["text"] for p in cut] == ["", "\ufffd"]


@pytest.mark.parametrize(
    "override",
    [
        {"temperature": -1},
        {"temperature": 5.1},
        {"top_k": 0},
        {"top_k": 258},
        {"top_p": 0},
        {"top_p": 1.1},
        {"max_new_tokens": 0},
        {"max_new_tokens": MAX_NEW_TOKENS + 1},
        {"seed": -1},
        {"prompt": ""},
        {"prompt": "z" * 17},
        {"prompt": "x" * 16},
        {"top_k": True},
    ],
)
def test_generation_validation(client: TestClient, override: dict[str, Any]) -> None:
    for path in ("/generate", "/generate/stream", "/trace"):
        res = client.post(path, json={"prompt": "x", "max_new_tokens": 1, **override})
        assert res.status_code in (400, 422) and res.json()["detail"]


def test_unencodable_input_has_field_errors(client: TestClient) -> None:
    for body, field in (
        ({"prompt": "\ud800"}, "prompt"),
        ({"prompt": "a", "temperature": float("inf")}, "temperature"),
    ):
        response = client.post(
            "/generate", content=json.dumps(body), headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", field]


@pytest.mark.parametrize("override", [{"layer": -1}, {"layer": 2}, {"head": 2}, {"head": "all"}])
def test_selection_validation(client: TestClient, override: dict[str, Any]) -> None:
    for path in ("/attention", "/trace", "/generate/stream"):
        body = {"text": "a"} if path == "/attention" else {"prompt": "a", "max_new_tokens": 1}
        assert client.post(path, json=body | override).status_code in (400, 422)
    for value in ("", "x" * 17):
        assert client.post("/attention", json={"text": value}).status_code in (400, 422)


def test_no_model() -> None:
    with TestClient(create_app(None, None, "cpu")) as client:
        assert client.get("/").is_success and client.get("/model").status_code == 503
        for path in ("/generate", "/generate/stream", "/trace", "/attention"):
            res = client.post(path, json={"text": "a"} if path == "/attention" else {"prompt": "a"})
            assert res.status_code == 503 and "checkpoint" in res.json()["detail"]


def test_shared_model_and_responsive_event_loop(client: TestClient, net: GPT) -> None:
    entered, release, overlapping = Event(), Event(), Event()

    def pause(module: GPT, inputs: tuple[Tensor, ...]) -> None:
        if entered.is_set() and not release.is_set():
            overlapping.set()
        entered.set()
        assert release.wait(10)

    with net.register_forward_pre_hook(pause), ThreadPoolExecutor(3) as pool:
        stream = pool.submit(frames, client, {"prompt": "a", "max_new_tokens": 1})
        try:
            assert entered.wait(5)
            attention = pool.submit(client.post, "/attention", json={"text": "ab"})
            assert pool.submit(client.get, "/model").result(timeout=3).is_success
            assert not overlapping.wait(0.1)
        finally:
            release.set()
        assert len(stream.result(timeout=5)) == 1
        assert attention.result(timeout=5).is_success


def test_disconnect_stops_inference(
    client: TestClient, net: GPT, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    async def disconnected(request: Request) -> bool:
        return bool(calls)

    monkeypatch.setattr(Request, "is_disconnected", disconnected)
    with net.register_forward_hook(lambda *args: calls.append(1)):
        assert len(frames(client, {"prompt": "a", "max_new_tokens": 8})) == 1
    assert calls == [1]


def store_checkpoint(net: GPT, directory: Path) -> Path:
    vocab = directory / "tokenizer.json"
    BPETokenizer().save(vocab)
    path = directory / "test.pt"
    save_checkpoint(
        path,
        {
            "config": {
                "model": asdict(net.config),
                "train": {"cpu_threads": 1},
                "data_dir": str(directory),
            },
            "model": net.state_dict(),
            "data_sha256": {"tokenizer.json": file_sha256(vocab)},
        },
    )
    return path


def test_serve_loads_saved_config(
    net: GPT, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store_checkpoint(net, tmp_path)

    def run(app: FastAPI, host: str, port: int) -> None:
        with TestClient(app) as web:
            meta = web.get("/model").json()
        assert meta["checkpoint"] == str(path) and meta["config"] == asdict(net.config)
        assert port == 9000 and host == "127.0.0.1"

    monkeypatch.setattr(serve.uvicorn, "run", run)
    monkeypatch.setenv("MINIGPT_CHECKPOINT", str(path))
    serve.main(["--port", "9000"])


@pytest.mark.slow
def test_smoke_script(net: GPT, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = store_checkpoint(net, tmp_path)
    output = tmp_path / "smoke.json"
    smoke.main(["--checkpoint", str(path), "--output", str(output), "--seed", "17"])
    assert json.loads(output.read_text())["server_stopped"]
