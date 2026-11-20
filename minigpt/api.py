"""HTTP interface for locally inspecting one decoder and its training vocabulary."""

from __future__ import annotations

import asyncio
import codecs
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from threading import Event, Lock
from time import perf_counter
from typing import Annotated, Any, Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from minigpt.generate import StepInfo, generate_stream
from minigpt.model import GPT
from minigpt.tokenizer import BPETokenizer
from minigpt.tokenizer.bpe import EOT_ID

MAX_NEW_TOKENS = 256
Text = Annotated[str, Field(min_length=1, max_length=65536)]
Head = Annotated[int, Field(ge=0)] | Literal["mean"]


class GenerationRequest(BaseModel):
    """Sampling parameters for a bounded continuation."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    prompt: Text
    temperature: float = Field(default=1, ge=0, le=5)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float = Field(default=1, gt=0, le=1)
    max_new_tokens: int = Field(default=32, ge=1, le=MAX_NEW_TOKENS)
    seed: int = Field(default=42, ge=0, le=2**63 - 1)
    use_cache: bool = True


class Selection(BaseModel):
    """Zero-based layer and head, with an arithmetic mean over heads available."""

    model_config = ConfigDict(extra="forbid", strict=True)
    layer: int = Field(default=0, ge=0)
    head: Head = "mean"


class TraceRequest(GenerationRequest, Selection):
    """Expose sampling evidence and attention at each draw."""


class AttentionRequest(Selection):
    """Inspect all query positions in the given text."""

    text: Text


def _candidate(step: StepInfo, token: dict[str, Any]) -> dict[str, Any]:
    idx = token["id"]
    return {
        **token,
        "raw_probability": float(step.raw_distribution[idx]),
        "final_probability": float(step.final_distribution[idx]),
    }


def create_app(
    model: GPT | None,
    tokenizer: BPETokenizer | None,
    device: torch.device | str,
    *,
    checkpoint: str | Path | None = None,
) -> FastAPI:
    """Construct an isolated app. Its inference workers serialize access to the shared model."""
    app = FastAPI(title="MiniGPT", docs_url=None, redoc_url=None)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        """Report field errors even when the rejected input cannot itself be JSON-encoded."""
        detail = [{k: e[k] for k in ("loc", "msg", "type")} for e in exc.errors()]
        return Response(
            json.dumps({"detail": detail}), status_code=422, media_type="application/json"
        )

    lock = Lock()
    device = torch.device(device)
    vocabulary = []
    if model is not None and tokenizer is not None:
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError("Tokenizer and model vocabulary sizes differ")
        model.to(device).eval()
        for idx in range(tokenizer.vocab_size):
            octets = tokenizer.decode_bytes([idx])
            vocabulary.append(
                {
                    "id": idx,
                    "display": octets.decode("utf-8", "backslashreplace"),
                    "bytes_hex": octets.hex(),
                }
            )

    def loaded() -> tuple[GPT, BPETokenizer]:
        if model is None or tokenizer is None:
            raise HTTPException(503, "No model loaded. Start the server with --checkpoint.")
        return model, tokenizer

    def encode(text: str, reserved: int = 0) -> list[int]:
        net, codec = loaded()
        ids = codec.encode(text)
        if len(ids) + reserved > net.config.block_size:
            raise HTTPException(
                400,
                f"Input has {len(ids)} tokens and requests {reserved} new "
                f"tokens. The model block_size is {net.config.block_size}.",
            )
        return ids

    def check_selection(selection: Selection) -> None:
        cfg = loaded()[0].config
        if selection.layer >= cfg.n_layer:
            raise HTTPException(400, f"layer must be below {cfg.n_layer}")
        if selection.head != "mean" and selection.head >= cfg.n_head:
            raise HTTPException(400, f"head must be 'mean' or below {cfg.n_head}")

    def prepare(body: GenerationRequest) -> list[int]:
        net, _ = loaded()
        if body.top_k is not None and body.top_k > net.config.vocab_size:
            raise HTTPException(400, f"top_k exceeds the vocabulary size {net.config.vocab_size}")
        if isinstance(body, Selection):
            check_selection(body)
        return encode(body.prompt, body.max_new_tokens)

    def steps(body: GenerationRequest, ids: list[int]) -> Iterator[StepInfo]:
        net, _ = loaded()
        tracing = isinstance(body, TraceRequest)
        return generate_stream(
            net,
            ids,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            seed=body.seed,
            use_cache=body.use_cache,
            eos_id=EOT_ID,
            attention_layer=body.layer if tracing else None,
            attention_head=body.head if tracing and body.head != "mean" else None,
        )

    def events(
        body: TraceRequest, ids: list[int], stopped: Event | None = None
    ) -> Iterator[dict[str, Any]]:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        history = ids.copy()
        source = steps(body, ids)
        elapsed = 0.0
        for index in range(body.max_new_tokens):
            with lock:
                if stopped is not None and stopped.is_set():
                    return
                start = perf_counter()
                step = next(source)
                elapsed += perf_counter() - start
            idx = step.token_id
            last = index + 1 == body.max_new_tokens or idx == EOT_ID
            chosen = _candidate(step, vocabulary[idx])
            candidates = [
                _candidate(step, vocabulary[i]) for i, _ in step.top_candidates(10, raw=True)
            ]
            # The panel shows a shortlist. Account for the rest without renormalizing the bars.
            yield {
                "index": index,
                "chosen": chosen,
                "candidates": candidates,
                "raw_other": max(0.0, 1 - sum(c["raw_probability"] for c in candidates)),
                "final_other": max(0.0, 1 - sum(c["final_probability"] for c in candidates)),
                "raw_entropy": step.raw_entropy,
                "final_entropy": step.final_entropy,
                "context_tokens": [vocabulary[i] for i in history],
                "attention": step.attention_row.tolist(),
                "text": decoder.decode(bytes.fromhex(chosen["bytes_hex"]), final=last),
                "tokens_per_second": (index + 1) / elapsed,
                "finished": last,
            }
            if last:
                return
            history.append(idx)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("minigpt").joinpath("web", "index.html").read_text(encoding="utf-8")

    @app.get("/favicon.ico", status_code=204)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/model")
    def metadata() -> dict[str, Any]:
        net, codec = loaded()
        return {
            "config": asdict(net.config),
            "params": net.count_params(),
            "non_embedding_params": net.count_params(non_embedding=True),
            "device": str(net.token_embedding.weight.device),
            "vocab_size": codec.vocab_size,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "max_new_tokens": MAX_NEW_TOKENS,
        }

    @app.post("/generate")
    def generate(body: GenerationRequest) -> dict[str, Any]:
        ids = prepare(body)
        _, codec = loaded()
        with lock:
            start = perf_counter()
            new = [step.token_id for step in steps(body, ids)]
            elapsed = perf_counter() - start
        return {
            "text": codec.decode(new),
            "token_ids": new,
            "tokens_per_second": len(new) / elapsed,
        }

    @app.post("/trace")
    def trace(body: TraceRequest) -> dict[str, Any]:
        ids = prepare(body)
        return {"steps": list(events(body, ids))}

    @app.post("/generate/stream")
    async def stream(body: TraceRequest, request: Request) -> StreamingResponse:
        ids = await asyncio.to_thread(prepare, body)

        async def frames() -> AsyncIterator[str]:
            stopped = Event()
            source = events(body, ids, stopped)
            try:
                while not await request.is_disconnected():
                    # Pull one step at a time. The worker may be waiting on another request.
                    event = await asyncio.to_thread(next, source, None)
                    if event is None:
                        break
                    yield "event: token\ndata: " + json.dumps(event, allow_nan=False) + "\n\n"
                    if event["finished"]:
                        break
            finally:
                stopped.set()

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/attention")
    def attention(body: AttentionRequest) -> dict[str, Any]:
        check_selection(body)
        ids = encode(body.text)
        net, _ = loaded()
        with lock, torch.inference_mode():
            maps = net(torch.tensor([ids], device=device), return_attn=True)[2][body.layer][0]
            matrix = maps.mean(0) if body.head == "mean" else maps[body.head]
            return {"tokens": [vocabulary[i] for i in ids], "matrix": matrix.tolist()}

    return app
