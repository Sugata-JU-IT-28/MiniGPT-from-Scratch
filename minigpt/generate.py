"""Autoregressive decoding with inspectable distributions and explicit context limits."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from minigpt.checkpoint import load_checkpoint
from minigpt.data import default_device
from minigpt.io import file_sha256
from minigpt.model import GPT, GPTConfig, KVCache
from minigpt.tokenizer import BPETokenizer

WindowPolicy = Literal["error", "reprefill"]


def load_model(path: Path, tokenizer_path: Path | None = None) -> tuple[GPT, BPETokenizer]:
    """Restore the decoder and the exact training vocabulary."""
    state = load_checkpoint(path)
    cfg = state["config"]
    vocab = tokenizer_path or Path(cfg["data_dir"]) / "tokenizer.json"
    if file_sha256(vocab) != state["data_sha256"]["tokenizer.json"]:
        raise ValueError("The tokenizer differs from the checkpoint's training tokenizer")
    device = default_device()
    if device.type == "cpu" and cfg["train"]["cpu_threads"] is not None:
        torch.set_num_threads(cfg["train"]["cpu_threads"])
    model = GPT(GPTConfig(**cfg["model"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), BPETokenizer.load(vocab)


@dataclass(frozen=True)
class StepInfo:
    """A sampled token with vocabulary-indexed probabilities and entropies in nats."""

    token_id: int
    raw_distribution: Tensor
    final_distribution: Tensor
    raw_entropy: float
    final_entropy: float
    past_kv: KVCache | None = None
    attention_row: Tensor | None = None

    def top_candidates(self, k: int = 5, *, raw: bool = False) -> list[tuple[int, float]]:
        """Return (token ID, probability) pairs, breaking ties by ascending token ID."""
        if k < 1:
            raise ValueError("k must be positive")
        probs = self.raw_distribution if raw else self.final_distribution
        ranks = torch.argsort(probs, descending=True, stable=True)[:k]
        return list(zip(ranks.tolist(), probs[ranks].tolist(), strict=True))


def penalize_repetitions(logits: Tensor, history: Sequence[int], penalty: float) -> Tensor:
    """Apply the CTRL sign-dependent penalty once per distinct seen token."""
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("repetition_penalty must be finite and positive")
    if penalty == 1 or not len(history):
        return logits
    ids = torch.tensor(sorted(set(history)), device=logits.device)
    adjusted = logits.clone()
    seen = logits[ids]
    adjusted[ids] = torch.where(seen < 0, seen * penalty, seen / penalty)
    return adjusted


def distributions(
    logits: Tensor,
    history: Sequence[int] = (),
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Apply repetition, temperature, exact top-k, minimal nucleus, then normalize."""
    if logits.ndim != 1 or not logits.numel():
        raise ValueError("Expected one vocabulary vector")
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Require finite temperature >= 0 and 0 < top_p <= 1")
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    scores = logits if logits.dtype == torch.float64 else logits.float()
    raw = scores.softmax(0)
    scores = penalize_repetitions(scores, history, repetition_penalty)
    if temperature <= torch.finfo(scores.dtype).tiny:
        probs = torch.zeros_like(scores)
        probs[scores.argmax()] = 1
        return raw, probs
    scores = (scores - scores.max()) / temperature
    if (top_k is not None and top_k < len(scores)) or top_p < 1:
        order = torch.argsort(scores, descending=True, stable=True)
        keep = order[:top_k]
        selected = scores[keep]
        if top_p < 1:
            mass = selected.softmax(0, dtype=torch.float64).cumsum(0)
            # Include the first token crossing the threshold; equal scores retain ID order.
            n = int(torch.searchsorted(mass, top_p, right=False).item()) + 1
            keep, selected = keep[:n], selected[:n]
        scores = torch.full_like(scores, -torch.inf).scatter_(0, keep, selected)
    return raw, scores.softmax(0)


def _ids(input_ids: Sequence[int] | Tensor) -> list[int]:
    if isinstance(input_ids, Tensor):
        if input_ids.ndim == 2 and input_ids.size(0) == 1:
            input_ids = input_ids[0]
        if input_ids.ndim != 1:
            raise ValueError("Sampling accepts one token sequence")
        return input_ids.tolist()
    return list(input_ids)


@torch.inference_mode()
def next_token_distribution(
    model: GPT,
    input_ids: Sequence[int] | Tensor,
    past_kv: KVCache | None = None,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seed: int = 42,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    history: Sequence[int] | None = None,
    attention_layer: int | None = None,
    attention_head: int | None = None,
) -> StepInfo:
    """Inspect one draw. With a cache, supply only the new suffix and full history for penalties."""
    ids = _ids(input_ids)
    if not ids:
        raise ValueError("input_ids must contain a prompt token")
    if past_kv is not None and repetition_penalty != 1 and history is None:
        raise ValueError("A cached repetition penalty needs the full token history")
    device = model.token_embedding.weight.device
    training = model.training
    if training:
        model.eval()
    try:
        result = model(
            torch.tensor([ids], dtype=torch.long, device=device),
            past_kv=past_kv,
            use_cache=use_cache,
            return_attn=attention_layer is not None,
        )
    finally:
        if training:
            model.train()
    raw, final = distributions(
        result[0][0, -1],
        ids if history is None else history,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(seed)
    chosen = (
        final.argmax() if temperature == 0 else torch.multinomial(final, 1, generator=generator)
    )
    row = None
    if attention_layer is not None:
        rows = result[2][attention_layer][0, :, -1, :]
        row = rows.mean(0) if attention_head is None else rows[attention_head].clone()
    return StepInfo(
        int(chosen.item()),
        raw,
        final,
        float(torch.special.entr(raw).sum()),
        float(torch.special.entr(final).sum()),
        result[-1] if use_cache else None,
        row,
    )


def generate_stream(
    model: GPT,
    input_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    seed: int = 42,
    use_cache: bool = True,
    window_policy: WindowPolicy = "error",
    attention_layer: int | None = None,
    attention_head: int | None = None,
) -> Iterator[StepInfo]:
    """Yield new tokens only. Re-prefilling reproduces a fresh forward on the cropped window."""
    history = _ids(input_ids)
    block = model.config.block_size
    if not history or max_new_tokens < 0 or window_policy not in ("error", "reprefill"):
        raise ValueError("Invalid prompt, max_new_tokens, or window_policy")
    if window_policy == "error" and len(history) + max_new_tokens > block:
        raise ValueError(
            "Prompt plus max_new_tokens exceeds block_size; use window_policy='reprefill'"
        )
    generator = torch.Generator(device=model.token_embedding.weight.device).manual_seed(seed)
    cache = None
    for _ in range(max_new_tokens):
        if cache is not None and cache[0][0].size(2) >= block:
            cache = None
        suffix = history[-block:] if cache is None else history[-1:]
        info = next_token_distribution(
            model,
            suffix,
            cache,
            generator=generator,
            history=history,
            use_cache=use_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            attention_layer=attention_layer,
            attention_head=attention_head,
        )
        cache = info.past_kv
        history.append(info.token_id)
        # Streaming clients can retain every step without retaining every prefix's KV tensors.
        yield StepInfo(
            info.token_id,
            info.raw_distribution,
            info.final_distribution,
            info.raw_entropy,
            info.final_entropy,
            attention_row=info.attention_row,
        )
        if info.token_id == eos_id:
            return


def generate(
    model: GPT,
    input_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    seed: int = 42,
    use_cache: bool = True,
    window_policy: WindowPolicy = "error",
) -> list[int]:
    """Return generated IDs, excluding the prompt and including a sampled EOS."""
    return [
        step.token_id
        for step in generate_stream(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_id=eos_id,
            seed=seed,
            use_cache=use_cache,
            window_policy=window_policy,
        )
    ]
