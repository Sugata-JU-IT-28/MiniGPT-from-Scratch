"""A decoder-only Transformer with independently selectable architecture components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LayerKV = tuple[Tensor, Tensor]
KVCache = tuple[LayerKV, ...]

PRESETS = {
    "baseline": {"norm_type": "layernorm", "pos_encoding": "learned", "mlp_type": "gelu"},
    "modern": {"norm_type": "rmsnorm", "pos_encoding": "rope", "mlp_type": "swiglu"},
}


@dataclass(frozen=True)
class GPTConfig:
    """Dimensions, regularization, and architecture of a causal language model."""

    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True
    use_sdpa: bool = False
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    pos_encoding: Literal["learned", "rope"] = "learned"
    mlp_type: Literal["gelu", "swiglu"] = "gelu"

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> GPTConfig:
        """Expand a named preset, allowing individual fields to override it."""
        if name not in PRESETS:
            raise ValueError(f"Unknown architecture preset {name!r}")
        return cls(**(PRESETS[name] | overrides))

    def __post_init__(self) -> None:
        dims = (self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd)
        if any(type(d) is not int or d <= 0 for d in dims):
            raise ValueError("Model dimensions must be positive integers")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for field in ("norm_type", "pos_encoding", "mlp_type"):
            if getattr(self, field) not in {p[field] for p in PRESETS.values()}:
                raise ValueError(f"Unknown {field} {getattr(self, field)!r}")
        if self.pos_encoding == "rope":
            assert (self.n_embd // self.n_head) % 2 == 0, "RoPE requires an even head dimension"


class RMSNorm(nn.Module):
    """Normalize by the root mean square in float32, then restore the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Scale each token independently across its features."""
        value = x.float()
        normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


def _norm(config: GPTConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd)
    return nn.LayerNorm(config.n_embd, bias=config.bias)


def rotary_tables(block_size: int, head_dim: int) -> tuple[Tensor, Tensor]:
    """Precompute the cosine and sine for each position and adjacent feature pair."""
    assert head_dim % 2 == 0, "RoPE requires an even head dimension"
    frequencies = 10000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(block_size, dtype=torch.float32), frequencies)
    return angles.cos(), angles.sin()


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> Tensor:
    """Rotate [B,H,T,D] using absolute positions shaped [T] or [B,T]."""
    assert x.size(-1) % 2 == 0, "RoPE requires an even head dimension"
    if positions.ndim not in (1, 2) or positions.size(-1) != x.size(-2):
        raise ValueError("positions must contain one absolute index for each time step")
    c = cos[positions].unsqueeze(-3).to(x.dtype)
    s = sin[positions].unsqueeze(-3).to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    """Fused QKV projections with a masked manual path and optional native SDPA."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_sdpa = config.use_sdpa
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.ones(config.block_size, config.block_size, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask, persistent=False)
        cos, sin = (
            rotary_tables(config.block_size, self.head_dim)
            if config.pos_encoding == "rope"
            else (None, None)
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        return_attn: bool = False,
        past_kv: LayerKV | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, LayerKV]:
        """Return projected values and, when requested, pre-dropout attention probabilities."""
        batch, steps, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, steps, 3, self.n_head, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.cos is not None:
            q = apply_rotary_emb(q, self.cos, self.sin, positions)
            k = apply_rotary_emb(k, self.cos, self.sin, positions)

        past_len = 0
        if past_kv is not None:
            old_k, old_v = past_kv
            if old_k.ndim != 4 or old_k.shape != old_v.shape:
                raise ValueError("past_kv must contain matching [B,H,T,D] keys and values")
            past_len = old_k.size(2)
            if old_k.shape != (batch, self.n_head, past_len, self.head_dim):
                raise ValueError("past_kv batch, head count or head dimension mismatch")
            if any(t.device != k.device or t.dtype != k.dtype for t in past_kv):
                raise ValueError("past_kv dtype and device must match current projections")
            k = torch.cat((old_k, k), dim=2)
            v = torch.cat((old_v, v), dim=2)
        elif use_cache:
            # Own compact storage; views into fused QKV would also keep discarded Q alive.
            k = k.clone(memory_format=torch.contiguous_format)
            v = v.clone(memory_format=torch.contiguous_format)
        total = past_len + steps
        if total > self.causal_mask.size(0):
            raise ValueError("Cached and new tokens exceed block_size")
        mask = self.causal_mask[past_len:total, :total]

        weights = None
        if not self.use_sdpa or return_attn:
            scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = scores.softmax(dim=-1)
        if self.use_sdpa:
            mixed = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                attn_mask=mask if past_len else None,
                is_causal=not past_len,
            )
        else:
            assert weights is not None
            mixed = self.attn_dropout(weights) @ v
        mixed = mixed.transpose(1, 2).reshape(batch, steps, width)
        result = (self.resid_dropout(self.proj(mixed)), weights if return_attn else None)
        return (*result, (k, v)) if use_cache else result


class MLP(nn.Module):
    """Use 4d GELU features or ceil((8d/3)/8)*8 gated features for similar weight counts."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        dim = config.n_embd
        gated = config.mlp_type == "swiglu"
        # Three matrices at 8d/3 match two matrices at 4d, before alignment.
        hidden = 8 * math.ceil(dim / 3) if gated else 4 * dim
        self.gate = nn.Linear(dim, hidden, bias=config.bias) if gated else None
        self.up = nn.Linear(dim, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Mix features independently at each position."""
        up = self.up(x)
        hidden = F.gelu(up) if self.gate is None else F.silu(self.gate(x)) * up
        return self.dropout(self.down(hidden))


class Block(nn.Module):
    """Apply pre-normalized attention and MLP residual updates."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = _norm(config)
        self.attn = CausalSelfAttention(config)
        self.norm2 = _norm(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        return_attn: bool = False,
        past_kv: LayerKV | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, LayerKV]:
        """Preserve the attention map when the caller requests it."""
        result = self.attn(self.norm1(x), positions, return_attn, past_kv, use_cache)
        x = x + result[0]
        return (x + self.mlp(self.norm2(x)), *result[1:])


class GPT(nn.Module):
    """A causal decoder returning logits, optional mean next-token loss, and optional maps."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.pos_encoding == "learned"
            else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = _norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, std=residual_std)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def count_params(self, non_embedding: bool = False) -> int:
        """Count unique parameters, optionally excluding token and position embedding tables."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embedding.weight.numel()
            if self.position_embedding is not None:
                total -= self.position_embedding.weight.numel()
        return total

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        return_attn: bool = False,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> (
        tuple[Tensor, Tensor | None]
        | tuple[Tensor, Tensor | None, list[Tensor]]
        | tuple[Tensor, Tensor | None, KVCache]
        | tuple[Tensor, Tensor | None, list[Tensor], KVCache]
    ):
        """Return logits and loss, optionally followed by attention maps and the extended cache."""
        if idx.ndim != 2 or not 0 < idx.size(1) <= self.config.block_size:
            raise ValueError("idx must have shape [batch, time] within the context window")
        past_len = 0
        if past_kv is not None:
            if len(past_kv) != len(self.blocks) or any(k.ndim != 4 for k, _ in past_kv):
                raise ValueError("past_kv needs one [B,H,T,D] key/value pair per layer")
            past_len = past_kv[0][0].size(2)
            if any(k.size(2) != past_len for k, _ in past_kv):
                raise ValueError("All cache layers must have the same past length")
        total = past_len + idx.size(1)
        if total > self.config.block_size:
            raise ValueError("Cached and new tokens exceed block_size")
        positions = torch.arange(past_len, total, device=idx.device)
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            x = x + self.position_embedding(positions)
        x = self.dropout(x)
        attention = []
        cache: list[LayerKV] = []
        for layer, block in enumerate(self.blocks):
            result = block(
                x, positions, return_attn, None if past_kv is None else past_kv[layer], use_cache
            )
            x, weights = result[:2]
            if use_cache:
                cache.append(result[2])
            if weights is not None:
                attention.append(weights)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx")
            loss = F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1))
        if return_attn:
            return (
                (logits, loss, attention, tuple(cache)) if use_cache else (logits, loss, attention)
            )
        return (logits, loss, tuple(cache)) if use_cache else (logits, loss)
