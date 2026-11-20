"""Architecture invariants across all normalization, position, and MLP combinations."""

import math
from dataclasses import replace
from itertools import product

import pytest
import torch
from torch import Tensor

from minigpt.model import GPT, MLP, GPTConfig, RMSNorm, apply_rotary_emb, rotary_tables
from minigpt.train import accumulate_gradients, build_optimizer

FLAGS = list(product(("layernorm", "rmsnorm"), ("learned", "rope"), ("gelu", "swiglu")))


@pytest.fixture(params=FLAGS, ids=["-".join(flags) for flags in FLAGS])
def config(request: pytest.FixtureRequest) -> GPTConfig:
    """A small model with independent toggles and an even per-head dimension."""
    norm, pos, mlp = request.param
    return GPTConfig(
        vocab_size=97,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=48,
        dropout=0.0,
        norm_type=norm,
        pos_encoding=pos,
        mlp_type=mlp,
    )


def test_shapes_initial_loss_causality_and_maps(config: GPTConfig) -> None:
    torch.manual_seed(113)
    model = GPT(config).eval()
    x = torch.randint(config.vocab_size, (4, 16))
    y = torch.randint(config.vocab_size, x.shape)
    logits, loss, weights = model(x, y, return_attn=True)
    assert logits.shape == (4, 16, config.vocab_size)
    assert loss is not None and abs(loss.item() / math.log(config.vocab_size) - 1) <= 0.05
    assert len(weights) == config.n_layer
    for attention in weights:
        assert attention.shape == (4, config.n_head, 16, 16)
        assert torch.count_nonzero(attention.triu(1)) == 0
        torch.testing.assert_close(
            attention.sum(-1), torch.ones_like(attention[..., 0]), rtol=0, atol=1e-6
        )
    changed = x.clone()
    changed[:, 9] = (x[:, 9] + 1) % config.vocab_size
    later, _ = model(changed)
    torch.testing.assert_close(logits[:, :9], later[:, :9], rtol=0, atol=1e-6)
    assert not torch.equal(logits[:, 9], later[:, 9])
    names = dict(model.named_parameters())
    assert ("position_embedding.weight" in names) == (config.pos_encoding == "learned")


def test_manual_equals_sdpa_float32(config: GPTConfig) -> None:
    torch.manual_seed(27)
    manual = GPT(config).float().eval()
    native = GPT(replace(config, use_sdpa=True)).float().eval()
    native.load_state_dict(manual.state_dict())
    x = torch.randn(2, 9, config.n_embd)
    pos = torch.arange(3, 12)
    a, _ = manual.blocks[0].attn(x, pos)
    b, _ = native.blocks[0].attn(x, pos)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
    idx = torch.randint(config.vocab_size, (2, 15))
    a, _, maps_a = manual(idx, return_attn=True)
    b, _, maps_b = native(idx, return_attn=True)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
    for left, right in zip(maps_a, maps_b, strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=0)


def test_accumulated_gradients_match_one_batch(config: GPTConfig) -> None:
    torch.manual_seed(71)
    full, micro = GPT(config), GPT(config)
    micro.load_state_dict(full.state_dict())
    x, y = torch.randint(config.vocab_size, (2, 8, 13))
    loss = accumulate_gradients(full, [(x, y)], 1)
    micro_loss = accumulate_gradients(micro, zip(x.chunk(2), y.chunk(2), strict=True), 2)
    torch.testing.assert_close(loss, micro_loss, rtol=0, atol=1e-6)
    for a, b in zip(full.parameters(), micro.parameters(), strict=True):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize("tied", [False, True])
def test_tying_counts_and_optimizer_partition(config: GPTConfig, tied: bool) -> None:
    model = GPT(replace(config, tie_weights=tied))
    assert (model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()) == tied
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    ids = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters()}
    for group in opt.param_groups:
        assert group["betas"] == (0.9, 0.95)
        for param in group["params"]:
            assert group["weight_decay"] == (0.1 if param.ndim >= 2 else 0)
    excluded = sum(p.numel() for name, p in model.named_parameters() if "embedding" in name)
    assert model.count_params(True) == model.count_params() - excluded


def test_rotary_positions_relative_offsets_and_norm() -> None:
    torch.manual_seed(115)
    cos, sin = rotary_tables(24, 8)
    x = torch.randn(2, 3, 10, 8)
    full = apply_rotary_emb(x, cos, sin, torch.arange(10))
    tail = apply_rotary_emb(x[:, :, 5:], cos, sin, torch.arange(5, 10))
    torch.testing.assert_close(tail, full[:, :, 5:], atol=1e-6, rtol=0)
    torch.testing.assert_close(full.norm(dim=-1), x.norm(dim=-1), atol=1e-6, rtol=0)
    q, k = x[:, :, :1], x[:, :, 1:2]

    def dot(m: int, n: int) -> Tensor:
        qm = apply_rotary_emb(q, cos, sin, torch.tensor([m]))
        kn = apply_rotary_emb(k, cos, sin, torch.tensor([n]))
        return (qm * kn).sum(-1)

    torch.testing.assert_close(dot(7, 2), dot(13, 8), atol=2e-6, rtol=0)
    torch.testing.assert_close(dot(4, 9), dot(6, 11), atol=2e-6, rtol=0)
    positions = torch.stack([torch.arange(10), torch.arange(10, 20)])
    rows = apply_rotary_emb(x, cos, sin, positions)
    torch.testing.assert_close(rows[:1], full[:1])
    torch.testing.assert_close(rows[1:], apply_rotary_emb(x[1:], cos, sin, positions[1]))
    with pytest.raises(AssertionError, match="even"):
        GPTConfig.from_preset("modern", n_embd=15, n_head=3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_float32_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(4)
    x = (torch.randn(2, 5, 24) * 0.002).to(dtype)
    norm = RMSNorm(24, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.3, 1.5, 24))
    expected = norm.weight * x.float() / torch.sqrt(x.float().square().mean(-1, True) + norm.eps)
    assert norm(x).dtype == dtype
    torch.testing.assert_close(norm(x), expected.to(dtype), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("width", [192, 256, 384])
def test_swiglu_parameter_match_and_alignment(width: int) -> None:
    cfg = GPTConfig(n_embd=width, n_head=4)
    gelu, swiglu = MLP(cfg), MLP(replace(cfg, mlp_type="swiglu"))
    assert swiglu.up.out_features == math.ceil((8 * width / 3) / 8) * 8
    dense = sum(p.numel() for p in gelu.parameters())
    gated = sum(p.numel() for p in swiglu.parameters())
    assert abs(gated / dense - 1) < 0.02


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["baseline", "modern"])
def test_can_overfit_one_fixed_batch(preset: str) -> None:
    torch.manual_seed(22)
    cfg = GPTConfig.from_preset(
        preset, n_layer=2, n_head=4, n_embd=48, block_size=12, vocab_size=41, dropout=0.0
    )
    model = GPT(cfg)
    opt = build_optimizer(model, 0.008, 0.0)
    sequences = torch.stack([torch.randperm(41)[:13] for _ in range(4)])
    sequences[:, 0] = torch.arange(4)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        _, loss = model(sequences[:, :-1], sequences[:, 1:])
        assert loss is not None
        if loss.item() < 0.1:
            break
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
