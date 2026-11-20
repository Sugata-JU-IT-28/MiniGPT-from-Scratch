"""Position-wise cache equivalence, sampling laws, window boundaries and streamed diagnostics."""

import math
from dataclasses import replace

import pytest
import torch

from minigpt.generate import (
    distributions,
    generate,
    generate_stream,
    next_token_distribution,
    penalize_repetitions,
)
from minigpt.model import GPT, GPTConfig


@pytest.mark.parametrize("preset", ["baseline", "modern"])
@pytest.mark.parametrize("sdpa", [False, True])
def test_token_and_chunk_cache_against_full_forward(preset: str, sdpa: bool) -> None:
    torch.manual_seed(918)
    cfg = GPTConfig.from_preset(
        preset,
        vocab_size=71,
        block_size=64,
        n_head=4,
        n_layer=2,
        n_embd=32,
        dropout=0.0,
        use_sdpa=sdpa,
    )
    model = GPT(cfg).eval()
    ids = torch.randint(0, 71, (3, 64))
    with torch.inference_mode():
        logits, _, maps = model(ids, return_attn=True)
        for widths in ([7] + [1] * 57, [7, 17, 3, 20, 17]):
            cache, start = None, 0
            for width in widths:
                end = start + width
                out, _, weights, cache = model(
                    ids[:, start:end], past_kv=cache, use_cache=True, return_attn=True
                )
                torch.testing.assert_close(out, logits[:, start:end], rtol=0, atol=1e-4)
                for (k, v), got, full in zip(cache, weights, maps, strict=True):
                    assert k.shape == v.shape == (3, 4, end, 8)
                    assert got.shape == (3, 4, width, end)
                    torch.testing.assert_close(got, full[:, :, start:end, :end], rtol=0, atol=1e-5)
                    assert torch.count_nonzero(got.triu(start + 1)) == 0
                start = end
            assert start == 64
        with pytest.raises(ValueError, match="block_size"):
            model(ids[:, :1], past_kv=cache, use_cache=True)


def test_exact_top_k_and_minimal_nucleus() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert distributions(logits, top_k=2)[1].tolist() == [0.5, 0.5, 0, 0]
    for p, count in [(1.0, 4), (0.75, 3), (0.751, 4), (0.001, 1)]:
        probs = distributions(logits, top_p=p)[1]
        assert torch.nonzero(probs).flatten().tolist() == list(range(count))
    scores = torch.tensor([0.45, 0.30, 0.20, 0.05], dtype=torch.float64).log()
    assert torch.count_nonzero(distributions(scores, top_p=0.70)[1]) == 2
    # Truncate to k first, then recompute the nucleus on that renormalized support.
    assert torch.nonzero(distributions(scores, top_k=2, top_p=0.55)[1]).flatten().tolist() == [0]
    for temp in (0.0, 1e-12):
        assert distributions(scores, temperature=temp)[1].tolist() == [1, 0, 0, 0]


def test_repetition_penalty_precedes_temperature_and_filters() -> None:
    logits = torch.tensor([2.0, -1.0, 1.5, 0.5])
    seen = [0, 1, 0]
    changed = penalize_repetitions(logits, seen, 2)
    torch.testing.assert_close(changed, torch.tensor([1.0, -2.0, 1.5, 0.5]))
    raw, final = distributions(logits, seen, temperature=0.5, repetition_penalty=2)
    torch.testing.assert_close(raw, logits.softmax(0))
    torch.testing.assert_close(final, (changed / 0.5).softmax(0))
    unpenalized = distributions(logits, temperature=0.5)[1]
    assert final[0] < unpenalized[0] and final[1] < unpenalized[1]
    torch.testing.assert_close(final[2] / final[3], unpenalized[2] / unpenalized[3])
    assert distributions(logits, seen, temperature=0, repetition_penalty=2)[1].argmax() == 2


@pytest.mark.slow
def test_sampling_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    target = torch.tensor([0.08, 0.17, 0.3, 0.45], dtype=torch.float64)
    model = GPT(GPTConfig(vocab_size=4, block_size=1, n_embd=8, n_head=1, n_layer=1)).eval()
    logits = target.log().reshape(1, 1, -1)
    monkeypatch.setattr(model, "forward", lambda *args, **kwargs: (logits, None))
    draws = torch.tensor(
        generate(
            model, [0], max_new_tokens=20_000, seed=11, use_cache=False, window_policy="reprefill"
        )
    )
    empirical = draws.bincount(minlength=4) / draws.numel()
    # Five marginal binomial standard errors per bin, including the rarest token.
    assert torch.all(
        (empirical - target).abs() < 5 * (target * (1 - target) / draws.numel()).sqrt()
    )


@pytest.mark.parametrize("preset", ["baseline", "modern"])
def test_stream_seed_entropy_eos_and_reprefill(preset: str) -> None:
    torch.manual_seed(4)
    cfg = GPTConfig.from_preset(preset, block_size=8, n_embd=24, n_head=3, n_layer=2, vocab_size=37)
    model = GPT(cfg).eval()
    prompt = [1, 2, 3]
    options = dict(
        max_new_tokens=19,
        window_policy="reprefill",
        seed=11,
        top_k=5,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    cached = list(generate_stream(model, prompt, **options))
    plain = list(generate_stream(model, prompt, use_cache=False, **options))
    assert [s.token_id for s in cached] == [s.token_id for s in plain]
    assert generate(model, prompt, **options) == [s.token_id for s in cached]
    assert generate(model, prompt, **(options | {"seed": 25})) != [s.token_id for s in cached]
    history = prompt.copy()
    for a, b in zip(cached, plain, strict=True):
        with torch.inference_mode():
            logits, _ = model(torch.tensor([history[-cfg.block_size :]]))
        torch.testing.assert_close(a.raw_distribution, logits[0, -1].softmax(0), rtol=0, atol=1e-6)
        history.append(a.token_id)
        torch.testing.assert_close(a.raw_distribution, b.raw_distribution, rtol=0, atol=1e-6)
        assert 0 <= a.final_entropy <= math.log(5) + 1e-6
        assert a.raw_entropy == pytest.approx(torch.special.entr(a.raw_distribution).sum().item())
        assert a.top_candidates(1)[0][1] == a.final_distribution.max().item()
        assert a.final_distribution[a.token_id] > 0 and a.past_kv is None
    greedy = generate(model, prompt, max_new_tokens=1, temperature=0)[0]
    assert generate(model, prompt, max_new_tokens=5, temperature=0, eos_id=greedy) == [greedy]
    with pytest.raises(ValueError, match="block_size"):
        generate(model, prompt, max_new_tokens=6)
    first = next_token_distribution(model, prompt)
    following = next_token_distribution(model, [greedy], past_kv=first.past_kv)
    direct = next_token_distribution(model, prompt + [greedy])
    torch.testing.assert_close(following.raw_distribution, direct.raw_distribution)
    long_prompt = [0] * 11
    assert generate(model, long_prompt, **options) == generate(
        model, long_prompt, use_cache=False, **options
    )


def test_invalid_cache_length_and_preserve_training_mode() -> None:
    cfg = GPTConfig(block_size=8, n_embd=16, n_head=2, n_layer=2)
    model = GPT(replace(cfg, vocab_size=23))
    with pytest.raises(ValueError, match="per layer"):
        model(torch.tensor([[1]]), past_kv=())
    stream = generate_stream(model, [1], max_new_tokens=3)
    next(stream)
    assert model.training
    stream.close()
