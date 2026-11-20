"""Check lossless byte handling, reference merge ranks, and persistence."""

import json
import random
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from minigpt.tokenizer import BPETokenizer
from minigpt.tokenizer.bpe import BASE_SIZE, END_OF_TEXT, EOT_ID, Pair, pre_tokens

TRAIN = (
    "low lower lowest newer wider wool woollier bananas banana bandana!!\n" * 4
    + "আমি हिंदी 中文 café abc_123 aaaa abba 👩🏽‍💻\t\r\n"
)
CASES = [
    "",
    "🧑🏽‍🚀👨‍👩‍👧‍👦",
    "বাংলা ভাষা",
    "हिन्दी और हिंदी",
    "學習漢字かな",
    "n\u0303 e\u0301",
    "foo\r\n\r\nbar",
    "\t \t    a   b      ",
    "_\x00_\u2009",
    " we're I'LL 42_²\t",
    "\U0010ffff\x00",
    END_OF_TEXT * 2,
]


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer().train(TRAIN, 290)


@pytest.mark.parametrize("text", CASES)
def test_unicode_partition_and_round_trip(tokenizer: BPETokenizer, text: str) -> None:
    assert "".join(pre_tokens(text)) == text
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode_bytes(ids) == text.encode("utf-8")
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


@pytest.mark.slow
def test_seeded_random_unicode(tokenizer: BPETokenizer) -> None:
    rng = random.Random(7931)
    common = "".join(CASES)
    for _ in range(500):
        chars: list[str] = []
        for _ in range(rng.randrange(81)):
            # Sample the complete scalar range, skipping the surrogate interval.
            n = rng.randrange(0x110000 - 0x800)
            scalar = chr(n if n < 0xD800 else n + 0x800)
            chars.append(rng.choice(common) if rng.random() < 0.4 else scalar)
        text = "".join(chars)
        assert "".join(pre_tokens(text)) == text
        ids = tokenizer.encode(text)
        assert tokenizer.decode_bytes(ids) == text.encode("utf-8")
        assert tokenizer.decode(ids) == text


def naive_merges(text: str, size: int) -> tuple[Pair, ...]:
    """Recount every adjacent pair in every occurrence at each step."""
    words = [list(p.encode()) for part in text.split(END_OF_TEXT) for p in pre_tokens(part)]
    merges: list[Pair] = []
    for new_id in range(BASE_SIZE, size):
        counts = Counter(p for word in words for p in pairwise(word))
        winner = min(counts, key=lambda p: (-counts[p], p))
        merges.append(winner)
        for word in words:
            i = 0
            while i < len(word) - 1:
                if word[i : i + 2] == list(winner):
                    word[i : i + 2] = [new_id]
                i += 1
    return tuple(merges)


@pytest.mark.parametrize(
    ("text", "size"),
    [
        (TRAIN, 290),
        ("aaaaa aaaa aaaaa aaaaaa aabaa baba ababab", 266),
        ("ba ab aa bb" + END_OF_TEXT + "ba ab aa bb", 262),
        ("zzyyxx", 260),
        ("খবর खबर 🤖 खबर", 270),
    ],
)
def test_incremental_matches_naive(text: str, size: int) -> None:
    assert BPETokenizer().train(text, size).merges == naive_merges(text, size)


def test_determinism_vocab_and_retraining(tokenizer: BPETokenizer) -> None:
    other = BPETokenizer().train(TRAIN, 290)
    assert other.merges == tokenizer.merges
    assert len(other.merges) == 290 - BASE_SIZE
    assert other.vocab_size == 290
    ids = other.encode(TRAIN)
    assert all(0 <= i < other.vocab_size for i in ids)
    other.train("", BASE_SIZE)
    assert other.encode(TRAIN) == list(TRAIN.encode())
    assert other.merges == ()


def test_encoding_uses_lowest_rank_first() -> None:
    tokenizer = BPETokenizer().train("bcbcbcbcab", 258)
    assert tokenizer.merges == ((ord("b"), ord("c")),)
    assert tokenizer.encode("abc") == [ord("a"), 257]
    tokenizer.train("aaaa", 258)
    assert tokenizer.encode("aaa") == [257, ord("a")]


def test_competing_learned_pairs_use_rank(tmp_path: Path) -> None:
    path = tmp_path / "competing.json"
    path.write_text(
        json.dumps({"version": 1, "vocab_size": 259, "merges": [[98, 99], [97, 98]]}),
        encoding="utf-8",
    )
    assert BPETokenizer.load(path).encode("abc") == [97, 257]


def test_special_tokens_and_serialization(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    text = END_OF_TEXT * 2 + TRAIN + END_OF_TEXT
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.encode(END_OF_TEXT) == [EOT_ID]
    for allow in (True, False):
        ids = loaded.encode(text, allow_special=allow)
        assert ids == tokenizer.encode(text, allow_special=allow)
        assert ids.count(EOT_ID) == (3 if allow else 0)
        assert loaded.decode(ids) == text
    assert BPETokenizer().encode(END_OF_TEXT, False) == list(END_OF_TEXT.encode())


def test_partial_bytes_invalid_ids_and_unreachable_size() -> None:
    tokenizer = BPETokenizer()
    assert tokenizer.decode_bytes([0xF0]) == b"\xf0"
    assert tokenizer.decode([0xF0]) == "\ufffd"
    with pytest.raises(ValueError, match="Unknown token ID"):
        tokenizer.decode([-1])
    with pytest.raises(ValueError, match="at least"):
        tokenizer.train("", 256)
    with pytest.raises(ValueError, match="supports only"):
        tokenizer.train("x", 258)
