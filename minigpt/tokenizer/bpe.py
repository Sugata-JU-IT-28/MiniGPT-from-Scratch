"""Lossless byte-level BPE with frequency-weighted incremental training."""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from itertools import pairwise
from pathlib import Path

from minigpt.io import write_json

Pair = tuple[int, int]
END_OF_TEXT = "<|endoftext|>"
EOT_ID = 256
BASE_SIZE = 257

# Stdlib \w includes underscore. Put it in the punctuation branch explicitly.
# The lookahead leaves a final space available as the prefix of the next word.
_PATTERN = re.compile(
    r"(?i:'(?:s|t|re|ve|m|ll|d))| ?[^\W\d_]+| ?\d+| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+"
)


def pre_tokens(text: str) -> Iterator[str]:
    """Yield a partition of the input, preserving every whitespace character."""
    return (match[0] for match in _PATTERN.finditer(text))


def _merge(ids: Sequence[int], pair: Pair, replacement: int) -> list[int]:
    """Replace non-overlapping occurrences of a pair from left to right."""
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(replacement)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def _learn(text: str, num_merges: int) -> list[Pair]:
    """Update only words containing the chosen pair, weighted by corpus frequency."""
    if not num_merges:
        return []
    pieces = Counter(piece for part in text.split(END_OF_TEXT) for piece in pre_tokens(part))
    words = [list(piece.encode("utf-8")) for piece in pieces]
    frequencies = list(pieces.values())
    counts: Counter[Pair] = Counter()
    members: dict[Pair, set[int]] = defaultdict(set)
    for index, word in enumerate(words):
        for pair, count in Counter(pairwise(word)).items():
            counts[pair] += count * frequencies[index]
            members[pair].add(index)

    # Python's min-heap orders by negative frequency, then the smallest pair tuple.
    heap = [(-count, pair) for pair, count in counts.items()]
    heapq.heapify(heap)
    merges: list[Pair] = []
    while len(merges) < num_merges:
        while heap:
            negative_count, best = heapq.heappop(heap)
            if counts[best] == -negative_count:
                break
        else:
            raise ValueError(f"Corpus supports only {BASE_SIZE + len(merges)} vocabulary entries")

        replacement = BASE_SIZE + len(merges)
        merges.append(best)
        deltas: Counter[Pair] = Counter()
        for index in tuple(members[best]):
            word = words[index]
            updated = _merge(word, best, replacement)
            before = Counter(pairwise(word))
            after = Counter(pairwise(updated))
            for pair in before.keys() | after.keys():
                delta = after[pair] - before[pair]
                if delta:
                    deltas[pair] += delta * frequencies[index]
                if before[pair] == 0:
                    members[pair].add(index)
                elif after[pair] == 0:
                    members[pair].remove(index)
            words[index] = updated
        for pair, delta in deltas.items():
            if delta:
                counts[pair] += delta
                if counts[pair]:
                    heapq.heappush(heap, (-counts[pair], pair))
                else:
                    del counts[pair], members[pair]
    return merges


class BPETokenizer:
    """Use IDs 0–255 for bytes, 256 for end-of-text, then learned pairs in rank order."""

    def __init__(self) -> None:
        """Start with the byte vocabulary and the end-of-text marker."""
        self._install([])

    def _install(self, merges: list[Pair]) -> None:
        self._merges = tuple(merges)
        self._ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._tokens = [bytes([i]) for i in range(256)] + [END_OF_TEXT.encode()]
        for left, right in merges:
            self._tokens.append(self._tokens[left] + self._tokens[right])
        self._cache: dict[str, tuple[int, ...]] = {}

    @property
    def merges(self) -> tuple[Pair, ...]:
        """Return learned pairs in encoding priority order."""
        return self._merges

    @property
    def vocab_size(self) -> int:
        """Count all base bytes, the special token, and learned merges."""
        return len(self._tokens)

    def train(self, text: str, vocab_size: int) -> BPETokenizer:
        """Fit exactly this vocabulary size. Ties go to the smallest pair tuple."""
        if vocab_size < BASE_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_SIZE}")
        self._install(_learn(text, vocab_size - BASE_SIZE))
        return self

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        cached = self._cache.get(piece)
        if cached is None:
            ids = list(piece.encode("utf-8"))
            while len(ids) > 1:
                ranks = (self._ranks[p] for p in pairwise(ids) if p in self._ranks)
                rank = min(ranks, default=None)
                if rank is None:
                    break
                ids = _merge(ids, self._merges[rank], BASE_SIZE + rank)
            cached = self._cache[piece] = tuple(ids)
        return cached

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Encode exact UTF-8 text and optionally recognize literal end-of-text markers."""
        ids: list[int] = []
        for i, part in enumerate(text.split(END_OF_TEXT) if allow_special else (text,)):
            if i:
                ids.append(EOT_ID)
            for piece in pre_tokens(part):
                ids.extend(self._encode_piece(piece))
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """Return the original bytes, including potentially incomplete UTF-8 sequences."""
        result = bytearray()
        for token_id in ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Unknown token ID {token_id}")
            result.extend(self._tokens[token_id])
        return bytes(result)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode as UTF-8, replacing malformed sequences in arbitrary token slices."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Store the ordered merge list with its byte vocabulary format version."""
        write_json(Path(path), {"version": 1, "vocab_size": self.vocab_size, "merges": self.merges})

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Restore and validate a saved byte-level tokenizer."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["version"] != 1 or data["vocab_size"] != BASE_SIZE + len(data["merges"]):
            raise ValueError("Invalid tokenizer version or vocabulary size")
        merges: list[Pair] = []
        seen: set[Pair] = set()
        for rank, pair in enumerate(data["merges"]):
            if len(pair) != 2 or any(
                type(i) is not int or not 0 <= i < BASE_SIZE + rank or i == EOT_ID for i in pair
            ):
                raise ValueError(f"Invalid merge at rank {rank}")
            left, right = pair
            if (left, right) in seen:
                raise ValueError(f"Duplicate merge at rank {rank}")
            seen.add((left, right))
            merges.append((left, right))
        model = cls()
        model._install(merges)
        return model
