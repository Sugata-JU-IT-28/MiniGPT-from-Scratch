"""Load TinyShakespeare or a local UTF-8 corpus and preserve its exact text."""

import hashlib
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen

from minigpt.io import write_bytes

DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
RAW_PATH = Path("data/raw/input.txt")


def read_corpus(input_path: Path | None = None) -> str:
    """Use an explicit local input, or atomically cache the one-time download."""
    if input_path is not None:
        return input_path.read_bytes().decode("utf-8")
    if not RAW_PATH.exists():
        try:
            with urlopen(DATASET_URL, timeout=60) as response:
                payload = response.read()
            if not payload:
                raise ValueError("empty response")
            payload.decode("utf-8")
        except (OSError, HTTPException, ValueError) as exc:
            raise SystemExit(
                f"Download failed ({exc}). Place the file at {RAW_PATH.as_posix()}."
            ) from exc
        write_bytes(RAW_PATH, payload)
    return RAW_PATH.read_bytes().decode("utf-8")


def split_text(text: str) -> tuple[str, str]:
    """Cut once at nine tenths of the Unicode code points, before tokenization."""
    cut = len(text) * 9 // 10
    return text[:cut], text[cut:]


def text_digest(text: str) -> str:
    """Identify an exact corpus, including line endings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
