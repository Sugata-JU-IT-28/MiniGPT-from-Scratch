"""Write generated artifacts without exposing partial files."""

import hashlib
import json
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile


def display_metric(value: float, places: int = 4) -> str:
    """Truncate a displayed metric while preserving the original in the JSON artifact."""
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN))


def file_sha256(path: Path) -> str:
    """Hash a dataset or checkpoint in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    """Publish a complete file by renaming a temporary file on the same volume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    """Replace a JSON artifact after its complete UTF-8 representation is on disk."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_bytes(path, payload.encode("utf-8"))
