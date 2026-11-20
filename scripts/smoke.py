"""Start a private MiniGPT server, verify its HTTP responses, and stop it."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from minigpt.data import seed_everything
from minigpt.io import write_json


def call(base: str, path: str, body: dict[str, Any] | None = None) -> bytes:
    """Raise on any HTTP failure and include the server's validation detail."""
    request = Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"{path} returned {exc.code}: {exc.read().decode()}") from exc


def verify(base: str, seed: int, meta: dict[str, Any]) -> list[str]:
    """Check payloads across the UI, generation, trace, and attention routes."""
    if b"Token inspector" not in call(base, "/") or call(base, "/favicon.ico"):
        raise ValueError("Unexpected UI or favicon")
    body = {
        "prompt": "K",
        "max_new_tokens": min(4, meta["config"]["block_size"] - 1),
        "seed": seed,
        "temperature": 0.8,
    }
    plain = json.loads(call(base, "/generate", body))
    trace = json.loads(call(base, "/trace", body))["steps"]
    frames = call(base, "/generate/stream", body).decode().strip().split("\n\n")
    events = [json.loads(frame.split("\ndata: ", 1)[1]) for frame in frames]
    if not all(frame.startswith("event: token\ndata: ") for frame in frames):
        raise ValueError("Malformed SSE frames")
    for records in (trace, events):
        if (
            [s["chosen"]["id"] for s in records] != plain["token_ids"]
            or "".join(s["text"] for s in records) != plain["text"]
            or not records[-1]["finished"]
            or any(len(s["candidates"]) != 10 for s in records)
        ):
            raise ValueError("Stream, trace and generation disagree")
    attn = json.loads(call(base, "/attention", {"text": "K", "layer": 0, "head": "mean"}))
    if (
        plain["tokens_per_second"] <= 0
        or len(attn["tokens"]) != 1
        or len(attn["matrix"]) != 1
        or not math.isclose(attn["matrix"][0][0], 1.0)
    ):
        raise ValueError("Unexpected generation rate or attention matrix")
    return ["/", "/favicon.ico", "/model", "/generate", "/generate/stream", "/trace", "/attention"]


def main(argv: Sequence[str] | None = None) -> None:
    """Exercise the installed CLI using an ephemeral loopback port."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=os.getenv("MINIGPT_CHECKPOINT", "checkpoints/final.pt")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSON verification record")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log = Path("logs") / f"smoke-{port}.log"
    log.parent.mkdir(exist_ok=True)
    executable = sys.executable
    env = os.environ.copy()
    if os.name == "nt" and sys.prefix != sys.base_prefix:
        # As in multiprocessing.spawn, own the interpreter rather than the venv's
        # redirector process. Terminating the redirector leaves Uvicorn listening.
        executable = sys._base_executable
        env["__PYVENV_LAUNCHER__"] = sys.executable
    with log.open("wb") as stream:
        proc = subprocess.Popen(
            [
                executable,
                "-m",
                "minigpt.serve",
                "--checkpoint",
                str(args.checkpoint),
                "--port",
                str(port),
                "--seed",
                str(args.seed),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited during startup. See {log}")
                try:
                    meta = json.loads(call(base, "/model"))
                    break
                except URLError:
                    time.sleep(0.1)
            else:
                raise TimeoutError(f"Server did not become ready. See {log}")
            if Path(meta["checkpoint"]).resolve() != args.checkpoint.resolve():
                raise ValueError("Unexpected checkpoint on smoke-test port")
            paths = verify(base, args.seed, meta)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("Smoke server still accepts connections after shutdown")
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "seed": args.seed,
        "device": meta["device"],
        "routes": paths,
        "server_stopped": True,
    }
    if args.output:
        write_json(args.output, report)
    print("Smoke passed: " + ", ".join(paths) + f". Server stopped. Log: {log}")


if __name__ == "__main__":
    main()
