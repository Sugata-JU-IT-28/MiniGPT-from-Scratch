"""Run Ruff and pytest and save the observed verification results."""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import date
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from minigpt.data import default_device, seed_everything
from minigpt.io import write_json


def case_counts(path: Path) -> dict[str, int]:
    """Read the outcomes of collected cases from pytest's JUnit report."""
    counts = dict.fromkeys(("passed", "skipped", "failed", "errors"), 0)
    for case in ET.parse(path).iter("testcase"):
        for tag, outcome in (("error", "errors"), ("failure", "failed"), ("skipped", "skipped")):
            if case.find(tag) is not None:
                counts[outcome] += 1
                break
        else:
            counts["passed"] += 1
    return counts


def main(argv: Sequence[str] | None = None) -> None:
    """Record exit status, test outcomes, and the available compute device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/verification.json"))
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    ruff = subprocess.run([sys.executable, "-m", "ruff", "check", "."], check=False)
    with TemporaryDirectory() as temp:
        xml = Path(temp) / "pytest.xml"
        pytest = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-ra", f"--junitxml={xml}"], check=False
        )
        results = {
            "date": date.today().isoformat(),
            "seed": args.seed,
            "device": str(default_device()),
            "python": sys.version.split()[0],
            "versions": {name: version(name) for name in ("torch", "numpy", "ruff", "pytest")},
            "ruff_exit_code": ruff.returncode,
            "pytest_exit_code": pytest.returncode,
            "tests": case_counts(xml) if xml.exists() else None,
        }
    write_json(args.output, results)
    print(json.dumps(results, indent=2))
    if ruff.returncode or pytest.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
