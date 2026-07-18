"""Run the repository-local CascadeRank manifest verifier from this plugin."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[5]
    environment = dict(os.environ)
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(root) + os.pathsep + inherited
    return subprocess.call(
        [sys.executable, "-m", "cascaderank.verify", *sys.argv[1:]],
        cwd=root,
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
