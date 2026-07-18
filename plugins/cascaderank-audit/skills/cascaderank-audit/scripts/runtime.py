"""Locate a CascadeRank runtime for plugin wrapper commands."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _ancestors(path: Path) -> list[Path]:
    resolved = path.resolve()
    return [resolved, *resolved.parents]


def find_source_root() -> Path | None:
    """Find a workspace checkout without assuming a plugin install location."""

    candidates = _ancestors(Path.cwd()) + _ancestors(Path(__file__).parent)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "cascaderank" / "audit.py").is_file():
            return candidate
    return None


def invoke(module: str, arguments: list[str]) -> int:
    """Invoke a runtime module from source when available, else site-packages."""

    root = find_source_root()
    environment = dict(os.environ)
    if root is not None:
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(root) + os.pathsep + inherited
    return subprocess.call(
        [sys.executable, "-m", module, *arguments],
        cwd=root,
        env=environment,
    )
