"""Run the repository-local CascadeRank manifest verifier from this plugin."""

from __future__ import annotations

import sys

from runtime import invoke


def main() -> int:
    return invoke("cascaderank.verify", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
