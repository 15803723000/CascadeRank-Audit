"""Run the repository-local CascadeRank Audit module from this plugin bundle."""

from __future__ import annotations

import sys

from runtime import invoke


def main() -> int:
    return invoke("cascaderank.audit", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
