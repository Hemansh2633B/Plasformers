"""Compatibility wrapper for launching training from tools/."""

from __future__ import annotations

import sys

from plas.cli import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "train", *sys.argv[1:]]
    main()
