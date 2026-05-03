"""Compatibility wrapper for export workflows."""

from __future__ import annotations

import sys

from plas.cli import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "export", *sys.argv[1:]]
    main()
