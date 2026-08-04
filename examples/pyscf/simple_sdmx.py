#!/usr/bin/env python
"""Compatibility example for the packaged CIDER24Xe exchange model."""

import sys

try:
    from .exchange_model import main
except ImportError:  # Direct execution from the examples directory.
    from exchange_model import main


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "CIDER24Xe"])
    main()
