#!/usr/bin/env python
"""Compatibility entry point for the production periodic example."""

try:
    from .production_calc import main
except ImportError:  # Direct execution: python examples/gpaw/simple_calc.py
    from production_calc import main


if __name__ == "__main__":
    main()
