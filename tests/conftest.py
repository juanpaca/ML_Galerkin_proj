"""Shared pytest configuration for the ML_Galerkin project test suite.

Ensures the repository root is importable so ``import src.*`` works
regardless of how pytest was started (``pytest``/``python -m pytest`` or
via ``tests/run_all.py``).
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)