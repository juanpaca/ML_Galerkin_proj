#!/usr/bin/env python3
"""Single-command runner for the entire test suite.

Usage (from the repository root):

    venv/bin/python tests/run_all.py
    venv/bin/python tests/run_all.py -x            # stop at first failure
    venv/bin/python tests/run_all.py tests/test_kan.py   # a subset

Equivalent to:  python -m pytest tests/
"""

import os
import sys

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TEST_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if __name__ == "__main__":
    args = sys.argv[1:] or ["-q"]
    raise SystemExit(pytest.main([TEST_DIR, *args]))