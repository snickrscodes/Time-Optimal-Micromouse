#!/usr/bin/env python3
"""Explicitly build the complete production native stack."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
make_args = os.environ.get("AME_MAKE_ARGS", "").split()
subprocess.run(["make", "-C", str(root), "native", *make_args], check=True)
