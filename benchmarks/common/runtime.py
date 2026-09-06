from __future__ import annotations

import contextlib

from optimization import reverse_solver


@contextlib.contextmanager
def reverse_backend(name: str):
    """Switch reverse backend for a benchmark and prove restoration on exit."""
    previous = reverse_solver.reverse_backend()
    try:
        with reverse_solver.using_reverse_backend(name):
            yield
    finally:
        if reverse_solver.reverse_backend() != previous:
            raise AssertionError("reverse backend context did not restore previous backend")
