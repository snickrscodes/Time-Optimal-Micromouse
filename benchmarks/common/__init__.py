"""Shared benchmark utilities.

This package initializer is intentionally import-light. Numerical/production
modules are imported only by the specific benchmark worker that needs them, so
orchestration parents remain free of solver lifecycle state.
"""
