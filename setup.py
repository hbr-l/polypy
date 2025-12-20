"""
Legacy setup.py kept for backward compatibility.

The project now uses pyproject.toml (PEP 621) for build and dependency
configuration. New installations should prefer:

    pip install .
or
    pip install -e .[dev]

This stub is only here to avoid breaking older tooling that still
expects a setup.py file.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
