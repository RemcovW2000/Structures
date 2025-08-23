"""Top-level package for structures.

This package provides data structures and utilities.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("structures")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
