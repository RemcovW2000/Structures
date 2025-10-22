"""Top-level package for structures.

This package provides data structures and utilities.
"""

from importlib.metadata import PackageNotFoundError, version

# Public API re-exports
from .composites.base_components.lamina import ElasticProperties, FailureProperties, Lamina

try:
    __version__ = version("structures")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Set global numpy print options for nicer matrix printing across the project
# These options apply whenever numpy arrays are printed (e.g., in tests/logs).
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=120)

__all__ = [
    "__version__",
    "ElasticProperties",
    "FailureProperties",
    "Lamina",
]
