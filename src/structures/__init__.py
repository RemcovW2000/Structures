"""Top-level package for structures.

This package provides data structures and utilities.
"""


# Public API re-exports
# Set global numpy print options for nicer matrix printing across the project
# These options apply whenever numpy arrays are printed (e.g., in tests/logs).
import numpy as np

from .composites.base_components.lamina import ElasticProperties, FailureProperties, Lamina

np.set_printoptions(precision=3, suppress=True, linewidth=120)

__all__ = [
    "ElasticProperties",
    "FailureProperties",
    "Lamina",
]
