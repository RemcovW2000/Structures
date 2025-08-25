"""Top-level package for structures.

This package provides data structures and utilities.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("structures")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Public API re-exports
from .panel.base_components.lamina import ElasticProperties, FailureProperties, Lamina  # noqa: F401

__all__ = [
    "__version__",
    "ElasticProperties",
    "FailureProperties",
    "Lamina",
]
