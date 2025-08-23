"""Example module with simple functionality.

This module is used to demonstrate testing and documentation generation.
"""

from __future__ import annotations


def fibonacci(n: int) -> int:
    """Compute the n-th Fibonacci number.

    Uses an iterative algorithm with O(n) time and O(1) space.

    Args:
        n: Index in the Fibonacci sequence (0-indexed). Must be >= 0.

    Returns:
        The n-th Fibonacci number.

    Raises:
        ValueError: If ``n`` is negative.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
