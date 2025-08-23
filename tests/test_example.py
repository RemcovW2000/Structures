import pytest

from structures.example import fibonacci


def test_fibonacci_basic() -> None:
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(10) == 55


def test_fibonacci_invalid() -> None:
    with pytest.raises(ValueError):
        fibonacci(-1)
