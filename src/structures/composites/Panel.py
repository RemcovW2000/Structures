from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable

import numpy as np


def calculate_ABD_matrix(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
        self = args[0]
        self.ABD_matrix = func(*args, **kwargs)
        self.A_matrix = self.ABD_matrix[0:3, 0:3]
        self.B_matrix = self.ABD_matrix[0:3, 3:6]
        self.D_matrix = self.ABD_matrix[3:6, 3:6]
        return self.ABD_matrix

    # mark the wrapper so subclasses are required to use this decorator
    wrapper._is_calculate_ABD_matrix = True
    return wrapper


class Panel(ABC):
    """Abstract panel base.

    Subclasses must implement `calculate_ABD_matrix()` to compute and set
    `A_matrix`, `B_matrix`, `D_matrix` and `h`. After setting those values
    they can call `calculate_equivalent_properties()` to populate the
    equivalent engineering properties (`Ex`, `Ey`, `vxy`, `vyx`, `Gxy`).
    """

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if "calculate_ABD_matrix" not in cls.__dict__:
            raise TypeError(
                "Subclasses must implement `calculate_ABD_matrix` and decorate it with `@calculate_ABD_matrix`"
            )

        impl = cls.__dict__["calculate_ABD_matrix"]
        if not getattr(impl, "_is_calculate_ABD_matrix", False):
            raise TypeError("`calculate_ABD_matrix` must be decorated with `@calculate_ABD_matrix`")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calculate_ABD_matrix()
        self.calculate_equivalent_properties()
        # hey

    @abstractmethod
    def calculate_ABD_matrix(self) -> None:
        """Compute and assign `A_matrix`, `B_matrix`, `D_matrix` and `h`.

        Implementations must set the concrete properties (for example via
        `self.A_matrix = ...`) so that subsequent calls to
        `calculate_equivalent_properties()` succeed.
        """
        raise NotImplementedError("Subclasses must implement calculate_ABD_matrix")

    def calculate_equivalent_properties(self) -> list[float]:
        """Calculate membrane-equivalent engineering properties from A_matrix and h.

        Returns [Ex, Ey, vxy, vyx, Gxy].

        Raises:
            ValueError: if `A_matrix` or `h` is not set.
        """
        A_bar = self.A_matrix / float(self.h)
        S = np.linalg.inv(A_bar)

        # assign via setters (they store as floats)
        self.Ex = 1.0 / float(S[0, 0])
        self.Ey = 1.0 / float(S[1, 1])
        self.Gxy = 1.0 / float(S[2, 2])
        self.vxy = -float(S[0, 1]) / float(S[0, 0])
        self.vyx = -float(S[0, 1]) / float(S[1, 1])

        return [self.Ex, self.Ey, self.vxy, self.vyx, self.Gxy]

    def calculate_equivalent_properties_bending(self) -> list[float]:
        """Calculate bending-equivalent engineering properties from D_matrix and h.

        Returns [E1b, E2b, G12b, v12b, v21b].

        Raises:
            ValueError: if `D_matrix` or `h` is not set.
        """
        D = np.linalg.inv(self.D_matrix)
        E1b = 12.0 / (self.h**3 * D[0, 0])
        E2b = 12.0 / (self.h**3 * D[1, 1])
        G12b = 12.0 / (self.h**3 * D[2, 2])
        v12b = -D[0, 1] / D[1, 1]
        v21b = -D[0, 1] / D[0, 0]
        return [E1b, E2b, G12b, v12b, v21b]
