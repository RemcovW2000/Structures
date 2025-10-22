import numpy as np


def rotation_matrix(theta: float) -> np.ndarray:
    """Return the 3x3 rotation matrix for a given angle in radians."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c**2, s**2, 2 * c * s],
            [s**2, c**2, -2 * c * s],
            [-c * s, c * s, c**2 - s**2],
        ]
    )
