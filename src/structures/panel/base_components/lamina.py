from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from structures.structural_entity import StructuralEntity


@dataclass(frozen=True)
class ElasticProperties:
    """Orthotropic in-plane elastic properties of a unidirectional lamina.

    Attributes:
        E1: Modulus along fiber direction 1 (MPa or consistent units).
        E2: Modulus transverse to fiber direction 2.
        G12: In-plane shear modulus.
        v12: Major Poisson's ratio (strain in 2 due to stress in 1).
    """

    E1: float
    E2: float
    G12: float
    v12: float


@dataclass(frozen=True)
class FailureProperties:
    """Lamina failure properties for fiber and inter-fiber criteria.

    Attributes:
        E11f: Fiber-direction modulus used in fiber failure relation (if applicable).
        v21f: Poisson's ratio 2->1 used in fiber failure adjustment.
        msf: Material safety/compensation factor (e.g., 1.3 for GFRP, 1.1 for CFRP).
        R11t: Longitudinal tensile strength.
        R11c: Longitudinal compressive strength (positive magnitude).
        Yt: Transverse tensile strength.
        Yc: Transverse compressive strength (positive magnitude).
        S: In-plane shear strength.
    """

    E11f: float
    v21f: float
    msf: float
    R11t: float
    R11c: float
    Yt: float
    Yc: float
    S: float


class Lamina(StructuralEntity):
    """Represents a single ply of a composite laminate.

    A Lamina lives inside a Laminate and stores orientation, thickness, material
    properties, and current strain/stress state for analysis and failure checks.

    Attributes:
        t: Ply thickness.
        theta_deg: Ply orientation angle in degrees (0 aligns material-1 with x-axis).
        elastic: ElasticProperties for this ply.
        failure: FailureProperties for this ply.
        rho: Density (mass per volume) in consistent units.
        z0, z1: Ply z-coordinates (assigned by laminate), if applicable.
        epsilon: Current strain vector in global coordinates (shape (3, 1) or (3,)).
        sigma: Current stress vector in global coordinates (shape (3, 1) or (3,)).
        failure_state: 0 = intact, 1 = inter-fiber failure, 2 = fiber failure.
    """

    def __init__(
        self,
        t: float,
        theta_deg: float,
        elastic: ElasticProperties,
        failure: FailureProperties,
        rho: float = 0.0,
        z0: Optional[float] = None,
        z1: Optional[float] = None,
        sigma: Optional[NDArray[np.float64]] = None,
        epsilon: Optional[NDArray[np.float64]] = None,
    ) -> None:
        super().__init__("lamina")

        # Geometric and material properties
        self.t: float = t
        self.theta_deg: float = theta_deg
        self.elastic: ElasticProperties = elastic
        self.failure: FailureProperties = failure
        self.rho: float = rho

        # Ply z coordinates (to be set by Laminate)
        self.z0: Optional[float] = z0
        self.z1: Optional[float] = z1

        # Analysis state
        self.epsilon: Optional[NDArray[np.float64]] = epsilon
        self.sigma: Optional[NDArray[np.float64]] = sigma
        self.max_stress: bool = False  # Use max-stress instead of Puck when True

        # Cached elastic and compliance/stiffness matrices (populated by helpers)
        self.E1: float = elastic.E1
        self.E2: float = elastic.E2
        self.G12: float = elastic.G12
        self.v12: float = elastic.v12
        self.v21 = self.v12 * self.E2 / self.E1
        self.Qbar: NDArray[np.float64] = np.empty((3, 3), dtype=float)
        self.Sbar: NDArray[np.float64] = np.empty((3, 3), dtype=float)

        self._compute_q_s()

    def _compute_q_s(self) -> None:
        m = np.cos(np.deg2rad(self.theta_deg))
        n = np.sin(np.deg2rad(self.theta_deg))

        denom = 1.0 - self.v12 * self.v21
        if denom == 0:
            # Fallback for pathological input
            denom = 1e-12

        Q11 = self.E1 / denom
        Q22 = self.E2 / denom
        Q12 = self.v12 * self.E2 / denom
        Q66 = self.G12

        m2, n2 = m * m, n * n
        m4, n4 = m2 * m2, n2 * n2

        Qxx = Q11 * m4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * n4
        Qxy = (Q11 + Q22 - 4.0 * Q66) * m2 * n2 + Q12 * (m4 + n4)
        Qyy = Q11 * n4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * m4
        Qxs = (Q11 - Q12 - 2.0 * Q66) * n * m2 * m + (Q12 - Q22 + 2.0 * Q66) * n2 * n * m
        Qys = (Q11 - Q12 - 2.0 * Q66) * m * n2 * n + (Q12 - Q22 + 2.0 * Q66) * m2 * m * n
        Qss = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * n2 * m2 + Q66 * (n4 + m4)

        self.Qbar: NDArray[np.float64] = np.array(
            [[Qxx, Qxy, Qxs], [Qxy, Qyy, Qys], [Qxs, Qys, Qss]], dtype=float
        )
        self.Sbar: NDArray[np.float64] = np.linalg.inv(self.Qbar)

    # --- Analysis ---------------------------------------------------------
    def stress_analysis(self) -> NDArray[np.float64]:
        """Compute global stress vector from current global strains.

        Requires self.epsilon to be set (shape (3,) or (3,1)).
        """
        if self.epsilon is None:
            raise ValueError("epsilon (strain state) must be set before stress analysis")
        eps = self.epsilon.reshape(3, 1) if self.epsilon.ndim == 1 else self.epsilon
        self.sigma = self.Qbar @ eps
        return self.sigma

    # --- Failure criteria -------------------------------------------------
    def failure_analysis(self) -> tuple[int, float, float]:
        """Calculate failure indicators for inter-fiber and fiber failure.

        Returns:
            Tuple of (failure_code, inter_fiber_indicator, fiber_indicator).
        """
        if self.sigma is None:
            raise ValueError(
                "sigma (stress state) not set; run stress_analysis first or set manually"
            )
        m = np.cos(np.deg2rad(self.theta_deg))
        n = np.sin(np.deg2rad(self.theta_deg))
        alpha = np.array(
            [[m * m, n * n, 2 * m * n], [n * n, m * m, -2 * m * n], [-m * n, m * n, m * m - n * n]],
            dtype=float,
        )
        sigma123 = alpha @ self.sigma

        if self.max_stress:
            iff = self._iff_max(sigma123)
            ff = self._ff_max(sigma123)
        else:
            iff = self._iff_puck(sigma123)
            ff = self._ff_puck(sigma123)

        failure_modes = [["inter_fiber_failure", float(iff)], ["fiber_failure", float(ff)]]
        self.finalize_failure_analysis(failure_modes)

        if iff >= 1:
            failure = 1
        elif ff >= 1:
            failure = 2
        else:
            failure = 0

        return failure, float(iff), float(ff)

    def _iff_puck(self, sigma: NDArray[np.float64]) -> float:
        """Inter-fiber failure (simplified Puck-like form).

        Note: This is a placeholder and should be validated; parameters may differ per material system.
        """
        s2 = float(sigma[1])
        s6 = float(sigma[2])

        # Intermediary values:
        p12_minus = 0.25
        s23A = (self.failure.S / (2 * p12_minus)) * (
            np.sqrt(1 + 2 * p12_minus * self.failure.Yc / self.failure.S) - 1
        )
        p23_minus = p12_minus * s23A / self.failure.S
        s12c = self.failure.S * np.sqrt(1 + 2 * p23_minus)

        if s2 >= 0:
            term = (
                1 - p12_minus * self.failure.Yt / self.failure.S
            )  # using p12_minus as placeholder
            f = (
                np.sqrt((s6 / self.failure.S) ** 2 + (term * (s2 / self.failure.Yt)) ** 2)
                + 0.3 * s2 / self.failure.S
            )
        elif abs(s2 / (abs(s6) + 1e-11)) <= (s23A / abs(s12c)):
            f = (np.sqrt(s6**2 + (p12_minus * s2) ** 2) + p12_minus * s2) / self.failure.S
        else:
            term1 = s6 / (2 * (1 + p23_minus) * self.failure.S)
            term2 = s2 / self.failure.Yc
            f = (term1**2 + term2**2) * (self.failure.Yc / -s2)
        return float(f)

    def _ff_puck(self, sigma: NDArray[np.float64]) -> float:
        """Fiber failure indicator (tension/compression)."""
        s1 = float(sigma[0])
        R11 = -self.failure.R11c if s1 < 0 else self.failure.R11t
        f = (1.0 / R11) * (
            s1
            - (self.v21 - self.failure.v21f * self.failure.msf * (self.E1 / self.failure.E11f))
            * float(sigma[1])
        )
        return float(f)

    def _iff_max(self, sigma: NDArray[np.float64]) -> float:
        s2 = float(sigma[1])
        s6 = float(sigma[2])
        y = -self.failure.Yc if s2 < 0 else self.failure.Yt
        ftrans = abs(s2) / abs(y)
        fshear = abs(s6) / abs(self.failure.S)
        return float(max(ftrans, fshear))

    def _ff_max(self, sigma: NDArray[np.float64]) -> float:
        s1 = float(sigma[0])
        R11 = -self.failure.R11c if s1 < 0 else self.failure.R11t
        return float(abs(s1) / abs(R11))

    # --- Utilities --------------------------------------------------------
    def calculate_weight_per_area(self) -> float:
        """Mass per unit area of the ply."""
        return self.t * self.rho

    def mass_per_unit_area(self) -> float:
        """Alias for calculate_weight_per_area with clearer naming."""
        return self.calculate_weight_per_area()


if __name__ == "__main__":
    lamina = Lamina(
        t=0.125,
        theta_deg=45.0,
        elastic=ElasticProperties(E1=135_000, E2=10_000, G12=5_000, v12=0.3),
        failure=FailureProperties(
            E11f=230_000, v21f=0.5, msf=1.1, R11t=1500, R11c=1200, Yt=40, Yc=200, S=70
        ),
        rho=1.6e-6,
    )
    lamina.epsilon = np.array([1e-4, 2e-4, 0.0])
    sigma = lamina.stress_analysis()
    print("sigma:", sigma.ravel())
    print("mass/area:", lamina.mass_per_unit_area())
