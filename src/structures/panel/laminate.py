import numpy as np

from structures.panel.base_components.lamina import Lamina

from ..structural_entity import FailureMode, StructuralEntity, failure_analysis
from .data_utils import PanelLoads, PanelStrains
from .math_utils import rotation_matrix
from .Panel import Panel, calculate_ABD_matrix


class Laminate(StructuralEntity, Panel):
    def __init__(
        self, laminas: list[Lamina], loads: PanelLoads = PanelLoads, strains: PanelStrains = None
    ) -> None:
        StructuralEntity.__init__(self)
        self.laminas: list[Lamina] = laminas
        self.loads: PanelLoads = loads
        self.strains: PanelStrains = strains
        self.sandwich: bool = False
        self.stackingsequence: list[float] = [lamina.theta for lamina in laminas]

        self.assign_lamina_height()

        Panel.__init__(self)

    @property
    def child_objects(self) -> list[Lamina]:
        return self.laminas

    def assign_lamina_height(self) -> None:
        """Assign z0 and z1 heights to each lamina in the laminate."""
        h = 0
        for lamina in self.laminas:
            lamina.z0 = h
            lamina.z1 = h + lamina.t
            h += lamina.t

        self.h = h

        # Center around z=0:
        for lamina in self.laminas:
            lamina.z0 = lamina.z0 - 0.5 * h
            lamina.z1 = lamina.z1 - 0.5 * h

    @calculate_ABD_matrix
    def calculate_ABD_matrix(self) -> np.ndarray:
        """Calculate the ABD matrix of the laminate."""
        ABD = np.zeros((6, 6))

        for lamina in self.laminas:
            ABD += self.lamina_ABD(Qbar=lamina.Qbar, z1=lamina.z1, z0=lamina.z0)

        self.A_matrix = ABD[0:3, 0:3]
        self.B_matrix = ABD[0:3, 3:6]
        self.D_matrix = ABD[3:6, 3:6]
        self.ABD_matrix = ABD

        self.ABD_matrix_inverse = np.linalg.inv(self.ABD_matrix)
        return self.ABD_matrix

    def lamina_ABD(self, Qbar: np.ndarray, z1: float, z0: float) -> np.ndarray:
        """Calculate the ABD matrix of a single lamina given its z0 and z1."""
        A_matrix = np.zeros((3, 3))
        B_matrix = np.zeros((3, 3))
        D_matrix = np.zeros((3, 3))

        # Calculate the difference (Z_k - Z_k-1)
        delta_Z = z1 - z0
        # Update A_ij by adding the product of Q(k) and the difference in Z
        A_matrix += Qbar * delta_Z

        # Now the same for b and d matrices:
        delta_Z_squared = z0**2 - z1**2
        B_matrix += 1 / 2 * (Qbar * delta_Z_squared)

        delta_Z_cubed = z1**3 - z0**3
        D_matrix += 1 / 3 * (Qbar * delta_Z_cubed)

        # Save ABD matrix
        LaminaABD = np.block([[A_matrix, B_matrix], [B_matrix, D_matrix]])
        return LaminaABD

    def calculate_weight_per_A(self) -> float:
        """Calculates weight per unit area of laminate."""
        return sum([lamina.calculate_weight_per_A() for lamina in self.laminas])

    def strains_from_loads(self) -> PanelStrains:
        return PanelStrains(np.linalg.inv(self.ABD_matrix) @ self.loads.array)

    def loads_from_strains(self) -> PanelLoads:
        return PanelLoads(self.ABD_matrix @ self.strains.array)

    def calculate_lamina_strains(self) -> None:
        """
        Calculate strains per lamina based on global laminate.

        Find max strain by comparing strain at z0 and z1 and picking the one with
        maximum absolute value. Set this as the lamina strain. This is used
        for failure analysis in the lamina.
        """
        strains = self.strains.array

        for lamina in self.laminas:
            max1 = max(
                strains[0] - lamina.z0 * strains[3], strains[0] - lamina.z1 * strains[3], key=abs
            )
            max2 = max(
                strains[1] - lamina.z0 * strains[4], strains[1] - lamina.z1 * strains[4], key=abs
            )
            max3 = max(
                strains[2] - lamina.z0 * strains[5], strains[2] - lamina.z1 * strains[5], key=abs
            )
            lamina.strains = np.array([max1, max2, max3])

    def calculate_equivalent_properties(self) -> list[float]:
        """Calculate equivalent engineering properties of the laminate."""
        A_bar = self.A_matrix / self.h
        S = np.linalg.inv(A_bar)

        self.Ex = 1.0 / float(S[0, 0])
        self.Ey = 1.0 / float(S[1, 1])
        self.Gxy = 1.0 / float(S[2, 2])
        self.vxy = -float(S[0, 1]) / float(S[0, 0])
        self.vyx = -float(S[0, 1]) / float(S[1, 1])
        return [self.Ex, self.Ey, self.vxy, self.vyx, self.Gxy]

    def calculate_equivalent_properties_bending(self) -> list[float]:
        D = np.linalg.inv(self.D_matrix)
        E1b = 12 / (self.h**3 * D[0, 0])
        E2b = 12 / (self.h**3 * D[1, 1])
        G12b = 12 / (self.h**3 * D[2, 2])
        v12b = -D[0, 1] / D[1, 1]
        v21b = -D[0, 1] / D[0, 0]
        return [E1b, E2b, G12b, v12b, v21b]

    def stress_analysis(self) -> np.ndarray:
        """Carry out stress analysis for all lamina in laminate."""
        self.calculate_lamina_strains()

        shape = (3, len(self.laminas))
        stresses = np.zeros(shape)
        for i, lamina in enumerate(self.laminas):
            stressesnonflat = lamina.loads_from_strains()
            stressesflat = stressesnonflat.flatten()
            stresses[:, i] = stressesflat
        return stresses

    @failure_analysis
    def failure_analysis(self) -> list[FailureMode]:
        self.stress_analysis()

        failure_indicators = []

        for lamina in self.laminas:
            failure_indicators.append(lamina.failure_analysis())

        max_failure_indicator = max(failure_indicators)
        failure_modes = [("first_ply_failure", max_failure_indicator)]
        return failure_modes

    def buckling_scaling_factor(self, n_crit: float) -> float:
        """Calculate buckling scaling factor."""
        return n_crit

    def n_crit(self) -> float:
        """Calculate critical load intensity given current loading direction."""
        maxfailurefactor = self.failure_analysis()
        n_crit = self.loads.array / maxfailurefactor
        return n_crit

    def calculate_ABD_offset(self, offset: float) -> np.ndarray:
        """Calculate the ABD matrix of the laminate given an offset."""
        ABD = np.zeros((6, 6))

        for lamina in self.laminas:
            z1 = lamina.z1 + offset
            z0 = lamina.z0 + offset
            ABD += self.lamina_ABD(Qbar=lamina.Qbar, z1=z1, z0=z0)
        return ABD

    def rotated_ABD(self, theta: float) -> np.ndarray:
        """Calculate the ABD matrix of the laminate rotated by an angle theta."""
        T = rotation_matrix(theta)
        # Extending T to a 6x6 transformation matrix
        T_ext = np.zeros((6, 6))
        T_ext[:3, :3] = T
        T_ext[3:, 3:] = T

        ABD_transformed = T_ext @ self.ABD_matrix @ T_ext.T
        return ABD_transformed

    # Function to transform the ABD matrix
    def principal_stresses_and_directions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the principal stresses and their corresponding directions
        given a 2D stress tensor.

        Parameters:
        stress_tensor (2x2 numpy array): The 2D stress tensor.

        Returns:
        tuple: A tuple containing:
               - principal_stresses (1D numpy array): The principal stresses.
               - principal_directions (2D numpy array): The corresponding principal
                directions (eigenvectors).
        """
        # find stress tensor based on loads and thickness:
        Sx = self.loads.Nx
        Sy = self.loads.Ny
        Sxy = self.loads.Nxy

        stress_tensor = np.array([[Sx, Sxy], [Sxy, Sy]])

        # Calculate the eigenvalues (principal stresses) and eigenvectors
        # (principal directions)
        principal_load_intensities, principal_directions = np.linalg.eig(stress_tensor)

        # Ensure the principal stresses are ordered from largest to smallest
        idx = np.argsort(principal_load_intensities)[::-1]
        principal_directions = principal_directions[:, idx]

        return principal_load_intensities, principal_directions
