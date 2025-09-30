import numpy as np

from structures.panel.base_components.lamina import Lamina

from ..structural_entity import StructuralEntity


class Laminate(StructuralEntity):
    def __init__(
        self, laminas: list[Lamina], Loads: list[float] = None, Strains: list[float] = None
    ) -> None:
        super().__init__()
        self.laminas: list[Lamina] = laminas
        self.Loads: list[float] = Loads
        self.Strains: list[float] = Strains
        self.sandwich: bool = False
        self.stackingsequence: list[float] = [lamina.theta for lamina in laminas]

        self.assign_lamina_height()
        self.calculate_ABD()
        self.calculate_equivalent_properties()

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

    def calculate_ABD(self) -> np.ndarray:
        """Calculate the ABD matrix of the laminate."""
        A_matrix = np.zeros((3, 3))
        B_matrix = np.zeros((3, 3))
        D_matrix = np.zeros((3, 3))

        for lamina in self.laminas:
            # Calculate the difference (Z_k - Z_k-1)
            delta_Z = lamina.z1 - lamina.z0
            # Update A_ij by adding the product of Q(k) and the difference in Z
            A_matrix += lamina.Qbar * delta_Z

            # Now the same for b and d matrices:
            delta_Z_squared = lamina.z0**2 - lamina.z1**2
            B_matrix += 1 / 2 * (lamina.Qbar * delta_Z_squared)

            delta_Z_cubed = lamina.z1**3 - lamina.z0**3
            D_matrix += 1 / 3 * (lamina.Qbar * delta_Z_cubed)

        # assign the matrices as attributes individually, this can be useful:
        # (but should be removed if this code should be used for high intensity
        # applications)
        self.A_matrix = A_matrix
        self.B_matrix = B_matrix
        self.D_matrix = D_matrix

        # Save ABD matrix
        self.ABD_matrix = np.block([[A_matrix, B_matrix], [B_matrix, D_matrix]])

        self.ABD_matrix_inverse = np.linalg.inv(self.ABD_matrix)
        return self.ABD_matrix

    def calculate_weight_per_A(self) -> float:
        """Calculates weight per unit area of laminate."""
        return sum([lamina.calculate_weight_per_A() for lamina in self.laminas])

    def calculate_strains_from_loads(self) -> np.ndarray:
        self.Strains = np.linalg.inv(self.ABD_matrix) @ self.Loads
        return self.Strains

    def calculate_loads_from_strains(self) -> np.ndarray:
        self.Loads = self.ABD_matrix @ self.Strains
        return self.Loads

    def calculate_lamina_strains(self) -> None:
        """
        Calculate strains per lamina based on global laminate.

        Find max strain by comparing strain at z0 and z1 and picking the one with
        maximum absolute value. Set this as the lamina strain. This is used
        for failure analysis in the lamina.
        """
        Strains = self.calculate_strains_from_loads()

        for lamina in self.laminas:
            max1 = max(
                Strains[0] - lamina.z0 * Strains[3], Strains[0] - lamina.z1 * Strains[3], key=abs
            )
            max2 = max(
                Strains[1] - lamina.z0 * Strains[4], Strains[1] - lamina.z1 * Strains[4], key=abs
            )
            max3 = max(
                Strains[2] - lamina.z0 * Strains[5], Strains[2] - lamina.z1 * Strains[5], key=abs
            )
            lamina.Epsilon = np.array([max1, max2, max3])

    def calculate_equivalent_properties(self) -> list[float]:
        """Calculate equivalent engineering properties of the laminate."""
        self.Ex = (self.A_matrix[0, 0] * self.A_matrix[1, 1] - self.A_matrix[0, 1] ** 2) / (
            self.h * self.A_matrix[1, 1]
        )
        self.Ey = (self.A_matrix[0, 0] * self.A_matrix[1, 1] - self.A_matrix[0, 1] ** 2) / (
            self.h * self.A_matrix[0, 0]
        )

        self.vxy = self.A_matrix[0, 1] / self.A_matrix[1, 1]
        self.vyx = self.A_matrix[0, 1] / self.A_matrix[0, 0]

        self.Gxy = self.A_matrix[2, 2] / self.h
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
            stressesnonflat = lamina.stress_analysis()
            stressesflat = stressesnonflat.flatten()
            stresses[:, i] = stressesflat
        return stresses

    def failure_analysis(self) -> float:
        super().failure_analysis()
        self.stress_analysis()

        failure_indicators = []

        for lamina in self.laminas:
            failure_indicators.append(lamina.failure_analysis())

        max_failure_indicator = max(failure_indicators)
        failure_modes = [["first_ply_failure", max_failure_indicator]]
        self.set_failure_indicators(failure_modes)
        return max_failure_indicator

    def buckling_scaling_factor(self, n_crit: float) -> float:
        """Calculate buckling scaling factor."""
        return n_crit

    def n_crit(self) -> float:
        """Calculate critical load intensity given current loading direction."""
        maxfailurefactor = self.failure_analysis()
        n_crit = self.Loads / maxfailurefactor
        return n_crit

    def calculate_abd_for_sandwich(self, corethickness: float) -> np.ndarray:
        """Calculate the ABD matrix of the laminate given an extra core thickness."""
        A_matrix = np.zeros((3, 3))
        B_matrix = np.zeros((3, 3))
        D_matrix = np.zeros((3, 3))

        for lamina in self.laminas:
            # Calculate the difference (Z_k - Z_k-1)
            z1 = lamina.z1 + corethickness / 2 + self.h / 2
            z0 = lamina.z0 + corethickness / 2 + self.h / 2

            delta_Z = z1 - z0
            # Update A_ij by adding the product of Q(k) and the difference in Z
            A_matrix += lamina.Qbar * delta_Z

            # Now the same for b and d matrices:
            delta_Z_squared = z1**2 - z0**2
            B_matrix += 1 / 2 * (lamina.Qbar * delta_Z_squared)

            delta_Z_cubed = z1**3 - z0**3
            D_matrix += 1 / 3 * (lamina.Qbar * delta_Z_cubed)

        CoreABD = np.block([[A_matrix, B_matrix], [B_matrix, D_matrix]])
        return CoreABD

    def rotated_ABD(self, theta: float) -> np.ndarray:
        """Calculate the ABD matrix of the laminate rotated by an angle theta."""
        T = self.rotation_matrix(theta)
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
        Sx = self.Loads[0]
        Sy = self.Loads[1]
        Sxy = self.Loads[2]

        stress_tensor = np.array([[Sx, Sxy], [Sxy, Sy]])

        # Calculate the eigenvalues (principal stresses) and eigenvectors
        # (principal directions)
        principal_loadintensities, principal_directions = np.linalg.eig(stress_tensor)

        # Ensure the principal stresses are ordered from largest to smallest
        idx = np.argsort(principal_loadintensities)[::-1]
        principal_directions = principal_directions[:, idx]

        return principal_loadintensities, principal_directions

    @staticmethod
    def rotation_matrix(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array(
            [
                [c**2, s**2, 2 * c * s],
                [s**2, c**2, -2 * c * s],
                [-c * s, c * s, c**2 - s**2],
            ]
        )
