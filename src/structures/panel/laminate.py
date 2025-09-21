# External packages
import copy

import numpy as np
from tqdm import tqdm

# Local imports - use relative imports within the package
from structures.panel.base_components.lamina import Lamina
from ..structural_entity import StructuralEntity


class Laminate(StructuralEntity):
    def __init__(self, laminas, Loads=None, Strains=None):
        super().__init__("laminate")
        self.laminas: list[Lamina] = laminas
        self.Loads: list[float] = Loads
        self.sandwich: bool = False

        # The laminate also has a failure state, which is a list with the failure state
        # of each lamina:
        self.progressive_damage_analysis: bool = False
        self.failure_state = np.zeros(len(self.laminas))

        # We calculate the thickness and find layer start and end height:
        h = 0
        for i in laminas:
            # assign z0 and z1 for each layer:
            i.z0 = h
            i.z1 = h + i.t

            # keep track of total h
            h += i.t

        # now we subtract 0.5 times the height of the laminate to center it around z=0
        for i in laminas:
            i.z0 = i.z0 - 0.5 * h
            i.z1 = i.z1 - 0.5 * h
        self.h: float = h

        self.stackingsequence: list[float] = [lamina.theta_ for lamina in laminas]

        # We calculate the ABD matrix in initialisation
        self.calculate_ABD()
        self.calculate_equivalent_properties()

    @property
    def child_objects(self) -> list[Lamina]:
        return self.laminas

    def calculate_ABD(self) -> None:
        # Initialize A_ij as a zero matrix

        # Initalizing the A, B and D matrix:
        A_matrix = np.zeros((3, 3))
        B_matrix = np.zeros((3, 3))
        D_matrix = np.zeros((3, 3))

        # Per lamina we calculate the three matrices
        for lamina in self.laminas:
            # First we recalculate the Q and S matrix of the lamina:
            lamina.calculate_QS()

            # Calculate the difference (Z_k - Z_k-1)
            delta_Z = lamina.z1 - lamina.z0
            # Update A_ij by adding the product of Q(k) and the difference in Z
            A_matrix += lamina.Q * delta_Z

            # Now the same for b and d matrices:
            delta_Z_squared = lamina.z1**2 - lamina.z0**2
            B_matrix += 1 / 2 * (lamina.Q * delta_Z_squared)

            delta_Z_cubed = lamina.z1**3 - lamina.z0**3
            D_matrix += 1 / 3 * (lamina.Q * delta_Z_cubed)

        # assign the matrices as attributes individually, this can be useful:
        # (but should be removed if this code should be used for high intensity
        # applications)
        self.A_matrix = A_matrix
        self.B_matrix = B_matrix
        self.D_matrix = D_matrix

        # Save ABD matrix
        self.ABD_matrix = np.block([[A_matrix, B_matrix], [B_matrix, D_matrix]])

        # We try the inversion but inversion is not so stable so we use an exception:
        try:
            self.ABD_matrix_inverse = np.linalg.inv(self.ABD_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("ABD matrix inversion failed.")

    def calculate_weight_per_A(self) -> float:
        """Calculates weight per unit area of laminate."""
        return sum([lamina.calculate_weight_per_A() for lamina in self.laminas])

    def calculate_strains(self) -> np.ndarray:
        # First we RECALCULATE the ABD matrix ->
        # this because based on the failure_state of the lamina,
        # They will have different Q matrices
        self.calculate_ABD()

        # Then we check if the loads are assigned and calculate the strains:
        if self.Loads is not None:
            self.Strains = np.zeros((6, 1))
            self.Strains = np.linalg.inv(self.ABD_matrix) @ self.Loads
        else:
            print("loads is nonetype")

        return self.Strains

    def get_strains(self) -> np.ndarray:
        """
        Calculate strains by inverting the ABD matrix and multiplying it with the Loads.

        Function takes into account the current failure state of the laminate,
        """
        Strains = np.linalg.inv(self.ABD_matrix) @ self.Loads
        return Strains

    def calculate_loads(self) -> None:
        """
        Recalculate the ABD matrix and compute laminate loads from assigned global strains.
        Raises ValueError if strains are not set.
        """
        self.calculate_ABD()
        if self.Strains is not None:
            self.Loads = np.zeros((6, 1))
            self.Loads = self.ABD_matrix @ self.Strains
        else:
            raise ValueError("Strains is NoneType, cannot calculate loads.")

    def calculate_lamina_strains(self) -> None:
        """
        Calculate strains per lamina based on global strains.
        """
        # To calculate lamina strains we first need global strains
        self.calculate_strains()

        Strains = self.Strains

        # we go through all the lamina:
        for i in self.laminas:
            # Given the fact that strain is a linear gradient in the laminate, we can
            # find the max strain per lamina by finding picking the max between strain
            # at z0 vs z1
            max1 = max(Strains[0] - i.z0 * Strains[3], Strains[0] - i.z1 * Strains[3], key=abs)
            max2 = max(Strains[1] - i.z0 * Strains[4], Strains[1] - i.z1 * Strains[4], key=abs)
            max3 = max(Strains[2] - i.z0 * Strains[5], Strains[2] - i.z1 * Strains[5], key=abs)
            i.Epsilon = np.array([max1, max2, max3])

    def calculate_equivalent_properties(self) -> tuple[list[float], list[float]]:
        """
        Calculate equivalent engineering properties of the laminate.
        """
        # Here we calculate the engineering constants (or equivalent properties):
        self.Ex = (self.A_matrix[0, 0] * self.A_matrix[1, 1] - self.A_matrix[0, 1] ** 2) / (
            self.h * self.A_matrix[1, 1]
        )
        self.Ey = (self.A_matrix[0, 0] * self.A_matrix[1, 1] - self.A_matrix[0, 1] ** 2) / (
            self.h * self.A_matrix[0, 0]
        )

        self.vxy = self.A_matrix[0, 1] / self.A_matrix[1, 1]
        self.vyx = self.A_matrix[0, 1] / self.A_matrix[0, 0]

        self.Gxy = self.A_matrix[2, 2] / self.h

        D = np.linalg.inv(self.D_matrix)

        E1b = 12 / (self.h**3 * D[0, 0])
        E2b = 12 / (self.h**3 * D[1, 1])
        G12b = 12 / (self.h**3 * D[2, 2])
        v12b = -D[0, 1] / D[1, 1]
        v21b = -D[0, 1] / D[0, 0]
        return [self.Ex, self.Ey, self.vxy, self.vyx, self.Gxy], [
            E1b,
            E2b,
            G12b,
            v12b,
            v21b,
        ]

    def stress_analysis(self) -> np.ndarray:
        # We need to make sure the lamina have strains:
        self.calculate_lamina_strains()

        # we need a method to store the stresses so we can check the stresses
        shape = (3, len(self.laminas))
        stresses = np.zeros(shape)
        for count, i in enumerate(self.laminas):
            # calling of the i.stress_analysis() method should also save the stresses as
            # attributes
            stressesnonflat = i.stress_analysis()
            stressesflat = stressesnonflat.flatten()
            stresses[:, count] = stressesflat
        return stresses

    # carry out failure analysis for all lamina in laminate
    def failure_analysis(self) -> float:
        super().failure_analysis()
        # We need to make sure the lamina have stresses:
        self.stress_analysis()

        # Initializing an array to save the failure factors:
        failure_indicators = []

        for count, lamina in enumerate(self.laminas):
            # Now run for the lamina, the failure analysis
            results = lamina.failure_analysis()
            # set the correct index of the failure_state:
            failure_indicators.append(max(results[1], results[2]))

        # We save the maximum failure factor in any of the lamina, to calculate the
        # next loadstep:
        max_failure_indicator = max(failure_indicators)

        failure_modes = [["first_ply_failure", max_failure_indicator]]
        self.finalize_failure_analysis(failure_modes)

        return max(
            value
            for key, value in self.failure_indicators.items()
            if isinstance(value, (int, float))
        )

    def buckling_scaling_factor(self, n_crit: float) -> float:
        """Calculate buckling scaling factor."""
        return n_crit

    def failure_analysis_pda(self):
        # We need to make sure the lamina have stresses:
        self.stress_analysis()

        # We make an array to track the failed lamina (which one failed):
        failedlamina = []

        # Initializing an array to save the failure factors:
        FailureFactors = []

        # We want to potentially save the lamina which failed, not useful in this
        # assignment though.
        for count, lamina in enumerate(self.laminas):
            # Now run for the lamina, the failure analysis
            results = lamina.failure_analysis(lamina.Sigma)

            # If the failure of the lamina is 1 (for IFF) or 2 (for FF), the lamina has
            # failed
            if results[0] >= 1:
                failedlamina.append(count)

            # set the correct index of the failure_state:
            self.failure_state[count] = lamina.failure_state
            FailureFactors.append(max(results[1], results[2]))

        # We save the maximum failure factor in any of the lamina, to calculate the next loadstep:
        maxfailurefactor = np.max(FailureFactors)
        return self.failure_state, failedlamina, maxfailurefactor

    def Ncrit(self) -> float:
        maxfailurefactor = self.failure_analysis()
        Ncrit = self.Loads / maxfailurefactor
        return Ncrit

    def progressive_damage_analysis(self, loadingratio, loadincrement):
        # Normalize the loading ratio
        normalized_loadingratio = loadingratio / np.max(np.abs(loadingratio))

        # Last ply failure false at first:
        LPF = False

        # Initialize the failure loads and strains as empty lists
        FailureLoadsList = []
        FailureStrainsList = []

        n = 1
        while not LPF:
            # Calculate the load for this iteration
            Loads = normalized_loadingratio * n * loadincrement

            # Set the load attribute
            self.Loads = Loads

            # Run the failure analysis for the laminate with this new load
            failure_state, failedlamina, maxfailurefactor = self.failure_analysis_pda()

            # If a lamina has failed, save these loads and strains
            if failedlamina:
                FailureLoadsList.append(Loads)
                FailureStrainsList.append(self.get_strains())

            # Check whether full failure of all lamina has been achieved
            if np.all(failure_state >= 1):
                LPF = True

            if maxfailurefactor < 0.998:
                # The load should be increased based on the max failure factor observed:
                nnew = n * (1 / maxfailurefactor) * 0.999 + 1
                n = nnew
            else:
                n += 1

        # Convert lists to NumPy arrays for final output
        # If lists are empty, initialize arrays as (n,0) to avoid shape mismatch
        if FailureLoadsList:
            FailureLoads = np.hstack(FailureLoadsList)
            FailureStrains = np.hstack(FailureStrainsList)
        else:
            FailureLoads = np.empty((6, 0))
            FailureStrains = np.empty((6, 0))

        return FailureLoads, FailureStrains

    def produce_failure_envelope(self, loadincrement):
        # We want to plot the stress and strain failure loads:
        angles = np.linspace(1, 360, 1440)
        E22vsE12FPF = []
        E22vsE12LPF = []

        S22vsS12FPF = []
        S22vsS12LPF = []

        FailureStrainsList = []

        for angle in tqdm(angles):
            loadingratio = np.array(
                [
                    [0],
                    [np.cos(np.deg2rad(angle))],
                    [np.sin(np.deg2rad(angle))],
                    [0],
                    [0],
                    [0],
                ]
            )

            FailureLoads, FailureStrains = self.progressive_damage_analysis(
                loadingratio, loadincrement
            )

            # We save the individual points as tuples: This is for one load case:
            E22vsE12 = tuple(zip(FailureStrains[1], FailureStrains[2]))
            FailureStrainsList.append(E22vsE12)

            # now we save the FPF and LPF:
            E22vsE12FPF.append(E22vsE12[0])
            E22vsE12LPF.append(E22vsE12[-1])

            S22vsS12 = tuple(zip(FailureLoads[1], FailureLoads[2]))
            # print('S22 vs S12 failure loads:', S22vsS12, 'at angle:', angle)
            # Here we again take the FPF and LPF
            S22vsS12FPF.append(S22vsS12[0])
            S22vsS12LPF.append(S22vsS12[-1])
            self.reset_failure_state()
        return E22vsE12FPF, E22vsE12LPF, S22vsS12FPF, S22vsS12LPF, FailureStrainsList

    def reset_failure_state(self):
        # First we reset the failure state vector in the laminate:
        self.failure_state = np.zeros(len(self.laminas))

        # Then we also reset these in the lamina
        for lamina in self.laminas:
            lamina.failure_state = 0

        # Lastly, we recalculate the ABD matrix:
        self.calculate_ABD()
        return

    def calculate_core_ABD(self, corethickness):
        # Initalizing the A, B and D matrix:
        A_matrix = np.zeros((3, 3))
        B_matrix = np.zeros((3, 3))
        D_matrix = np.zeros((3, 3))

        # Per lamina we calculate the three matrices
        for lamina in self.laminas:
            # First we recalculate the Q and S matrix of the lamina:
            lamina.calculate_QS()

            # Calculate the difference (Z_k - Z_k-1)
            z1 = lamina.z1 + corethickness / 2 + self.h / 2
            z0 = lamina.z0 + corethickness / 2 + self.h / 2
            delta_Z = z1 - z0
            # Update A_ij by adding the product of Q(k) and the difference in Z
            A_matrix += lamina.Q * delta_Z

            # Now the same for b and d matrices:
            delta_Z_squared = z1**2 - z0**2
            B_matrix += 1 / 2 * (lamina.Q * delta_Z_squared)

            delta_Z_cubed = z1**3 - z0**3
            D_matrix += 1 / 3 * (lamina.Q * delta_Z_cubed)

        # Save ABD matrix
        CoreABD = np.block([[A_matrix, B_matrix], [B_matrix, D_matrix]])
        return CoreABD

    # Function to create the rotation matrix
    def rotation_matrix(self, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array(
            [
                [c**2, s**2, 2 * c * s],
                [s**2, c**2, -2 * c * s],
                [-c * s, c * s, c**2 - s**2],
            ]
        )

    # Function to transform the ABD matrix
    def rotated_ABD(self, theta):
        ABD = self.ABD_matrix
        T = self.rotation_matrix(theta)
        # Extending T to a 6x6 transformation matrix
        T_ext = np.zeros((6, 6))
        T_ext[:3, :3] = T
        T_ext[3:, 3:] = T

        # Transform the ABD matrix
        ABD_transformed = T_ext @ self.ABD_matrix @ T_ext.T
        return ABD_transformed

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
