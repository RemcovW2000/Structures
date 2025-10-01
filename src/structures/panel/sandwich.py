import numpy as np
from base_components.core import Core

from structures.panel.data_utils import PanelLoads, PanelStrains
from structures.panel.laminate import Laminate
from structures.structural_entity import StructuralEntity


class Sandwich(StructuralEntity):
    def __init__(
        self,
        bottom_laminate: Laminate,
        top_laminate: Laminate,
        core: Core,
        loads: PanelLoads = None,
        strains: PanelStrains = None,
    ):
        super().__init__()

        self.bottom_laminate: Laminate = bottom_laminate  # bottom laminate
        self.top_laminate: Laminate = top_laminate  # top laminate
        self.core: Core = core
        self.sandwich: bool = True

        self.loads: PanelLoads = loads
        self.strains: PanelStrains = strains
        self.h: float = self.bottom_laminate.h + self.top_laminate.h + self.core.h

        # ABD matrix assigned upon initialisation:
        self.calculate_equivalent_ABD()
        self.calculate_equivalent_properties()

    @property
    def child_objects(self) -> list[StructuralEntity]:
        return [self.bottom_laminate, self.top_laminate, self.core]

    def calculate_equivalent_ABD(self) -> np.ndarray:
        """
        Calculates the ABD matrix for the sandwich structure
        :return:
        """
        # TODO: add transverse shear effects
        bottom_ABD = self.bottom_laminate.calculate_ABD_offset(-self.core.h / 2)
        top_ABD = self.top_laminate.calculate_ABD_offset(self.core.h / 2)

        totalABD = bottom_ABD + top_ABD

        self.ABD_matrix = totalABD
        return totalABD

    def calculate_equivalent_properties(self) -> None:
        # Here we calculate the engineering constants (or equivalent properties):
        self.Ex = (self.ABD_matrix[0, 0] * self.ABD_matrix[1, 1] - self.ABD_matrix[0, 1] ** 2) / (
            self.h * self.ABD_matrix[1, 1]
        )
        self.Ey = (self.ABD_matrix[0, 0] * self.ABD_matrix[1, 1] - self.ABD_matrix[0, 1] ** 2) / (
            self.h * self.ABD_matrix[0, 0]
        )

        self.vxy = self.ABD_matrix[0, 1] / self.ABD_matrix[1, 1]
        self.vyx = self.ABD_matrix[0, 1] / self.ABD_matrix[0, 0]

        self.Gxy = self.ABD_matrix[2, 2] / self.h

    def failure_analysis(self) -> float:
        """
        Perform failure analysis on sandwich panel.

        FI>0, when FI = 1 failure occurs at the applied load
        loads must be assigned to the sandwich panel before calling this function.
        """
        super().failure_analysis()

        # First calculate loads on each facesheet:

        # calculate loads on facesheets:
        self.face_sheet_load_distribution()

        # check first ply failure for both facesheets:
        FPFFI = self.laminate_fpf()

        # check for wrinkling in both facesheets:
        # NOTE: both facesheets must be analyzed separately

        l1_stresses, l1_directions = self.bottom_laminate.principal_stresses_and_directions()

        l2_stresses, l2_directions = self.top_laminate.principal_stresses_and_directions()

        # using the directions we can find the material strength in that direction
        wrinklingFI1 = self.wrinkling_analysis(l1_stresses, l1_directions, self.bottom_laminate)
        wrinklingFI2 = self.wrinkling_analysis(l2_stresses, l2_directions, self.top_laminate)

        failure_modes = [
            ["wrinkling", wrinklingFI1],
            ["wrinkling", wrinklingFI2],
            ["first_ply_failure", FPFFI],
        ]
        self.finalize_failure_analysis(failure_modes)

        return max(
            value
            for key, value in self.failure_indicators.items()
            if isinstance(value, (int, float))
        )

    def calculate_weight_per_area(self) -> float:
        """Calculates weight per unit area of laminate."""
        return np.sum(
            [
                self.bottom_laminate.calculate_weight_per_A(),
                self.top_laminate.calculate_weight_per_A(),
                self.core.calculate_weight_per_A(),
            ]
        )

    def laminate_fpf(self) -> float:
        # loads are assigned
        max_fi1 = self.bottom_laminate.failure_analysis()
        max_fi2 = self.top_laminate.failure_analysis()
        return max(max_fi1, max_fi2)

    def buckling_scaling_factor(self, Ncrit: float) -> float:
        # TODO: implement correct k
        # Assuming k = 1 for now:
        # k = 1
        Nxcrit = Ncrit / (1 + Ncrit / (self.core.h * self.core.G))
        return Nxcrit

    def wrinkling_analysis(
        self, laminate_stresses: np.ndarray, laminate_directions: np.ndarray, laminate: Laminate
    ) -> float:
        # We obtain stresses and directions:
        # TODO: take into account assymetric laminates
        negative_values = laminate_stresses[laminate_stresses < 0]
        if len(negative_values) > 0:
            # Find the index of the negative value with the largest absolute value
            max_negative_index = np.where(
                laminate_stresses == negative_values[np.argmax(np.abs(negative_values))]
            )[0][0]
            direction = laminate_directions[max_negative_index]

            # now calculate E at the given angle:
            vx, vy = direction[0], direction[1]
            theta = np.arctan2(vy, vx)  # Calculate the angle theta in radians

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Calculate 1/E_theta using the formula
            E_theta_inv = (
                (cos_theta**4) / laminate.Ex
                + (sin_theta**4) / laminate.Ey
                + (2 * sin_theta**2 * cos_theta**2) / laminate.Gxy
                + (2 * laminate.vxy * cos_theta**2 * sin_theta**2) / laminate.Ex
            )

            # Inverse to get E_theta
            E_theta = 1 / E_theta_inv

            # use the E_theta to find the FI in the given direction
            G_core = self.core.Gxbarz(np.deg2rad(theta))
            Ez = self.core.coreproperties["Ez"]
            t_core = self.core.h
            t_face = laminate.h

            symthick = abs(self.SymThickWrinkling(Ez, t_face, E_theta, G_core))

            asymthin = abs(self.SymThinWrinkling(Ez, t_core, t_face, E_theta, G_core))
            if symthick < asymthin:
                Nwrinkle = symthick
            else:
                Nwrinkle = asymthin

            FI = abs(laminate_stresses[max_negative_index] / Nwrinkle)

        else:
            FI = 0
        return FI

    def face_sheet_load_distribution(self) -> None:
        # Normal loads are as follows:
        Nx = self.loads.Nx
        Ny = self.loads.Ny

        # Divide normal loads between facesheets based on EA of facesheets

        # Assign Nx:
        Ext1 = self.bottom_laminate.Ex * self.bottom_laminate.h
        Ext2 = self.top_laminate.Ex * self.top_laminate.h

        Nx1 = Nx * (Ext1 / (Ext1 + Ext2))
        Nx2 = Nx * (Ext2 / (Ext1 + Ext2))

        # Assign Ny:
        Eyt1 = self.bottom_laminate.Ey * self.bottom_laminate.h
        Eyt2 = self.top_laminate.Ey * self.top_laminate.h

        Ny1 = Ny * (Eyt1 / (Eyt1 + Eyt2))
        Ny2 = Ny * (Eyt2 / (Eyt1 + Eyt2))

        # Divide shear loads between facesheets based on shear stifness GA of facesheets
        Gt1 = self.bottom_laminate.Gxy * self.bottom_laminate.h
        Gt2 = self.top_laminate.Gxy * self.top_laminate.h

        Ns1 = self.loads.Nxy * (Gt1 / (Gt1 + Gt2))
        Ns2 = self.loads.Nxy * (Gt2 / (Gt1 + Gt2))

        # Mx = self.loads[3]
        # My = self.loads[4]
        # Ms = self.loads[5]

        # moments are a bit more complicated but not strictly neccesary now
        # TODO: add facesheet loads due to moments

        self.bottom_laminate.Loads = [Nx1, Ny1, Ns1, 0, 0, 0]
        self.top_laminate.Loads = [Nx2, Ny2, Ns2, 0, 0, 0]

    def shear_load_wrinkling_Ncrit(self) -> float:
        t_face = self.bottom_laminate.h
        ABD_matrix45 = self.bottom_laminate.rotated_ABD(np.deg2rad(45))

        # In this case we check for shear buckling:
        vxy = ABD_matrix45[0, 1] / ABD_matrix45[1, 1]
        vyx = ABD_matrix45[0, 1] / ABD_matrix45[0, 0]
        D11f = ABD_matrix45[3, 3]

        # And we calculate the effective E modulus of the face sheet:
        E_face = (12 * (1 - vxy * vyx) * D11f) / (t_face**3)

        G_45 = self.core.Gxbarz(np.deg2rad(45))
        Ez = self.core.coreproperties["Ez"]
        t_core = self.core.thickness

        # Check which formula to use for wrinkling:
        symthick = self.SymThickWrinkling(Ez, t_face, E_face, G_45)

        asymthin = self.SymThinWrinkling(Ez, t_core, t_face, E_face, G_45)

        if symthick < asymthin:
            Nwrinkle = symthick
            print("SymThickWrinkling")
        else:
            Nwrinkle = asymthin
        return Nwrinkle

    def SymThickWrinkling(self, Ez: float, t_face: float, E_f: float, G_45: float) -> float:
        """Calculate crit load intensity for symmetric wrinkling thick laminates."""
        Ns_w = 0.43 * t_face * (E_f * Ez * G_45) ** (1 / 3)
        return Ns_w

    def SymThinWrinkling(
        self, Ez: float, t_core: float, t_face: float, E_f: float, G_45: float
    ) -> float:
        """Calculate crit load intensity for symmetric wrinkling thin laminates."""
        Ns_w = 0.816 * np.sqrt(E_f * Ez * t_face**3 / t_core) + G_45 * t_core / 6
        return Ns_w

    def AsymThickWrinkling(self, Ez: float, t_core: float, t_face: float, E_f: float) -> float:
        """Calculate crit load intensity for asymmetric wrinkling thick laminates."""
        Ns_w = 0.33 * t_face * E_f * np.sqrt(Ez * t_face / (E_f * t_core))
        return Ns_w

    def AsymThinWrinkling(
        self, Ez: float, t_core: float, t_face: float, E_f: float, G_45: float
    ) -> float:
        """Calculate crit load intensity for asymmetric wrinkling thin laminates."""
        Ns_w = 0.33 * t_face ** (3 / 2) * np.sqrt(Ez * E_f / t_core)
        return Ns_w
