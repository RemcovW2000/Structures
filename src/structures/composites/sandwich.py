import numpy as np

from structures.composites.base_components.core import Core
from structures.composites.data_utils import PanelLoads, PanelStrains
from structures.composites.laminate import Laminate
from structures.composites.Panel import Panel, calculate_ABD_matrix
from structures.structural_entity import FailureMode, StructuralEntity, failure_analysis


class Sandwich(StructuralEntity, Panel):
    def __init__(
        self,
        bottom_laminate: Laminate,
        top_laminate: Laminate,
        core: Core,
        loads: PanelLoads = None,
        strains: PanelStrains = None,
    ):
        StructuralEntity.__init__(self)

        self.bottom_laminate: Laminate = bottom_laminate  # bottom laminate
        self.top_laminate: Laminate = top_laminate  # top laminate
        self.core: Core = core
        self.sandwich: bool = True

        self.loads: PanelLoads = loads
        self.strains: PanelStrains = strains
        self.h: float = self.bottom_laminate.h + self.top_laminate.h + self.core.h

        Panel.__init__(self)

    @property
    def child_objects(self) -> list[StructuralEntity]:
        return [self.bottom_laminate, self.top_laminate]

    @calculate_ABD_matrix
    def calculate_ABD_matrix(self) -> np.ndarray:
        """
        Calculates the ABD matrix for the sandwich structure
        """
        bottom_ABD = self.bottom_laminate.calculate_ABD_offset(
            offset=-(self.core.h + self.bottom_laminate.h) / 2
        )
        top_ABD = self.top_laminate.calculate_ABD_offset(
            offset=(self.core.h + self.top_laminate.h) / 2
        )

        totalABD = bottom_ABD + top_ABD

        self.ABD_matrix = totalABD
        self.A_matrix = totalABD[0:3, 0:3]
        self.B_matrix = totalABD[0:3, 3:6]
        self.D_matrix = totalABD[3:6, 3:6]
        return totalABD

    def strains_from_loads(self) -> PanelStrains:
        return PanelStrains(np.linalg.inv(self.ABD_matrix) @ self.loads.array)

    def loads_from_strains(self) -> PanelLoads:
        return PanelLoads(self.ABD_matrix @ self.strains.array)

    @failure_analysis
    def failure_analysis(self) -> list[FailureMode]:
        """Perform failure analysis on sandwich panel."""
        # calculate loads on facesheets:
        self.assign_facesheet_strains()

        # check first ply failure for both facesheets:
        max_fi1 = self.bottom_laminate.fi
        max_fi2 = self.top_laminate.fi
        first_ply_failure = max(max_fi1, max_fi2)

        # check for wrinkling in both facesheets:
        l1_stresses, l1_directions = self.bottom_laminate.principal_stresses_and_directions()

        l2_stresses, l2_directions = self.top_laminate.principal_stresses_and_directions()

        # using the directions we can find the material strength in that direction
        # wrinklingFI1 = self.wrinkling_analysis(l1_stresses, l1_directions, self.bottom_laminate)
        # wrinklingFI2 = self.wrinkling_analysis(l2_stresses, l2_directions, self.top_laminate)

        return [
            # ("wrinkling", wrinklingFI1),
            # ("wrinkling", wrinklingFI2),
            ("first_ply_failure", first_ply_failure),
        ]

    def calculate_weight_per_area(self) -> float:
        """Calculates weight per unit area of laminate."""
        return np.sum(
            [
                self.bottom_laminate.calculate_weight_per_A(),
                self.top_laminate.calculate_weight_per_A(),
                self.core.calculate_weight_per_A(),
            ]
        )

    def buckling_scaling_factor(self, Ncrit: float) -> float:
        """
        Returns scaling factor by which to multiply panel buckling load.

        Compensates for core transverse shearing.
        """
        # TODO: take into account directionality of buckling
        Nxcrit = Ncrit / (1 + Ncrit / (self.core.h * self.core.properties.Gxz))
        return Nxcrit

    def wrinkling_analysis(
        self, laminate_stresses: np.ndarray, laminate_directions: np.ndarray, laminate: Laminate
    ) -> float:
        """
        Perform wrinkling analysis.

        Laminates have Loads assigned.
        """
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
            Ez = self.core.properties.Ez
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

    def assign_facesheet_strains(self) -> None:
        """Sets loads for facesheets."""
        strains = self.strains.array
        Sx_top = strains[0] + (self.core.h / 2 + self.top_laminate.h / 2) * strains[3]
        Sy_top = strains[1] + (self.core.h / 2 + self.top_laminate.h / 2) * strains[4]
        Sxy_top = strains[2] + (self.core.h / 2 + self.top_laminate.h / 2) * strains[5]

        Sx_bot = strains[0] - (self.core.h / 2 + self.bottom_laminate.h / 2) * strains[3]
        Sy_bot = strains[1] - (self.core.h / 2 + self.bottom_laminate.h / 2) * strains[4]
        Sxy_bot = strains[2] - (-self.core.h / 2 + self.bottom_laminate.h / 2) * strains[5]

        Kx = strains[3]
        Ky = strains[4]
        Kxy = strains[5]

        self.top_laminate.strains = PanelStrains(np.array([Sx_top, Sy_top, Sxy_top, Kx, Ky, Kxy]))
        self.bottom_laminate.strains = PanelStrains(
            np.array([Sx_bot, Sy_bot, Sxy_bot, Kx, Ky, Kxy])
        )

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
        Ez = self.core.properties.Ez
        t_core = self.core.h

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
