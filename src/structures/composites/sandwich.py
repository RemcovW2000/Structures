import numpy as np

from structures.composites.base_components.core import Core
from structures.composites.data_utils import PanelLoads, PanelStrains
from structures.composites.laminate import Laminate
from structures.composites.math_utils import rotation_matrix
from structures.composites.Panel import Panel, calculate_ABD_matrix
from structures.structural_entity import FailureMode, StructuralEntity, failure_analysis

WRINKLING_ANGLE_STEP = 5  # degrees


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

        wrinkling_fi_bottom = 0.0
        wrinkling_fi_top = 0.0
        crimping_fi = 0.0
        dimpling_fi_top = 0.0
        dimpling_fi_bottom = 0.0
        for theta in range(0, 180, WRINKLING_ANGLE_STEP):
            w_fi_bot = self.wrinkling_analysis(self.bottom_laminate, theta)
            w_fi_top = self.wrinkling_analysis(self.top_laminate, theta)

            if w_fi_bot > wrinkling_fi_bottom:
                wrinkling_fi_bottom = w_fi_bot
            if w_fi_top > wrinkling_fi_top:
                wrinkling_fi_top = w_fi_top

            crimping_fi = self.crimping_analysis(np.deg2rad(theta))
            dimpling_fi_top = self.dimpling_analysis(
                laminate=self.top_laminate, theta=np.deg2rad(theta)
            )
            dimpling_fi_bottom = self.dimpling_analysis(
                laminate=self.bottom_laminate, theta=np.deg2rad(theta)
            )

        return [
            ("wrinkling_bottom", wrinkling_fi_bottom),
            ("wrinkling_top", wrinkling_fi_top),
            ("crimping", crimping_fi),
            ("dimpling_top", dimpling_fi_top),
            ("dimpling_bottom", dimpling_fi_bottom),
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

    def assign_facesheet_strains(self) -> None:
        """Sets strains for facesheets based on sandwich panel strains."""
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

    def buckling_scaling_factor(self, Ncrit: float) -> float:
        """
        Returns scaling factor by which to multiply panel buckling load.

        Compensates for core transverse shearing.
        """
        # TODO: take into account directionality of buckling
        Nxcrit = Ncrit / (1 + Ncrit / (self.core.h * self.core.properties.Gxz))
        return Nxcrit

    def wrinkling_analysis(self, laminate: Laminate, theta: float) -> float:
        """
        Perform wrinkling analysis at specific angle.

        Finds laminates with compressive loads.
        """
        # TODO: take into account assymetric laminates
        loads_vector = np.array(
            [laminate.loads.Nx, laminate.loads.Ny, laminate.loads.Nxy], dtype=float
        )
        loads_vector_rotated = rotation_matrix(np.deg2rad(theta)) @ loads_vector
        Nx_rotated = loads_vector_rotated[0]

        if Nx_rotated > 0:
            return 0.0

        E_rotated = laminate.calculate_equivalent_properties_rotated(theta).E1

        G_core = self.core.Gxbarz(np.deg2rad(theta))
        Ez = self.core.properties.Ez
        t_core = self.core.h
        t_face = laminate.h

        # symmetric:
        z_c_sym = 0.91 * t_face * (Ez * E_rotated / (G_core**2)) ** (1 / 3)
        if t_core >= 2 * z_c_sym:
            sym_Nwrinkle = self.SymThickWrinkling(Ez, t_face, E_rotated, G_core)
        else:
            sym_Nwrinkle = self.SymThinWrinkling(Ez, t_core, t_face, E_rotated, G_core)

        # antisymmetric:
        z_c_antisym = 1.5 * t_face * (Ez * E_rotated / (G_core**2)) ** (1 / 3)
        if t_core >= 2 * z_c_antisym:
            asym_Nwrinkle = self.AsymThickWrinkling(Ez, t_core, t_face, E_rotated)
        else:
            asym_Nwrinkle = self.AsymThinWrinkling(Ez, t_core, t_face, E_rotated, G_core)

        Nwrinkle = min(sym_Nwrinkle, asym_Nwrinkle)

        fi = float(abs(Nx_rotated / Nwrinkle))
        return fi

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

    def crimping_analysis(self, theta: float) -> float:
        loads_vector = np.array([self.loads.Nx, self.loads.Ny, self.loads.Nxy], dtype=float)
        loads_vector_rotated = rotation_matrix(np.deg2rad(theta)) @ loads_vector
        Nx_rotated = loads_vector_rotated[0]

        if Nx_rotated > 0:
            return 0.0

        Ncrimping = self.core.h * self.core.Gxbarz(theta)
        fi = float(abs(Nx_rotated / Ncrimping))
        return fi

    def dimpling_analysis(self, laminate: Laminate, theta: float) -> float:
        """Perform dimpling analysis."""
        loads_vector = np.array([self.loads.Nx, self.loads.Ny, self.loads.Nxy], dtype=float)
        loads_vector_rotated = rotation_matrix(np.deg2rad(theta)) @ loads_vector
        Nx_rotated = loads_vector_rotated[0]

        if Nx_rotated > 0:
            return 0.0

        rotated_properties = laminate.calculate_equivalent_properties_rotated(theta)
        E_rotated = rotated_properties.E1
        vxy_rotated = rotated_properties.v12
        vyx_rotated = vxy_rotated * (rotated_properties.E2 / E_rotated)

        Nx_dim = (
            2
            * ((E_rotated * laminate.h**3) / (1 - vxy_rotated * vyx_rotated))
            / self.core.properties.cell_diameter**2
        )
        fi = float(abs(Nx_rotated / Nx_dim))
        return fi
