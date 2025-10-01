from dataclasses import dataclass


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
        msf: MaterialProperties safety/compensation factor (e.g., 1.3 for GFRP, 1.1 for CFRP).
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


@dataclass(frozen=True)
class CoreProperties:
    """Core material properties for sandwich panels.

    Attributes:
        Ez: modulus in z direction.
        Sxz: shear strength xz direction (transverse out of plane)
        Gxz: shear modulus xz direction (transverse out of plane)
        Syz: shear strength yz direction (transverse out of plane)
        Gyz: shear modulus yz direction (transverse out of plane)
        Xc: compressive strength???????????????????????????? #TODO: figure out
        rho: Density (mass per unit volume, e.g., g/mm³).
    """

    Ez: float
    Sxz: float
    Gxz: float
    Syz: float
    Gyz: float
    Xc: float
    rho: float


@dataclass(frozen=True)
class MaterialProperties:
    """Composite material definition combining elastic and failure properties.

    Attributes:
        name: Name of the material.
        t: Ply thickness (mm or consistent units).
        rho: Density (mass per unit volume, e.g., g/mm³).
        elastic_properties: Instance of ElasticProperties defining elastic behavior.
        failure_properties: Instance of FailureProperties defining failure criteria.
    """

    name: str
    t: float
    rho: float
    elastic_properties: ElasticProperties
    failure_properties: FailureProperties


dataclass(frozen=True)


class PanelLoadCase:
    """
    Load case for a panel type object, namely sandwich or laminate.

    Attributes:
        Nx: normal load intensity in x direction (N/mm)
        Ny: normal load intensity in y direction (N/mm)
        Nxy: shear load intensity in xy direction (N/mm)
        Mx: bending moment intensity (bending the x axis around the y axis)
        My: bending moment intensity (bending the y axis around the x axis)
        Mxy: shear moment intensity (bending moment around the x or y axis)

    ┌─────►
    │    x
    │       Nxy  ▲
    │y       ◄───┤Ny
    ▼     ┌──────┴──────┐
          │             │
      Nxy▲│             │
         ││             │ Nx
      ◄──┴┤             ├┬──►
       Nx │             ││
          │             ││
          │             │▼Nxy
          └──────┬──────┘
               Ny├──►
                 ▼  Nxy
    """

    Nx: float
    Ny: float
    Nxy: float
    Mx: float
    My: float
    Mxy: float
