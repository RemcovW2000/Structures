import numpy as np

from structures.composites.data.lamina_props import Christos
from structures.composites.utils import laminate_builder
from structures.FEM.plate_element import CompositeElement, Node, Orientation, Vector
from structures.FEM.solver import FEMSolver


def test_integration_laminate_one_element_inplane_extension() -> None:
    """Build symmetric quasi-isotropic laminate (T700 data), get ABD properties."""
    laminate = laminate_builder(
        [0, 90, 45, -45], symmetry=True, copycenter=True, multiplicity=1, material_props=Christos
    )
    ABD = laminate.ABD_matrix
    h = float(laminate.h)
    Ex = float(laminate.Ex)
    Gxy = float(laminate.Gxy)

    # Shear stiffness for Mindlin transverse shear (avoid singular w)
    kappa = 5.0 / 6.0
    As = kappa * h * np.diag([Gxy, Gxy]).astype(float)

    # Single unit quad element (counter-clockwise)
    nodes = [
        Node(0, 0.0, 0.0, 0.0),
        Node(1, 1.0, 0.0, 0.0),
        Node(2, 1.0, 1.0, 0.0),
        Node(3, 0.0, 1.0, 0.0),
    ]

    # Orientation aligned to global XY
    orientation = Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0))

    # Element and solver
    element = CompositeElement(id=0, nodes=nodes, orientation=orientation, ABD=ABD, As=As)
    solver = FEMSolver(n_nodes=len(nodes), dof_per_node=5)
    solver.add_composite_element(element)

    # Fix out-of-plane DOFs and left edge in-plane DOFs
    for nid in range(len(nodes)):
        solver.fix(nid, "w", 0.0)
        solver.fix(nid, "rx", 0.0)
        solver.fix(nid, "ry", 0.0)
    for nid in (0, 3):
        solver.fix(nid, "u", 0.0)
        solver.fix(nid, "v", 0.0)

    # Apply in-plane x-loads at right edge nodes
    p = 100.0
    for nid in (1, 2):
        solver.load(nid, "u", p)

    u, K, f = solver.solve()

    # Analytical right-edge displacement approximation: u = N_x/(Ex*h)*L, with N_x = total force per unit width
    u_analytical = 2.0 * p / (Ex * h)

    dpn = 5
    u1 = u[1 * dpn + 0]
    u2 = u[2 * dpn + 0]

    # Allow moderate tolerance due to element formulation and single-element mesh
    assert np.isfinite(u1) and np.isfinite(u2)
    assert u1 > 0 and u2 > 0
    assert np.isclose(u1, u_analytical, rtol=0.2, atol=0.0)
    assert np.isclose(u2, u_analytical, rtol=0.2, atol=0.0)
