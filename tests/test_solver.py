import numpy as np
import pytest

from structures.FEM.plate_element import Node, Vector, Orientation, CompositeElement
from structures.FEM.solver import FEMSolver


def test_one_element_uniaxial_membrane_extension():
    # Analytical setup: unit square (0<=x,y<=1), thickness t, Ex along x, nu=0
    E = 1000.0
    t = 1.0
    G = 400.0  # not used when gamma_xy=0, but keep A66 positive

    # Build ABD: only A block nonzero, B=D=0, nu=0 => A12=0
    ABD = np.zeros((6, 6), dtype=float)
    A = np.diag([E * t, E * t, G * t])  # A11, A22, A66
    ABD[0:3, 0:3] = A

    # Element nodes: unit square CCW (ids must be 0..3 for assembly)
    nodes = [
        Node(0, 0.0, 0.0, 0.0),
        Node(1, 1.0, 0.0, 0.0),
        Node(2, 1.0, 1.0, 0.0),
        Node(3, 0.0, 1.0, 0.0),
    ]

    # Orientation aligned to global axes
    orientation = Orientation(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0))

    # Build composite element (no shear/bending needed for membrane test)
    element = CompositeElement(id=0, nodes=nodes, orientation=orientation, ABD=ABD, As=None)

    # FEM model
    solver = FEMSolver(n_nodes=len(nodes), dof_per_node=5)
    solver.add_composite_element(element)

    # Boundary conditions:
    # - Fix out-of-plane DOFs everywhere to avoid singularities (w, rx, ry)
    for nid in range(len(nodes)):
        solver.fix(nid, "w", 0.0)
        solver.fix(nid, "rx", 0.0)
        solver.fix(nid, "ry", 0.0)
    # - Fix left edge (nodes 0 & 3) in-plane DOFs to remove rigid body motion
    for nid in (0, 3):
        solver.fix(nid, "u", 0.0)
        solver.fix(nid, "v", 0.0)

    # Loads: +Fx at the two right-edge nodes (1 & 2)
    p = 10.0
    for nid in (1, 2):
        solver.load(nid, "u", p)

    # Solve
    u, K, f = solver.solve()

    # Analytical displacement on right edge: u = epsilon_x * 1, epsilon_x = N_x / (E*t)
    # Total force = 2p; unit edge length => average N_x = 2p
    u_analytical = 2.0 * p / (E * t)

    # Extract FE displacements at nodes 1 and 2 (u component)
    dpn = 5
    u1 = u[1 * dpn + 0]
    u2 = u[2 * dpn + 0]

    # Compare within a small tolerance
    assert np.isclose(u1, u_analytical, rtol=1e-5, atol=1e-8)
    assert np.isclose(u2, u_analytical, rtol=1e-5, atol=1e-8)
