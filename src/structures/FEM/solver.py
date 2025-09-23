from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from structures.FEM.mesh import Mesh

# Simple node/element-based FEM solver for plate/shell elements -----------------


class DofName(enum.Enum):
    U = ("u", 0)
    V = ("v", 1)
    W = ("w", 2)
    RX = ("rx", 3)
    RY = ("ry", 4)

    def __new__(cls, description: str, index: int) -> DofName:
        obj = object.__new__(cls)
        obj.index = index
        obj.description = description
        return obj


@dataclass(frozen=True)
class Dirichlet:
    node_id: int
    dof: DofName
    value: float = 0.0


@dataclass(frozen=True)
class NodalLoad:
    node_id: int
    dof: DofName
    value: float


@dataclass(frozen=True)
class NodalDeformation:
    """Per-node deformation container for convenience mapping.

    Holds the original Node reference and all 5 DOF components.
    """

    node: Node
    u: float
    v: float
    w: float
    rx: float
    ry: float


class FEMSolver:
    """Minimal node/element FEM solver for plate elements.

    - DOFs per node (Mindlin): [u, v, w, rx, ry] => 5 DOF/node
    - Add elements via add_element(Ke, connectivity)
    - Apply BCs via fix(node, dof[, value])
    - Apply loads via load(node, dof, value)
    - Call solve() to obtain full displacement vector (including fixed DOFs)

    Contract:
      - Node ids are 0-based and consistent with element connectivities.
      - Ke is in local element axes; in this demo, local == global.
      - Assembling multiple elements is supported (same dof_per_node across).
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.n_nodes = len(mesh.nodes)
        self.dpn = mesh.dof_per_node
        self.no_dof = mesh.no_dof
        self.K = np.zeros((self.no_dof, self.no_dof), dtype=float)
        self.f = np.zeros(self.no_dof, dtype=float)
        self._dirichlet: list[Dirichlet] = []
        self.u: Optional[np.ndarray] = None
        self._elements: list[tuple[int, ...]] = []  # store element connectivities
        self.nodal_deformations: dict[int, NodalDeformation] = {}

        self.assemble_stifness_matrix(self.mesh)

    # --- DOF mapping -----------------------------------------------------

    def dof_index(self, node_id: int, dof: DofName) -> int:
        return node_id * self.dpn + dof.index

    def element_dof_indices(self, connectivity: Iterable[int]) -> np.ndarray:
        idx: list[int] = []
        for nid in connectivity:
            base = nid * self.dpn
            idx.extend([base + i for i in range(self.dpn)])
        return np.array(idx, dtype=int)

    # --- Assembly --------------------------------------------------------
    def add_element(self, Ke: np.ndarray, connectivity: Iterable[int]) -> None:
        edofs = self.element_dof_indices(connectivity)
        self.K[np.ix_(edofs, edofs)] += Ke
        self._elements.append(tuple(int(i) for i in connectivity))

    def add_composite_element(self, element: CompositeElement) -> None:
        """Assemble a CompositeElement using its globally rotated stiffness.

        Requires element.Ke_global to be present (as provided by CompositeElement).
        """
        Ke = element.Ke_global
        conn = tuple(n.id for n in element.nodes)
        self.add_element(Ke, connectivity=conn)

    def assemble_stifness_matrix(self, mesh: Mesh) -> None:
        for element in mesh.elements:
            self.add_composite_element(element)

    # --- Loads & BCs -----------------------------------------------------
    def load(self, node_id: int, dof: DofName, value: float) -> None:
        self.f[self.dof_index(node_id, dof)] += value

    def fix(self, node_id: int, dof: DofName, value: float = 0.0) -> None:
        self._dirichlet.append(Dirichlet(node_id, dof, value))

    def fix_all_at_node(self, node_id: int, value: float = 0.0) -> None:
        for dof in DofName:
            self.fix(node_id, dof, value)

    # --- Solve -----------------------------------------------------------
    def solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve K u = f with Dirichlet constraints.

        Returns:
          - u: full displacement vector (size no_dof)
          - K: assembled global stiffness
          - f: assembled global load vector
        """
        if not self._dirichlet:
            raise ValueError("No Dirichlet BCs set; system may be singular.")

        fixed = np.array([self.dof_index(b.node_id, b.dof) for b in self._dirichlet], dtype=int)
        fixed_values = np.array([b.value for b in self._dirichlet], dtype=float)
        free = np.setdiff1d(np.arange(self.no_dof, dtype=int), fixed, assume_unique=False)

        # Partition
        K_ff = self.K[np.ix_(free, free)]
        K_fc = self.K[np.ix_(free, fixed)]
        f_f = self.f[free] - K_fc @ fixed_values

        # Solve reduced system
        u_f = np.linalg.solve(K_ff, f_f)

        # Full solution
        u = np.zeros(self.no_dof, dtype=float)
        u[free] = u_f
        u[fixed] = fixed_values
        self.u = u
        return u, self.K, self.f

    # --- Post-processing ----------------w--------------------------------
    def get_component(self, dof: DofName) -> np.ndarray:
        """Return a nodal array of a displacement/rotation component.

        dof: one of DofName.{U,V,W,RX,RY}
        """
        if self.u is None:
            raise RuntimeError("No solution available; call solve() first.")
        j = dof.index
        return np.array([self.u[i * self.dpn + j] for i in range(self.n_nodes)])

    def build_nodal_deformations(self) -> dict[int, NodalDeformation]:
        """Populate and return a mapping of node.id to NodalDeformation.

        Requires a solved system (self.u not None). The provided nodes must
        have ids that match the assembly ordering (0..n_nodes-1).
        """
        nodes = []
        mapping: dict[int, NodalDeformation] = {}
        for node in nodes:
            base = node.id * self.dpn
            nd = NodalDeformation(
                node=node,
                u=float(self.u[base + DofName.U.index]),
                v=float(self.u[base + DofName.V.index]),
                w=float(self.u[base + DofName.W.index]),
                rx=float(self.u[base + DofName.RX.index]),
                ry=float(self.u[base + DofName.RY.index]),
            )
            mapping[node.id] = nd
        self.nodal_deformations = mapping
        return mapping

    def plot_deformed(self, nodes_xyz: np.ndarray, scale: float = 1.0) -> None:
        """Plot undeformed and deformed mesh (Z displaced by scale * w) for all added elements.

        Inputs:
          nodes_xyz: (n_nodes, 3) coordinates array
          scale: visual scale factor for w displacement (adds to z)
        """
        if self.u is None:
            raise RuntimeError("No solution available; call solve() first.")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        w = self.get_component(DofName.W)
        XYZ = np.asarray(nodes_xyz, dtype=float)
        XYZ_def = XYZ.copy()
        XYZ_def[:, 2] += scale * w

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Build faces from stored elements (supports 3- and 4-node elements)
        faces = [list(conn) for conn in self._elements]
        if faces:
            polys0 = [XYZ[np.array(face, dtype=int)] for face in faces]
            ax.add_collection3d(
                Poly3DCollection(polys0, facecolor=(0.7, 0.7, 0.7, 0.2), edgecolor="k")
            )

            polys1 = [XYZ_def[np.array(face, dtype=int)] for face in faces]
            ax.add_collection3d(
                Poly3DCollection(polys1, facecolor=(0.1, 0.3, 0.8, 0.6), edgecolor="k")
            )

        ax.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], color="k", s=20, label="undeformed")
        ax.scatter(XYZ_def[:, 0], XYZ_def[:, 1], XYZ_def[:, 2], color="r", s=30, label="deformed")

        ax.set_title(f"Mindlin plate: deformed (w scaled x{scale:.0f})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper right")

        # Fit limits
        xmin, ymin, zmin = XYZ.min(axis=0)
        xmax, ymax, zmax = XYZ.max(axis=0)
        zmin = min(zmin, XYZ_def[:, 2].min())
        zmax = max(zmax, XYZ_def[:, 2].max())
        pad = 0.2 * max(xmax - xmin, ymax - ymin, 1.0)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_zlim(zmin - pad, zmax + pad)

        import matplotlib.pyplot as plt  # noqa: E402 keep local for optional dep

        plt.tight_layout()
        plt.show()


# Demo: 1-element plate with BC/loads API -------------------------------------
if __name__ == "__main__":
    from structures.FEM.plate_element import (
        CompositeElement,
        Node,
        Orientation,
        Vector,
    )
    from structures.panel.utils import laminate_builder

    # Build symmetric quasi-isotropic laminate and compute ABD
    laminate = laminate_builder(
        [0, 0, 0], symmetry=True, copycenter=True, multiplicity=1, type="T700"
    )
    ABD = laminate.ABD_matrix

    # Choose a simple transverse shear stiffness to avoid singular w-DOFs
    # For demo: Gxz≈Gyz≈Gxy, As = kappa*h*diag([Gxz, Gyz])
    kappa = 5.0 / 6.0
    h = float(laminate.h)
    G = float(getattr(laminate, "Gxy", 1.0e3))  # fallback if not present
    As = kappa * h * np.diag([G, G]).astype(ABD.dtype)

    # Unit square element nodes (counter-clockwise)
    nodes = [
        Node(0, 0.0, 0.0, 0.0),
        Node(1, 1.0, 0.0, 0.0),
        Node(2, 1.0, 1.0, 0.0),
        Node(3, 0.0, 1.0, 0.0),
    ]

    # Local plate orientation aligned with global axes
    orientation = Orientation(Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1))

    # Build element (Ke computed inside CompositeElement)
    element = CompositeElement(id=0, nodes=nodes, orientation=orientation, ABD=ABD, As=As)

    mesh = Mesh(nodes=nodes, elements=[element])

    # Build solver for 4 nodes x 5 dof/node
    solver = FEMSolver(mesh)

    # Boundary conditions:
    # - Fix node 0 and node 1 fully (all 5 DOFs)
    solver.fix_all_at_node(0)
    solver.fix_all_at_node(1)

    # Loads:
    # - Apply downward load to w-DOF at nodes 1 and 2
    solver.load(1, DofName.W, -1.0)
    solver.load(2, DofName.W, -1.0)

    # Solve
    u, K, f = solver.solve()

    # Plot
    XYZ = np.array([[n.x, n.y, n.z] for n in nodes])
    # Build nodal deformation map and plot
    solver.build_nodal_deformations()
    solver.plot_deformed(XYZ, scale=1000.0)
