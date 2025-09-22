import numpy as np
from typing import Optional, Tuple


class Vector:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class Node:
    def __init__(self, id: int, x: float, y: float, z: float):
        self.id = id
        self.x = x
        self.y = y
        self.z = z

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


class Vector:
    def __init__(self, x: float, y: float, z: float):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.norm: float = self.calculate_norm()

    def calculate_norm(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def is_orthogonal_to(self, other: "Vector", tol: float = 1e-6) -> bool:
        dot_product = self.x * other.x + self.y * other.y + self.z * other.z
        return abs(dot_product) < tol

    def dot(self, other: "Vector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


class Orientation:
    def __init__(self, vx: Vector, vy: Vector, vz: Vector):
        self.vx = vx
        self.vy = vy
        self.vz = vz

        self.orthogonality_tolerance = 1e-6

    def is_orthogonal(self) -> bool:
        return (
            self.vx.is_orthogonal_to(self.vy, self.orthogonality_tolerance)
            and self.vy.is_orthogonal_to(self.vz, self.orthogonality_tolerance)
            and self.vz.is_orthogonal_to(self.vx, self.orthogonality_tolerance)
        )

    def as_matrix(self) -> np.ndarray:
        """Return a 3x3 rotation matrix with columns [ex ey ez]."""
        return np.column_stack(
            (
                np.array([self.vx.x, self.vx.y, self.vx.z], dtype=float),
                np.array([self.vy.x, self.vy.y, self.vy.z], dtype=float),
                np.array([self.vz.x, self.vz.y, self.vz.z], dtype=float),
            )
        )


class Element:
    def __init__(self, id: int, nodes: list[Node], orientation: Orientation):
        self.id: int = id
        self.nodes: list[Node] = nodes
        self.orientation: Orientation = orientation


class CompositeElement(Element):
    def __init__(
        self,
        id: int,
        nodes: list[Node],
        orientation: Orientation,
        ABD: np.ndarray,
        As: Optional[np.ndarray] = None,
    ):
        super().__init__(id, nodes, orientation)
        self.ABD = ABD
        self.As = As
        # Compute local stiffness
        self.Ke_local, self.orientation, self.origin = plate4_mindlin_stiffness(
            ABD, self.nodes, orientation, As
        )
        # Build local->global transformation per node (5x5), then expand to 20x20
        R = self.orientation.as_matrix()
        Tn = np.zeros((5, 5), dtype=float)
        Tn[0:3, 0:3] = R  # translations
        Tn[3:5, 3:5] = R[0:2, 0:2]  # rotations about x,y (ignore rz)
        self.T_local_to_global = np.zeros((20, 20), dtype=float)
        for a in range(4):
            i0 = a * 5
            self.T_local_to_global[i0 : i0 + 5, i0 : i0 + 5] = Tn
        # Rotate to global
        self.Ke_global = self.T_local_to_global.T @ self.Ke_local @ self.T_local_to_global


def plate4_mindlin_stiffness(
    ABD: np.ndarray,
    nodes: list[Node],
    orientation: Orientation,
    As: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Orientation, np.ndarray]:
    """
    Build the 4-node Mindlin-Reissner plate/shell element stiffness matrix from ABD and node coords.

    Inputs:
      ABD: 6x6 laminate stiffness (A,B,D partitioned as ABD[:3,:3], ABD[:3,3:], ABD[3:,3:])
      nodes: a list of 4 Node objects (approximately coplanar)
      orientation: Orientation object defining local axes (ex, ey, ez)
      As: optional 2x2 transverse shear stiffness matrix (e.g., kappa * h * [[Gxz, 0],[0, Gyz]])

    Returns:
      Ke_local: (20,20) element stiffness in local plate axes with 5 dof/node [u, v, w, rx, ry]
      orientation: the same Orientation passed in (use orientation.as_matrix() for algebra)
      origin: (3,) element centroid in global coords

    Notes:
      - Element dofs per node: [u, v, w, rx, ry] in local axes.
      - Ke is returned in local axes. To assemble in a global system, keep a consistent element
        local frame per element or extend to a 6 dof/node shell formulation.
      - Requires nodes to be approximately coplanar.
    """
    dtype = ABD.dtype
    node_coordinates_global = np.array([n.as_array() for n in nodes], dtype=dtype)

    # Partition ABD
    A_extensional = ABD[0:3, 0:3]
    B_coupling = ABD[0:3, 3:6]
    D_bending = ABD[3:6, 3:6]

    # Local plate frame from provided orientation
    R = orientation.as_matrix()
    # Element centroid in global coordinates
    origin = node_coordinates_global.mean(axis=0)

    # Project nodes to local 2D plane
    nodes_local = (R.T @ (node_coordinates_global - origin).T).T  # shape (4,3)
    xy = nodes_local[:, :2].copy()  # (x,y) in local plane

    # 2x2 Gauss quadrature
    g = 1.0 / np.sqrt(3.0)
    gauss = [(-g, -g, 1.0), (g, -g, 1.0), (g, g, 1.0), (-g, g, 1.0)]

    nnode = 4
    dof_per_node = 5
    ndof = nnode * dof_per_node
    Ke = np.zeros((ndof, ndof), dtype=dtype)

    def shape_Q4(xi, eta):
        N = np.array(
            [
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta),
            ],
            dtype=dtype,
        )
        dN_dxi = np.array(
            [
                -0.25 * (1 - eta),
                0.25 * (1 - eta),
                0.25 * (1 + eta),
                -0.25 * (1 + eta),
            ],
            dtype=dtype,
        )
        dN_deta = np.array(
            [
                -0.25 * (1 - xi),
                -0.25 * (1 + xi),
                0.25 * (1 + xi),
                0.25 * (1 - xi),
            ],
            dtype=dtype,
        )
        return N, dN_dxi, dN_deta

    for xi, eta, wgt in gauss:
        N, dN_dxi, dN_deta = shape_Q4(xi, eta)

        # Jacobian (2x2) from local param -> local physical (x,y)
        J = np.zeros((2, 2), dtype=dtype)
        # x = sum N_i * x_i, etc.; J = [dx/dxi dy/dxi; dx/deta dy/deta]
        for i in range(nnode):
            x_i, y_i = xy[i, 0], xy[i, 1]
            J[0, 0] += dN_dxi[i] * x_i
            J[0, 1] += dN_dxi[i] * y_i
            J[1, 0] += dN_deta[i] * x_i
            J[1, 1] += dN_deta[i] * y_i

        detJ = np.linalg.det(J)
        if detJ <= np.finfo(float).eps:
            raise ValueError("Element Jacobian is singular or inverted.")
        invJ = np.linalg.inv(J)

        # Spatial derivatives dN/dx, dN/dy
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        # Build B matrices
        Bm = np.zeros((3, ndof), dtype=dtype)  # membrane
        Bb = np.zeros((3, ndof), dtype=dtype)  # bending (curvatures)
        Bs = np.zeros((2, ndof), dtype=dtype)  # transverse shear

        for a in range(nnode):
            col = a * dof_per_node
            # Membrane strains: [ex, ey, gxy] = [u,x, v,y, u,y+v,x]
            Bm[0, col + 0] = dN_dx[a]  # u,x
            Bm[1, col + 1] = dN_dy[a]  # v,y
            Bm[2, col + 0] = dN_dy[a]  # u,y
            Bm[2, col + 1] = dN_dx[a]  # v,x

            # Bending curvatures: [kx, ky, kxy] = [rx,x, ry,y, rx,y+ry,x]
            Bb[0, col + 3] = dN_dx[a]  # rx,x
            Bb[1, col + 4] = dN_dy[a]  # ry,y
            Bb[2, col + 3] = dN_dy[a]  # rx,y
            Bb[2, col + 4] = dN_dx[a]  # ry,x

            # Transverse shear: [gxz, gyz] = [rx + w,x, ry + w,y]
            Bs[0, col + 2] = dN_dx[a]  # w,x
            Bs[0, col + 3] = N[a]  # rx
            Bs[1, col + 2] = dN_dy[a]  # w,y
            Bs[1, col + 4] = N[a]  # ry

        # Integrate stiffness at Gauss point
        dA = detJ * wgt
        Kgp = (
            Bm.T @ A_extensional @ Bm
            + Bm.T @ B_coupling @ Bb
            + Bb.T @ B_coupling.T @ Bm
            + Bb.T @ D_bending @ Bb
        )
        if As is not None:
            Kgp += Bs.T @ As @ Bs

        Ke += Kgp * dA
    return Ke, orientation, origin


if __name__ == "__main__":
    # Two-element assembly example: build laminate, make 2 quads side-by-side, assemble global K
    from structures.panel.utils import laminate_builder

    # Build a symmetric quasi-isotropic laminate (example material set in data)
    laminate = laminate_builder([0, 90, 45, -45], True, True, 1, type="T700")
    ABD = laminate.ABD_matrix

    # Optional shear stiffness (commented out for simplicity)
    # kappa = 5.0 / 6.0
    # h = laminate.h
    # G = laminate.Gxy  # rough approx for Gxz, Gyz if needed
    # As = kappa * h * np.diag([G, G])
    As = None

    # Define 6 unique global Node instances for two unit quads in XY-plane (z=0)
    # Node layout (indices in parentheses):
    # (0)---(1)---(2)
    #  |  e0 | e1  |
    # (3)---(4)---(5)
    all_nodes: list[Node] = [
        Node(0, 0.0, 0.0, 0.0),  # 0
        Node(1, 1.0, 0.0, 0.0),  # 1
        Node(2, 2.0, 0.0, 0.0),  # 2
        Node(3, 0.0, 1.0, 0.0),  # 3
        Node(4, 1.0, 1.0, 0.0),  # 4
        Node(5, 2.0, 1.0, 0.0),  # 5
    ]

    # Element connectivities (counter-clockwise):
    # e0: (0,1,4,3); e1: (1,2,5,4)
    element_connectivities = [
        (0, 1, 4, 3),
        (1, 2, 5, 4),
    ]

    # Define a common Orientation aligned with global axes (XY plane)
    ex = Vector(1.0, 0.0, 0.0)
    ey = Vector(0.0, 1.0, 0.0)
    ez = Vector(0.0, 0.0, 1.0)
    global_xy_orientation = Orientation(ex, ey, ez)

    dof_per_node = 5
    ndof = len(all_nodes) * dof_per_node
    K_global = np.zeros((ndof, ndof), dtype=ABD.dtype)

    def dof_indices(conn: Tuple[int, int, int, int]) -> np.ndarray:
        idx = []
        for n_id in conn:
            base = n_id * dof_per_node
            idx.extend([base + 0, base + 1, base + 2, base + 3, base + 4])
        return np.array(idx, dtype=int)

    # Assemble
    first_orientation_matrix = None
    first_origin = None
    for conn in element_connectivities:
        element_nodes = [all_nodes[i] for i in conn]
        Ke_local, orient, origin = plate4_mindlin_stiffness(
            ABD, element_nodes, global_xy_orientation, As=As
        )
        gdofs = dof_indices(conn)
        # Elements lie in XY plane and orientation matches global axes -> assemble directly
        K_global[np.ix_(gdofs, gdofs)] += Ke_local
        if first_orientation_matrix is None:
            first_orientation_matrix, first_origin = orient.as_matrix(), origin

    # Print a brief summary
    np.set_printoptions(precision=3, suppress=True)
    print("Global stiffness K shape:", K_global.shape)
    print("First element centroid (global):", first_origin)
    print("First element local axes (columns of R):\n", first_orientation_matrix)
    print("K_global top-left 12x12 block:\n", K_global[:12, :12])
