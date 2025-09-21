import numpy as np
from typing import Optional, Tuple


def plate4_mindlin_stiffness(
    ABD: np.ndarray,
    nodes_xyz: np.ndarray,
    As: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the 4-node Mindlin-Reissner plate/shell element stiffness matrix from ABD and node coords.

    Inputs:
      ABD: 6x6 laminate stiffness (A,B,D partitioned as ABD[:3,:3], ABD[:3,3:], ABD[3:,3:])
      nodes_xyz: (4,3) array of node coordinates in 3D
      As: optional 2x2 transverse shear stiffness matrix (e.g., kappa * h * [[Gxz, 0],[0, Gyz]])

    Returns:
      Ke_local: (20,20) element stiffness in local plate axes with 5 dof/node [u, v, w, rx, ry]
      R: (3,3) local-to-global rotation matrix [ex ey ez] as columns
      origin: (3,) element centroid in global coords

    Notes:
      - Element dofs per node: [u, v, w, rx, ry] in local axes.
      - Ke is returned in local axes. To assemble in a global system, keep a consistent element
        local frame per element or extend to a 6 dof/node shell formulation.
      - Requires nodes to be approximately coplanar.
    """
    # Validate input
    if ABD.shape != (6, 6):
        raise ValueError("ABD must be 6x6.")
    if nodes_xyz.shape != (4, 3):
        raise ValueError("This implementation supports 4-node quads only with shape (4,3).")
    if As is not None and As.shape != (2, 2):
        raise ValueError("As must be 2x2 when provided.")

    dtype = ABD.dtype
    nodes_xyz = np.asarray(nodes_xyz, dtype=dtype)

    # Partition ABD
    A = ABD[0:3, 0:3]
    B = ABD[0:3, 3:6]
    D = ABD[3:6, 3:6]

    # Build local plate frame from best-fit plane (PCA)
    origin = nodes_xyz.mean(axis=0)
    X = nodes_xyz - origin
    C = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(C)
    # Largest two eigenvectors span the plane
    idx = np.argsort(eigvals)[::-1]
    ex = eigvecs[:, idx[0]]
    ey = eigvecs[:, idx[1]]
    # Ensure orthonormal, right-handed frame
    ex = ex / np.linalg.norm(ex)
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    R = np.column_stack((ex, ey, ez))  # columns are local axes in global coords

    # Project nodes to local 2D plane
    nodes_local = (R.T @ (nodes_xyz - origin).T).T  # shape (4,3)
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
        Kgp = Bm.T @ A @ Bm + Bm.T @ B @ Bb + Bb.T @ B.T @ Bm + Bb.T @ D @ Bb
        if As is not None:
            Kgp += Bs.T @ As @ Bs

        Ke += Kgp * dA

    return Ke, R, origin


class Vector:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class Orientation:
    def __init__(self, ex: Vector, ey: Vector, ez: Vector):
        self.ex = ex
        self.ey = ey
        self.ez = ez

    def is_orthogonal(self) -> bool:
        # Check if the vectors are orthogonal and normalized
        def dot(v1: Vector, v2: Vector) -> float:
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

        def norm(v: Vector) -> float:
            return (v.x**2 + v.y**2 + v.z**2) ** 0.5

        return (
            abs(dot(self.ex, self.ey)) < 1e-6
            and abs(dot(self.ey, self.ez)) < 1e-6
            and abs(dot(self.ez, self.ex)) < 1e-6
            and abs(norm(self.ex) - 1) < 1e-6
            and abs(norm(self.ey) - 1) < 1e-6
            and abs(norm(self.ez) - 1) < 1e-6
        )


class Node:
    def __init__(self, id: int, x: float, y: float, z: float):
        self.id = id
        self.x = x
        self.y = y
        self.z = z


class Element:
    def __init__(self, id: int, nodes: list[Node], orientation: Orientation):
        self.id = id
        self.nodes = nodes
        self.orientation = orientation


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
        self.Ke_local, self.R, self.origin = plate4_mindlin_stiffness(
            ABD, self.get_node_coords(), As
        )


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

    # Define 6 unique global nodes for two unit quads in XY-plane (z=0)
    # Node layout (indices in parentheses):
    # (0)---(1)---(2)
    #  |  e0 | e1  |
    # (3)---(4)---(5)
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [2.0, 0.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [1.0, 1.0, 0.0],  # 4
            [2.0, 1.0, 0.0],  # 5
        ],
        dtype=ABD.dtype,
    )

    # Element connectivities (counter-clockwise):
    # e0: (0,1,4,3); e1: (1,2,5,4)
    elements = [
        (0, 1, 4, 3),
        (1, 2, 5, 4),
    ]

    dof_per_node = 5
    ndof = nodes.shape[0] * dof_per_node
    K_global = np.zeros((ndof, ndof), dtype=ABD.dtype)

    def dof_indices(conn):
        idx = []
        for n in conn:
            base = n * dof_per_node
            idx.extend([base + 0, base + 1, base + 2, base + 3, base + 4])
        return np.array(idx, dtype=int)

    # Assemble
    first_R = None
    first_origin = None
    for conn in elements:
        nodes_xyz = nodes[list(conn)]
        Ke_local, R, origin = plate4_mindlin_stiffness(ABD, nodes_xyz, As=As)
        gdofs = dof_indices(conn)
        # Elements lie in XY plane -> local ~ global, assemble directly
        K_global[np.ix_(gdofs, gdofs)] += Ke_local
        if first_R is None:
            first_R, first_origin = R, origin

    # Print a brief summary
    np.set_printoptions(precision=3, suppress=True)
    print("Global stiffness K shape:", K_global.shape)
    print("First element centroid (global):", first_origin)
    print("First element local axes (columns of R):\n", first_R)
    print("K_global top-left 12x12 block:\n", K_global[:12, :12])
