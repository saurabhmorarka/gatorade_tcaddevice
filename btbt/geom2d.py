"""Triangle-level geometry shared by the 2D BTBT models (btbt/local2d.py,
btbt/paths2d.py): per-triangle P1 gradient operators, the semiconductor/
oxide split of the triangles, recovered nodal gradients, and fast point
location for the field-line tracer.

psi is piecewise linear (P1) on the triangles, so its gradient is constant
on each triangle: grad psi|_t = sum_k C[t,k] psi[T[t,k]]. Everything here
is computed on SEMICONDUCTOR triangles only - the oxide's field (a factor
eps_si/eps_ox = 3x larger than the silicon's next to it) must not leak
into a silicon interface node's tunneling field.
"""
import numpy as np
import scipy.sparse as sp
from matplotlib.tri import Triangulation


class TriGeom:
    def __init__(self, mesh, mat):
        P, T = mesh.points, mesh.triangles
        self.points, self.triangles = P, T
        N, M = len(P), len(T)
        cen = P[T].mean(axis=1)
        tri_ins, _ = mesh.domain.material_props_at(cen[:, 0], cen[:, 1], mat)
        self.tri_semi = ~np.asarray(tri_ins, dtype=bool)

        e1 = P[T[:, 1]] - P[T[:, 0]]
        e2 = P[T[:, 2]] - P[T[:, 0]]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        C1 = np.stack([e2[:, 1], -e2[:, 0]], axis=1) / det[:, None]
        C2 = np.stack([-e1[:, 1], e1[:, 0]], axis=1) / det[:, None]
        self.C = np.stack([-(C1 + C2), C1, C2], axis=1)          # (M,3,2)
        self.area = 0.5 * np.abs(det)

        # node <- semiconductor-triangle area weights (rows sum to 1 where a
        # node touches any semiconductor triangle)
        s = np.flatnonzero(self.tri_semi)
        rows = T[s].ravel()
        cols = np.repeat(s, 3)
        w = np.repeat(self.area[s], 3)
        node_area = np.bincount(rows, weights=w, minlength=N)
        self.node_area = node_area
        safe = np.where(node_area > 0, node_area, 1.0)
        self.W = sp.csr_matrix((w / safe[rows], (rows, cols)), shape=(N, M))

        # band structure per node (energies in eV, the solver's reference:
        # Ei = -(psi + dEi), Ec = Ei + ec_off, Ev = Ec - Eg). Single-material
        # meshes built before heterojunction support fall back to mat.
        self.dEi = mesh.dEi_arr if getattr(mesh, "dEi_arr", None) is not None else np.zeros(N)
        self.Eg = mesh.Eg_arr if getattr(mesh, "Eg_arr", None) is not None else np.full(N, mat.Eg_eV)
        self.ec_off = mesh.ec_off_arr if getattr(mesh, "ec_off_arr", None) is not None else \
            np.full(N, mat.Vt * np.log(mat.Nc / mat.ni))
        self.Eg_ref = mat.Eg_eV

        self.is_semi_node = node_area > 0
        if mesh.ni_arr is not None:
            self.is_semi_node &= mesh.ni_arr > 0
        # Si/SiO2 interface length per semiconductor node (cm): half of every
        # oxide-triangle edge whose two ends are semiconductor nodes (each
        # interface edge borders exactly one oxide triangle)
        if_len = np.zeros(N)
        for t in np.flatnonzero(~self.tri_semi):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                i, j = T[t, a], T[t, b]
                if self.is_semi_node[i] and self.is_semi_node[j]:
                    L = np.hypot(*(P[i] - P[j]))
                    if_len[i] += 0.5 * L
                    if_len[j] += 0.5 * L
        self.if_len = if_len

        bc = np.array(mesh.boundary_bc_type)
        is_contact = np.zeros(N, dtype=bool)
        is_contact[mesh.boundary_point_index[np.char.startswith(bc.astype(str), "contact:")]] = True
        self.is_contact = is_contact

        self._tri = Triangulation(P[:, 0], P[:, 1], T)
        self._finder = self._tri.get_trifinder()
        # barycentric map per triangle: lam[:2] = Minv @ (x - v2)
        V = P[T]
        A = np.stack([V[:, 0] - V[:, 2], V[:, 1] - V[:, 2]], axis=2)     # (M,2,2) columns v0-v2, v1-v2
        self._Minv = np.linalg.inv(A)
        self._v2 = V[:, 2]
        # semiconductor triangles around each node (padded with -1), for the
        # tracer's first step out of a start node
        s_tris = np.flatnonzero(self.tri_semi)
        v = T[s_tris].ravel()
        t = np.repeat(s_tris, 3)
        order = np.argsort(v, kind="stable")
        v, t = v[order], t[order]
        deg = np.bincount(v, minlength=N)
        start = np.concatenate([[0], np.cumsum(deg)[:-1]])
        col = np.arange(len(v)) - start[v]
        self.node_tris = np.full((N, max(1, deg.max())), -1)
        self.node_tris[v, col] = t

    def tri_grad(self, psi):
        """(M,2) constant gradient of the P1 interpolant on every triangle."""
        return np.einsum("mkd,mk->md", self.C, psi[self.triangles])

    def node_grad(self, psi, g_tri=None):
        """(N,2) recovered nodal gradient: semiconductor-triangle area-weighted
        average (0 at pure-oxide nodes)."""
        g = self.tri_grad(psi) if g_tri is None else g_tri
        return self.W @ g

    def barycentric(self, pts, tri):
        """(K,3) barycentric coordinates of pts (K,2) in triangles tri (K,)."""
        l01 = np.einsum("kij,kj->ki", self._Minv[tri], pts - self._v2[tri])
        return np.column_stack([l01, 1.0 - l01.sum(axis=1)])

    def locate(self, pts, guess):
        """Triangle index containing each point (-1 outside the mesh), trying
        the guess triangle first - a field-line step almost always stays in
        (or next to) the triangle it started in, so only the misses go
        through the (much slower) trapezoid-map search."""
        tri = np.asarray(guess).copy()
        ok = tri >= 0
        if np.any(ok):
            lam = self.barycentric(pts[ok], tri[ok])
            inside = np.all(lam >= -1e-10, axis=1)
            idx = np.flatnonzero(ok)
            ok[idx[~inside]] = False
        miss = ~ok
        if np.any(miss):
            tri[miss] = self._finder(pts[miss, 0], pts[miss, 1])
        return tri

    def interp(self, f, pts, tri):
        """P1 interpolation of nodal field(s) f ((N,) or (N,d)) at pts in tri."""
        lam = self.barycentric(pts, tri)
        vals = f[self.triangles[tri]]                             # (K,3) or (K,3,d)
        if vals.ndim == 2:
            return np.sum(lam * vals, axis=1)
        return np.einsum("kj,kjd->kd", lam, vals)
