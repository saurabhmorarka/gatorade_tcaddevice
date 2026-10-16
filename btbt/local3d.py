"""The local tunneling models (Kane band-to-band, Hurkx trap-assisted; see
btbt/local2d.py) on a 3D box-FV mesh (mesh3d/octree.py, fab/simulate.py).

btbt/local2d.py takes the node field from the P1 gradient of the 2D
triangles; the octree mesh has no elements, only nodes and edges. Here the
field at a node is the magnitude of a least-squares gradient over the
node's SEMICONDUCTOR edges (an oxide edge's field is eps_si/eps_ox larger
and must not leak into an interface node's tunneling field, as in 2D):

    g_i = M_i^-1 sum_e w_e t_e (psi_j - psi_i) / L_e,   M_i = sum_e w_e t_e t_e^T,

t_e the edge's unit vector, w_e = 1/L_e^2. Exact for a linear psi. On the
half-width symmetry plane the normal component is zero by symmetry (the
one-sided fit would otherwise invent one) and is dropped there. Everything
else - Kane, Hurkx, the Jacobian over U = (psi, phin, phip) - is
btbt/local2d.py's, with one change to Kane: the rate carries Hurkx's
band-to-band occupation factor (Hurkx, Klaassen, Knuvers, IEEE TED 39(2),
1992; Sentaurus Band2Band(Hurkx)),

    G_kane = A F^P exp(-B/F) * (ni^2 - n p) / ((n + ni)(p + ni)),

zero in equilibrium (np = ni^2), generation where the junction or gate
depletes both carriers, recombination where np > ni^2. Without it the
local Kane rate fires wherever the built-in field is high: an abrupt
7e20 / 1e16 S/D junction reaches 3e6 V/cm and generated 6e-7 A in a
near-equilibrium planar state (planar_tiny), a current that is not there.
"""
import numpy as np
import scipy.sparse as sp

from btbt.local2d import LocalTunneling2D


def node_gradient_operators(mesh, mask_edges=None, zero_axis_nodes=None):
    """(G_0, G_1, G_2): sparse (N, N) with (G_a @ psi)[i] the least-squares
    gradient component a at node i over the edges in mask_edges (default
    all). zero_axis_nodes: {axis: bool (N,)} - that component is zeroed at
    those nodes (a symmetry plane normal to axis)."""
    P = np.asarray(mesh.points, float)
    N, dim = P.shape
    E = np.asarray(mesh.edges)
    if mask_edges is not None:
        E = E[mask_edges]
    ii, jj = E[:, 0], E[:, 1]
    d = P[jj] - P[ii]
    L = np.linalg.norm(d, axis=1)
    t = d / L[:, None]
    w = 1.0 / L ** 2
    M = np.zeros((N, dim, dim))
    tt = w[:, None, None] * t[:, :, None] * t[:, None, :]
    for a in range(dim):
        for b in range(dim):
            M[:, a, b] = np.bincount(ii, tt[:, a, b], minlength=N) + np.bincount(jj, tt[:, a, b], minlength=N)
    # a node with too few edges to span 3D (or none) gets a pseudo-inverse:
    # its gradient is the fit within the span of its edges
    Minv = np.linalg.pinv(M, rcond=1e-8)
    ci = np.einsum("eab,eb->ea", Minv[ii], t) * (w / L)[:, None]      # node i's weight on (psi_j - psi_i)
    cj = np.einsum("eab,eb->ea", Minv[jj], t) * (w / L)[:, None]      # node j's (same t, same difference)
    G = []
    for a in range(dim):
        r = np.concatenate([ii, ii, jj, jj])
        c = np.concatenate([jj, ii, jj, ii])
        v = np.concatenate([ci[:, a], -ci[:, a], cj[:, a], -cj[:, a]])
        if zero_axis_nodes and a in zero_axis_nodes:
            v = np.where(zero_axis_nodes[a][r], 0.0, v)
        G.append(sp.csr_matrix((v, (r, c)), shape=(N, N)))
    return G


class Geom3D:
    """The attributes LocalTunneling2D reads, for a 3D box-FV mesh."""

    def __init__(self, mesh, mat, symmetry_plane=None):
        N = len(mesh.points)
        self.dEi = mesh.dEi_arr if getattr(mesh, "dEi_arr", None) is not None else np.zeros(N)
        self.Eg = mesh.Eg_arr if getattr(mesh, "Eg_arr", None) is not None else np.full(N, mat.Eg_eV)
        self.ec_off = mesh.ec_off_arr if getattr(mesh, "ec_off_arr", None) is not None else \
            np.full(N, mat.Vt * np.log(mat.Nc / mat.ni))
        self.Eg_ref = mat.Eg_eV
        semi = ~np.asarray(mesh.is_insulator, bool) if mesh.is_insulator is not None else np.ones(N, bool)
        if mesh.ni_arr is not None:
            semi &= mesh.ni_arr > 0
        self.is_semi_node = semi
        bc = np.array(mesh.boundary_bc_type).astype(str)
        is_contact = np.zeros(N, dtype=bool)
        is_contact[mesh.boundary_point_index[np.char.startswith(bc, "contact:")]] = True
        self.is_contact = is_contact
        E = np.asarray(mesh.edges)
        semi_edges = semi[E[:, 0]] & semi[E[:, 1]]
        zero = None
        if symmetry_plane is not None:                   # (axis, coordinate in the mesh's units)
            a, c = symmetry_plane
            P = np.asarray(mesh.points, float)
            tol = 1e-6 * (P[:, a].max() - P[:, a].min())
            zero = {a: np.abs(P[:, a] - c) <= tol}
        self.G = node_gradient_operators(mesh, semi_edges, zero)


class LocalTunneling3D(LocalTunneling2D):
    # Both rates have the SRH form G = (ni^2 - n p) x (a positive factor):
    # they only pull n p toward ni^2, so the quasi-Fermi levels keep the
    # contacts' range (the maximum principle behind newton_solve_2d's
    # quasi-Fermi box, which therefore stays on).
    keeps_qf_box = True

    def __init__(self, mesh, mat, ni_arr, kane=None, hurkx=None, symmetry_plane=None):
        self.geom, self.mat, self.ni = Geom3D(mesh, mat, symmetry_plane), mat, ni_arr
        self.kane, self.hurkx = kane, hurkx
        self.T = mat.Vt * 1.602176634e-19 / 1.380649e-23
        self.free = self.geom.is_semi_node & ~self.geom.is_contact

    def node_field(self, psi, jacobian=False):
        g = [Ga @ psi for Ga in self.geom.G]
        mag = np.sqrt(sum(x * x for x in g))
        if not jacobian:
            return mag, None
        inv = 1.0 / np.maximum(mag, 1e-30)
        dF = sum(sp.diags(x * inv) @ Ga for x, Ga in zip(g, self.geom.G))      # d|g_i|/dpsi_k
        return mag, dF.tocsr()

    def _occupation(self, psi, phin, phip):
        """D = (ni^2 - n p) / ((n + ni)(p + ni)) and dD/dn, dD/dp."""
        n, p = self._np(psi, phin, phip)
        ni = self.ni
        ok = ni > 0
        den = np.where(ok, (n + ni) * (p + ni), 1.0)
        num = ni ** 2 - n * p
        D = np.where(ok, num / den, 0.0)
        dD_dn = np.where(ok, (-p * den - num * (p + ni)) / den ** 2, 0.0)
        dD_dp = np.where(ok, (-n * den - num * (n + ni)) / den ** 2, 0.0)
        return D, dD_dn, dD_dp, n, p

    def parts(self, psi, phin, phip):
        Gk, Gt, F = super().parts(psi, phin, phip)
        D = self._occupation(psi, phin, phip)[0]
        return np.where(self.free, Gk * D, 0.0), Gt, F

    def __call__(self, psi, phin, phip, jacobian=False):
        from btbt.tat_kernel import hurkx_enhancement
        from tat.tat import btbt_generation
        N = len(psi)
        Vt = self.mat.Vt
        F, dF = self.node_field(psi, jacobian)
        D, dD_dn, dD_dp, n, p = self._occupation(psi, phin, phip)
        G = np.zeros(N)
        dG_dF = np.zeros(N)
        dG_dn = np.zeros(N)
        dG_dp = np.zeros(N)
        if self.kane is not None:
            Gk, dGk = btbt_generation(F, self._kane())
            G += Gk * D
            dG_dF += dGk * D
            dG_dn += Gk * dD_dn
            dG_dp += Gk * dD_dp
        if self.hurkx is not None:
            Gn_, dGn_, Gp_, dGp_ = self._gammas(F)
            dGt, d_dn, d_dp, d_dGn, d_dGp = hurkx_enhancement(n, p, self.ni, self.mat.tau_n, self.mat.tau_p,
                                                              Gn_, Gp_)
            G += dGt
            dG_dF += d_dGn * dGn_ + d_dGp * dGp_
            dG_dn += d_dn
            dG_dp += d_dp
        free = self.free
        G = np.where(free, G, 0.0)
        if not jacobian:
            return G, G, None, None
        dF = dF.tocoo()
        node = np.arange(N)
        rows = [dF.row, node, node, node]
        cols = [dF.col, node, N + node, 2 * N + node]
        data = [np.where(free, dG_dF, 0.0)[dF.row] * dF.data,
                np.where(free, dG_dn * n / Vt - dG_dp * p / Vt, 0.0),
                np.where(free, -dG_dn * n / Vt, 0.0), np.where(free, dG_dp * p / Vt, 0.0)]
        dU = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(N, 3 * N))
        return G, G, dU, dU
