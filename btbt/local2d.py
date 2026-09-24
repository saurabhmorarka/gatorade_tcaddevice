"""The 1D LOCAL tunneling models (tat/tat.py: Kane band-to-band via
btbt_generation, Hurkx trap-assisted via hurkx_gamma, both used unchanged)
on the 2D mesh, fully Newton-coupled - "the 1D diode model, as is, in 2D".

Local field at a node = semiconductor-triangle area-weighted average of the
P1 field magnitude |grad psi| on the triangles around it - the 2D analog of
tat/newton_solver_tat.py::_node_field's length-weighted average of the two
edge fields. Kane generates electrons and holes at the node; Hurkx
enhances the node's SRH rate with Gamma(F_node) for both carriers
(btbt/tat_kernel.py; the solver's own SRH term stays, only the enhancement
is added).

What this model does NOT know (see btbt/kernel.py): whether the bands bend
by Eg (Kane) or Eg/2 (trap to band) within reach, how the field varies
along the tunneling distance, and - for Kane - the occupation of the states
(it generates carriers in equilibrium wherever the field is high).
btbt/main_btbt2d_sweep.py runs it next to the nonlocal model to show what
that costs in 2D.
"""
import numpy as np
import scipy.sparse as sp

from tat.tat import btbt_generation, hurkx_gamma
from btbt.tat_kernel import hurkx_enhancement


class LocalTunneling2D:
    def __init__(self, geom, mat, ni_arr, kane=None, hurkx=None):
        """kane: tat.tat.KaneBTBTModel or None (off); hurkx:
        tat.tat.HurkxTATModel or None (off)."""
        self.geom, self.mat, self.ni = geom, mat, ni_arr
        self.kane, self.hurkx = kane, hurkx
        self.T = mat.Vt * 1.602176634e-19 / 1.380649e-23
        self.free = geom.is_semi_node & ~geom.is_contact
        s = np.flatnonzero(geom.tri_semi)
        T = geom.triangles[s]
        self._s = s
        self._rows = np.repeat(T, 3, axis=1).ravel()
        self._cols = np.tile(T, (1, 3)).ravel()
        self._wa = geom.area[s][:, None] / np.where(geom.node_area > 0, geom.node_area, 1.0)[T]

    def node_field(self, psi, jacobian=False):
        g = self.geom
        gt = g.tri_grad(psi)
        mag = np.hypot(gt[:, 0], gt[:, 1])
        F = g.W @ mag
        if not jacobian:
            return F, None
        s = self._s
        u = gt[s] / np.maximum(mag[s], 1e-30)[:, None]
        dmag = np.einsum("sd,skd->sk", u, g.C[s])                 # d|g_t|/dpsi_k
        vals = (self._wa[:, :, None] * dmag[:, None, :]).ravel()
        N = len(psi)
        return F, sp.csr_matrix((vals, (self._rows, self._cols)), shape=(N, N))

    def parts(self, psi, phin, phip):
        """Per-node (G_kane, dG_tat, F) - for diagnostics."""
        n, p = self._np(psi, phin, phip)
        F, _ = self.node_field(psi)
        Gk = np.zeros_like(F) if self.kane is None else btbt_generation(F, self.kane)[0]
        Gt = np.zeros_like(F)
        if self.hurkx is not None:
            Gam, _ = hurkx_gamma(F, self.T, self.hurkx)
            Gt = hurkx_enhancement(n, p, self.ni, self.mat.tau_n, self.mat.tau_p, Gam, Gam)[0]
        return np.where(self.free, Gk, 0.0), np.where(self.free, Gt, 0.0), F

    def _np(self, psi, phin, phip):
        Vt = self.mat.Vt
        return self.ni * np.exp((psi - phin) / Vt), self.ni * np.exp((phip - psi) / Vt)

    def __call__(self, psi, phin, phip, jacobian=False):
        N = len(psi)
        Vt = self.mat.Vt
        F, dF = self.node_field(psi, jacobian)
        G = np.zeros(N)
        dG_dF = np.zeros(N)
        dG_dn = dG_dp = None
        if self.kane is not None:
            Gk, dGk = btbt_generation(F, self.kane)
            G += Gk
            dG_dF += dGk
        if self.hurkx is not None:
            n, p = self._np(psi, phin, phip)
            Gam, dGam = hurkx_gamma(F, self.T, self.hurkx)
            dGt, d_dn, d_dp, d_dGn, d_dGp = hurkx_enhancement(n, p, self.ni, self.mat.tau_n, self.mat.tau_p, Gam, Gam)
            G += dGt
            dG_dF += (d_dGn + d_dGp) * dGam
            dG_dn, dG_dp = d_dn, d_dp
        free = self.free
        G = np.where(free, G, 0.0)
        if not jacobian:
            return G, G, None, None
        dpsi = sp.diags(np.where(free, dG_dF, 0.0)) @ dF
        if dG_dn is not None:
            n, p = self._np(psi, phin, phip)
            dpsi = dpsi + sp.diags(np.where(free, dG_dn * n / Vt - dG_dp * p / Vt, 0.0))
            dphin = sp.diags(np.where(free, -dG_dn * n / Vt, 0.0))
            dphip = sp.diags(np.where(free, dG_dp * p / Vt, 0.0))
        else:
            dphin = dphip = sp.csr_matrix((N, N))
        dU = sp.hstack([dpsi, dphin, dphip]).tocsr()
        return G, G, dU, dU
