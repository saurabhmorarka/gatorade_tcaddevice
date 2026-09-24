"""2D nonlocal tunneling: tunneling paths traced along ELECTRIC FIELD LINES
through the triangle mesh (btbt/kernel.py has the band-to-band physics,
btbt/tat_kernel.py the trap-assisted physics).

Field-line tracing. From a start node r0 the path follows the field line
uphill (sign=+1, direction +grad psi = -E) or downhill (sign=-1) in psi,
dr/ds = sign * grad psi / |grad psi|, until sign*(psi(r) - psi(r0)) reaches a
target energy/q:
  * Kane band-to-band: target Eg, uphill. The valence band at r0 lines up
    with the conduction band at r (kernel.py eq. 1).
  * Hurkx trap-assisted, electron: target Eg/2, uphill. A midgap trap at r0
    lines up with the conduction band at r.
  * Hurkx trap-assisted, hole: target Eg/2, downhill. The trap lines up with
    the valence band at r.
It is integrated with midpoint (RK2) steps. The direction comes from the
recovered nodal gradient, interpolated linearly inside each triangle, so the
field lines are smooth and bend with the real 2D field (around the gate-edge
corner, or a junction corner). The energy uses the exact P1 potential. Steps
are adaptive, ds = DS_FRAC * Eg/|grad psi|, clipped to [DS_MIN, DS_MAX]. The
crossing inside the last step is found by linear interpolation.

A path stops short of its target if it leaves the semiconductor (steps into
an oxide triangle or out of the mesh - a field line that ends on the gate or
the free surface before the bands have bent enough), stops climbing (a
potential extremum), or exceeds its length cap.

How each mechanism uses the paths:
  * Kane: only COMPLETE paths tunnel. G = A F_eff^P exp(-B/F_eff) with
    F_eff = Eg/l. Holes are generated at r0; the same pair rate goes to the
    electron equation at the end point (barycentric split over the end
    triangle's non-contact vertices), times the occupation factor D
    (kernel.py eq. 4). No path, no tunneling - this is what captures the
    GIDL onset (surface band bending must exceed Eg) that a local model
    misses.
  * Hurkx: Gamma_n and Gamma_p use the field averaged along the electron
    (uphill) and hole (downhill) half-gap paths, F_eff = dpsi/l (below
    F_START_TAT, the local node field). If a path
    stops short, the average over the part that was traced is used. Hurkx's
    Gamma already integrates over tunneling depths below the full Eg/2, so
    a partial path still gives some enhancement, and in a uniform field
    F_eff = F exactly (the local model). Both carriers stay at the trap node.

Coupling. The paths, F_eff and Gamma depend nonlocally on psi, which would
give a dense Jacobian. They are frozen during each Newton solve and
recomputed between solves (lagged outer iteration, validated in 1D by
btbt/main_btbt_1d.py). The trap-assisted rate's dependence on the LOCAL n
and p stays implicit in the Newton Jacobian.
"""
import numpy as np
import scipy.sparse as sp

from tat.tat import hurkx_gamma
from btbt.kernel import path_rate, occupation_factor
from btbt.tat_kernel import hurkx_enhancement

F_START_KANE = 5.0e4     # V/cm - Kane start threshold
F_START_TAT = 5.0e4      # V/cm - below this Gamma uses the local field (Gamma <~ 1 there and
                         # the field barely varies over a tunneling length, so the two agree)
F_EFF_MIN = 2.5e5        # V/cm -> Kane path cap Eg/F_EFF_MIN ~ 45 nm (exp(-B/F) < 1e-37 beyond)
L_MAX_TAT = 1.0e-5       # cm (100 nm) - half-gap path cap; a longer path uses its partial average
DS_FRAC = 0.05
DS_MIN = 0.05e-7         # cm (0.05 nm)
DS_MAX = 2.0e-7          # cm (2 nm, ~ the mesh spacing where paths run)


class FieldLineTracer:
    def __init__(self, geom, Eg_eV):
        self.geom = geom
        self.Eg = Eg_eV

    def _first_triangle(self, starts, d):
        """The start node's semiconductor triangle whose centroid lies
        furthest along the start direction d."""
        g = self.geom
        cand = g.node_tris[starts]                                   # (K,D)
        c = g.points[g.triangles[np.maximum(cand, 0)]].mean(axis=2) - g.points[starts][:, None, :]
        score = np.where(cand >= 0, np.einsum("kdj,kj->kd", c, d), -np.inf)
        best = cand[np.arange(len(starts)), np.argmax(score, axis=1)]
        return np.where(np.isfinite(score.max(axis=1)), best, -1)

    def trace(self, psi, g_node, starts, target, sign, L_max, record=None):
        """Trace from each node in `starts`. Returns dict: l (K,) length at
        which sign*(psi - psi0) reached `target` (inf if never), end (K,2)
        and end_tri (K,) there, dpsi/s (K,) the potential gain and length
        reached when the path stopped, and 'poly' (list of polylines) for
        the indices in `record`."""
        g = self.geom
        K = len(starts)
        gm = np.hypot(g_node[starts, 0], g_node[starts, 1])
        pos = g.points[starts].copy()
        tri = self._first_triangle(starts, sign * g_node[starts] / np.maximum(gm, 1e-30)[:, None])
        psi0 = psi[starts]
        dpsi = np.zeros(K)
        s = np.zeros(K)
        l = np.full(K, np.inf)
        end = np.full((K, 2), np.nan)
        end_tri = np.full(K, -1)
        alive = tri >= 0
        rec = None if record is None else {int(k): [pos[k].copy()] for k in record}

        def direction(p, t):
            v = sign * g.interp(g_node, p, t)
            m = np.hypot(v[:, 0], v[:, 1])
            return v / np.maximum(m, 1e-30)[:, None], m

        while np.any(alive):
            a = np.flatnonzero(alive)
            p, t = pos[a], tri[a]
            d1, m1 = direction(p, t)
            ds = np.clip(DS_FRAC * self.Eg / np.maximum(m1, 1e-30), DS_MIN, DS_MAX)
            mid = p + 0.5 * ds[:, None] * d1
            tm = g.locate(mid, t)
            bad = tm < 0
            tm = np.where(bad, t, tm)
            bad |= ~g.tri_semi[tm]
            d2, _ = direction(mid, tm)
            new = p + ds[:, None] * d2
            tn = g.locate(new, tm)
            bad |= tn < 0
            tn = np.where(tn < 0, tm, tn)
            bad |= ~g.tri_semi[tn]
            dnew = sign * (g.interp(psi, new, tn) - psi0[a])
            dold = dpsi[a]
            done = ~bad & (dnew >= target)
            stall = ~bad & (dnew < dold - 1e-12)
            frac = np.where(done, (target - dold) / np.maximum(dnew - dold, 1e-30), 0.0)
            ok = ~bad & ~stall
            i_done = a[done]
            l[i_done] = s[i_done] + frac[done] * ds[done]
            end[i_done] = p[done] + frac[done, None] * (new[done] - p[done])
            end_tri[i_done] = tn[done]
            # advance the paths whose step was valid
            step_len = np.where(done, frac * ds, ds)
            upd = a[ok]
            s[upd] += step_len[ok]
            dpsi[upd] = np.where(done[ok], target, dnew[ok])
            pos[upd] = new[ok]
            tri[upd] = tn[ok]
            if rec is not None:
                for j, k in enumerate(a):
                    if int(k) in rec and ok[j]:
                        rec[int(k)].append(end[k].copy() if done[j] else new[j].copy())
            stop = done | bad | stall | (s[a] > L_max)
            alive[a[stop]] = False

        out = dict(l=l, end=end, end_tri=end_tri, dpsi=dpsi, s=s)
        if rec is not None:
            out["poly"] = [np.array(rec[int(k)]) for k in record]
        return out


class NonlocalTunneling2D:
    """Nonlocal Kane band-to-band (complete Eg paths, carriers split between
    path ends) plus nonlocal-field Hurkx trap-assisted tunneling. Call
    prepare(psi, phin, phip) to (re)compute the frozen path quantities, then
    use the object as the Newton solver's generation hook."""

    def __init__(self, geom, mat, ni_arr, cv_semi, kane=None, hurkx=None, surface_depth_cm=2.0e-6,
                 x_mid_cm=None):
        self.geom, self.mat, self.ni, self.cv = geom, mat, ni_arr, cv_semi
        self.kane, self.hurkx = kane, hurkx
        self.Eg, self.Vt = mat.Eg_eV, mat.Vt
        self.T = mat.Vt * 1.602176634e-19 / 1.380649e-23
        self.free = geom.is_semi_node & ~geom.is_contact
        self.tracer = FieldLineTracer(geom, mat.Eg_eV)
        self.surface_depth = surface_depth_cm
        P = geom.points
        self.x_mid = 0.5 * (P[:, 0].min() + P[:, 0].max()) if x_mid_cm is None else x_mid_cm
        N = len(P)
        self.Gn_k = np.zeros(N)
        self.Gp_k = np.zeros(N)
        self.Gam_n = np.zeros(N)
        self.Gam_p = np.zeros(N)
        self.kane_info = None

    # --- frozen (lagged) part ---
    def prepare(self, psi, phin, phip, record=None):
        g = self.geom
        g_node = g.node_grad(psi)
        gm = np.hypot(g_node[:, 0], g_node[:, 1])
        N = len(psi)
        polys = None
        if self.kane is not None:
            st = np.flatnonzero(self.free & (gm > F_START_KANE))
            rec = None if record is None else np.flatnonzero(np.isin(st, record))
            tr = self.tracer.trace(psi, g_node, st, self.Eg, +1, self.Eg / F_EFF_MIN, record=rec)
            polys = tr.get("poly")
            self.Gn_k, self.Gp_k, self.kane_info = self._kane_deposit(st, tr, phin, phip)
            if polys is not None:
                self.kane_info["poly"] = polys
                self.kane_info["poly_start"] = st[rec]
        if self.hurkx is not None:
            st = np.flatnonzero(self.free & (gm > F_START_TAT))
            F_loc = g.W @ np.hypot(*g.tri_grad(psi).T)               # same node field as local2d
            Gam_loc = np.where(self.free, hurkx_gamma(F_loc, self.T, self.hurkx)[0], 0.0)
            Gn = Gam_loc.copy()
            Gp = Gam_loc.copy()
            for sign, G_out in ((+1, Gn), (-1, Gp)):
                tr = self.tracer.trace(psi, g_node, st, 0.5 * self.Eg, sign, L_MAX_TAT)
                F_eff = np.where(tr["s"] > 0, tr["dpsi"] / np.maximum(tr["s"], 1e-30), 0.0)
                G_out[st] = hurkx_gamma(F_eff, self.T, self.hurkx)[0]
            self.Gam_n, self.Gam_p = Gn, Gp

    def _kane_deposit(self, st, tr, phin, phip):
        g = self.geom
        N = len(phin)
        G, F_eff = path_rate(tr["l"], self.Eg, self.kane)
        ok = G > 0
        st, l, G, F_eff = st[ok], tr["l"][ok], G[ok], F_eff[ok]
        end, et = tr["end"][ok], tr["end_tri"][ok]
        phin_end = g.interp(phin, end, et) if len(st) else np.zeros(0)
        GD = G * occupation_factor(phin_end, phip[st], self.Vt)
        pair = GD * self.cv[st]
        Gp = np.zeros(N)
        np.add.at(Gp, st, GD)
        verts = g.triangles[et]
        lam = np.clip(g.barycentric(end, et), 0.0, None) if len(st) else np.zeros((0, 3))
        lam = np.where(self.free[verts], lam, 0.0)
        wsum = lam.sum(axis=1)
        has = wsum > 0
        elec = np.zeros(N)
        np.add.at(elec, verts[has].ravel(), (pair[has, None] * lam[has] / wsum[has, None]).ravel())
        np.add.at(elec, st[~has], pair[~has])
        Gn = np.where(self.cv > 0, elec / np.where(self.cv > 0, self.cv, 1.0), 0.0)
        y_min = np.minimum(g.points[st, 1], end[:, 1]) if len(st) else np.zeros(0)
        info = dict(start=st, end=end, l=l, F_eff=F_eff, GD=GD, pair=pair, surface=y_min < self.surface_depth,
                    drain_side=g.points[st, 0] > self.x_mid)
        return Gn, Gp, info

    # --- Newton hook ---
    def _np(self, psi, phin, phip):
        Vt = self.Vt
        return self.ni * np.exp((psi - phin) / Vt), self.ni * np.exp((phip - psi) / Vt)

    def tat_rate(self, psi, phin, phip):
        n, p = self._np(psi, phin, phip)
        return np.where(self.free, hurkx_enhancement(n, p, self.ni, self.mat.tau_n, self.mat.tau_p,
                                                     self.Gam_n, self.Gam_p)[0], 0.0)

    def __call__(self, psi, phin, phip, jacobian=False):
        N = len(psi)
        Gn, Gp = self.Gn_k.copy(), self.Gp_k.copy()
        if self.hurkx is None:
            return Gn, Gp, None, None
        n, p = self._np(psi, phin, phip)
        dG, d_dn, d_dp, _, _ = hurkx_enhancement(n, p, self.ni, self.mat.tau_n, self.mat.tau_p,
                                                 self.Gam_n, self.Gam_p)
        f = self.free
        dG = np.where(f, dG, 0.0)
        Gn += dG
        Gp += dG
        if not jacobian:
            return Gn, Gp, None, None
        Vt = self.Vt
        dpsi = sp.diags(np.where(f, d_dn * n / Vt - d_dp * p / Vt, 0.0))
        dphin = sp.diags(np.where(f, -d_dn * n / Vt, 0.0))
        dphip = sp.diags(np.where(f, d_dp * p / Vt, 0.0))
        dU = sp.hstack([dpsi, dphin, dphip]).tocsr()
        return Gn, Gp, dU, dU


def integrate(mask_rate, cv, mask):
    """q * sum(rate * cv) over mask, A/um of width."""
    from core.params import Q
    return float(Q * np.sum((mask_rate * cv)[mask]) * 1e-4)
