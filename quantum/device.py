"""Schrodinger-Poisson channel charge on a 2D/3D box-FV device mesh
(plans/quantum_plan.md, phases P2/P3).

The quantum box: lines start at every semiconductor node that sits on a
GATE dielectric (an insulator neighbour, with a gate-contact node within
`max_tox` straight across that dielectric). Each line runs along the
interface normal into the body:
  - to the next insulator wall, if it is within `depth` (FinFET fin,
    nanosheet: both walls confine -> volume inversion), else
  - to `depth` (planar bulk), with the classical taper of lines.py.
Nothing here is planar-specific: the start points and normals come from the
mesh. `spacing_nm` thins the lines to one per lattice cell, centred on the
gated area (None: every interface column; larger than the gate: one line at
the centre); nodes not on a line take Lambda from the nearest line at the
same depth. `margin_nm` extends the box past the gate along the same wall
(under the spacers), with Lambda tapered to 0 over `margin_taper_nm`.
`tol` (V, outer loop): 3 mV changes the planar Id by < 0.3% vs 1 mV.

The mesh must have straight node columns along the normal under the gate
(true for the rectilinear and octree fab meshes: interfaces sit on grid
planes). psi on a line is interpolated from its column's nodes, linearly
extrapolated into the oxide (uniform field there).

Crystal orientation: device axes (x = transport, y = depth, z = width) in
crystal coordinates; the default is the standard (001) wafer with a <110>
channel, so a planar top surface is (001) and a FinFET sidewall (110).
"""
import numpy as np
from scipy.spatial import cKDTree

from quantum.lines import Line, quantum_potential, taper
from quantum.coupling import sp_loop

CRYSTAL_AXES = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 1.0, 0.0]]) / np.array([[2 ** 0.5], [1.0], [2 ** 0.5]])


class DeviceQuantum:
    def __init__(self, mesh, mat, gate="gate", carrier=None, spacing_nm=None, depth=20e-7, t_pen=1e-7,
                 h=1e-8, n_states=10, statistics="boltzmann", taper_frac=0.25, max_tox=5e-7,
                 crystal_axes=CRYSTAL_AXES, col_tol=1e-10, tol=3e-3, margin_nm=None, margin_taper_nm=3.0,
                 log=None):
        self.mesh, self.mat = mesh, mat
        self.n_states, self.statistics, self.tol = n_states, statistics, tol
        P = np.asarray(mesh.points, float)
        N, d = P.shape
        self.N = N
        ins = np.asarray(mesh.is_insulator, bool)
        bt = np.array(mesh.boundary_bc_type)
        G = mesh.boundary_point_index[bt == f"contact:{gate}"]
        if not len(G):
            raise ValueError(f"no contact named {gate!r} on the mesh")
        gtree = cKDTree(P[G])

        # interface nodes on the gate dielectric, with their inward normal
        ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
        a = np.concatenate([ii, jj])
        b = np.concatenate([jj, ii])
        sel = ~ins[a] & ins[b]
        a, b = a[sel], b[sel]
        u = P[a] - P[b]
        u /= np.linalg.norm(u, axis=1)[:, None]
        dist, k = gtree.query(P[a])
        v = P[G[k]] - P[a]
        cosang = -(v * u).sum(1) / np.maximum(np.linalg.norm(v, axis=1), 1e-30)
        ok = (dist <= max_tox) & (cosang > 0.9)
        starts = {}
        for node, nrm, c, dg in zip(a[ok], u[ok], cosang[ok], dist[ok]):
            if node not in starts or c > starts[node][1]:
                starts[node] = (nrm, c, dg)
        if not starts:
            raise ValueError("no gate-dielectric interface found")
        st = np.array(sorted(starts))
        nrms = np.array([starts[i][0] for i in st])
        tox = np.array([starts[i][2] for i in st])

        if carrier is None:                                  # the inversion carrier of the body
            carrier = "n" if np.median(mesh.Cdop[st]) < 0 else "p"
        self.carrier = carrier

        # one group per normal direction (planar: one; a fin: three)
        keys = [tuple(np.round(n_, 6)) for n_ in nrms]
        self.lines, self.normals = [], []
        prof_rows, prof_cols, prof_vals = [], [], []
        f0 = 0
        extra_nodes, extra_line, extra_s = [], [], []
        seen_walls = set()
        for key in sorted(set(keys)):
            n_hat = np.array(key)
            grp = st[[kk == key for kk in keys]]
            gtox = tox[[kk == key for kk in keys]]
            # transverse coordinates (d-1 of them)
            basis = np.linalg.svd(np.eye(d) - np.outer(n_hat, n_hat))[0][:, :d - 1]
            T = P @ basis
            S = P @ n_hat
            wlat = np.ones(len(grp))
            if margin_nm:
                # the same wall continues past the gate (under the spacers): lines there too,
                # Lambda tapered to 0 over the outer margin_taper_nm (no step at the gate edge)
                same = (u @ n_hat > 0.99) & ~np.isin(a, grp)
                cand = np.unique(a[same])
                dl = cKDTree(T[grp]).query(T[cand])[0] if len(cand) else np.zeros(0)
                inm = dl <= margin_nm * 1e-7
                cand, dl = cand[inm], dl[inm]
                t0_ = max(margin_nm - margin_taper_nm, 0.0) * 1e-7
                wl = 0.5 * (1.0 + np.cos(np.pi * np.clip((dl - t0_) / max(margin_nm * 1e-7 - t0_, 1e-30), 0, 1)))
                grp = np.concatenate([grp, cand])
                gtox = np.concatenate([gtox, np.full(len(cand), t_pen)])
                wlat = np.concatenate([wlat, wl])
            T_all = T[grp]                         # the box footprint, before any thinning
            if spacing_nm:
                # lattice cells centred on the gated area: a spacing larger than the
                # gate leaves the single line at its centre (option (a) of the plan)
                c0 = T[grp[wlat >= 1.0]].mean(axis=0)
                a_ = spacing_nm * 1e-7
                cell = np.floor((T[grp] - c0) / a_ + 0.5).astype(np.int64)
                center = c0 + cell * a_
                dc = np.linalg.norm(T[grp] - center, axis=1)
                pick = {}
                for gi, c_, dd in zip(range(len(grp)), map(tuple, cell), dc):
                    if c_ not in pick or dd < pick[c_][1]:
                        pick[c_] = (gi, dd)
                keep = np.array(sorted(g_ for g_, _ in pick.values()))
                grp, gtox, wlat = grp[keep], gtox[keep], wlat[keep]
            ttree = cKDTree(T)
            cols = ttree.query_ball_point(T[grp], r=col_tol)
            line_ids = []
            for p0, col, tx, wl_ in zip(grp, cols, gtox, wlat):
                col = np.asarray(col)
                s = S[col] - S[p0]
                semi_c = ~ins[col]
                after = s[~semi_c & (s > 0)]
                s_ins = after.min() if len(after) else np.inf
                semi_s = s[semi_c & (s >= -1e-12) & (s < s_ins)]
                s_wall = semi_s.max() if np.isfinite(s_ins) else np.inf
                wall_end = s_wall <= depth
                tp = min(t_pen, tx)
                s_end = s_wall + t_pen if wall_end else depth
                body_end = s_wall if wall_end else depth
                m_semi = semi_c & (s >= -1e-12) & (s <= body_end + 1e-12)
                nodes = col[m_semi]
                sn = s[m_semi]
                o = np.argsort(sn)
                nodes, sn = nodes[o], sn[o]
                if len(nodes) < 2:
                    continue
                if wall_end:                      # the same body column seen from the opposite wall
                    ck = (min(nodes[0], nodes[-1]), max(nodes[0], nodes[-1]))
                    if ck in seen_walls:
                        continue
                    seen_walls.add(ck)
                # psi profile: the column's oxide node(s) next to the start (and the far wall) + body nodes
                m_ox = ~semi_c & (s < 0) & (s >= -tx - 1e-12)
                m_far = (~semi_c & (s > 0) & (s <= s_wall + tx + 1e-12)) if wall_end else np.zeros(len(col), bool)
                pn = np.concatenate([col[m_ox], nodes, col[m_far]])
                ps = np.concatenate([s[m_ox], sn, s[m_far]])
                o = np.argsort(ps)
                pn, ps = pn[o], ps[o]
                n_f = int(round((s_end + tp) / h))
                sf = -tp + h * np.arange(n_f + 1)
                is_semi = (sf >= -1e-3 * h) & (sf <= body_end + 1e-3 * h)
                mid = 0.5 * (sn[1:] + sn[:-1])
                lo = np.concatenate([[sf[0]], mid])
                hi = np.concatenate([mid, [sf[-1] if wall_end else min(sf[-1], sn[-1] + 0.5 * (sn[-1] - sn[-2]))]])
                V = np.clip(hi, 0.0, body_end) - np.clip(lo, 0.0, body_end)
                w = taper(sn, 0.0, s_end, wall_end, taper_frac) * wl_
                # linear-interpolation weights of the fine points on the profile (extrapolated at the ends)
                kk_ = np.clip(np.searchsorted(ps, sf) - 1, 0, len(ps) - 2)
                t_ = (sf - ps[kk_]) / (ps[kk_ + 1] - ps[kk_])
                rows = f0 + np.arange(len(sf))
                prof_rows += [rows, rows]
                prof_cols += [pn[kk_], pn[kk_ + 1]]
                prof_vals += [1.0 - t_, t_]
                L = Line(s=sf, is_semi=is_semi, h=h, dd_idx=nodes, s_node=sn, lo=lo, hi=hi, V=V, w=w)
                L.f0, L.normal = f0, crystal_axes.T[:, :d] @ n_hat           # crystal coordinates
                L.s_end, L.wall_end = s_end, wall_end
                f0 += len(sf)
                line_ids.append(len(self.lines))
                self.lines.append(L)
            # the box's other semiconductor nodes: Lambda from the nearest line at the same depth
            if line_ids:
                # inside the footprint: within one interface-column spacing of a box column
                fp = cKDTree(T_all)
                spac = np.median(fp.query(T_all, k=2)[0][:, 1]) if len(T_all) > 1 else 1e-7
                R = 0.75 * spac * np.sqrt(2)
                on_line = np.zeros(N, bool)
                for i in line_ids:
                    on_line[self.lines[i].dd_idx] = True
                dfp, _ = fp.query(T, distance_upper_bound=R)
                cand = np.flatnonzero(np.isfinite(dfp) & ~ins & ~on_line)
                starts_T = np.array([T[self.lines[i].dd_idx[0]] for i in line_ids])
                _, kl = cKDTree(starts_T).query(T[cand]) if len(cand) else (None, np.zeros(0, int))
                kl_full = np.zeros(N, np.int64)
                kl_full[cand] = kl
                kl = kl_full
                for node in cand:
                    Li = self.lines[line_ids[kl[node]]]
                    s = S[node] - S[Li.dd_idx[0]]
                    if 0.0 <= s <= Li.s_node[-1]:
                        extra_nodes.append(node)
                        extra_line.append(line_ids[kl[node]])
                        extra_s.append(s)
        import scipy.sparse as sp
        self.W = sp.csr_matrix((np.concatenate(prof_vals), (np.concatenate(prof_rows), np.concatenate(prof_cols))),
                               shape=(f0, N))
        self.extra_nodes = np.array(extra_nodes, dtype=np.int64)
        self.extra_line = np.array(extra_line, dtype=np.int64)
        self.extra_s = np.array(extra_s)
        if log:
            nw = sum(1 for L in self.lines if L.wall_end)
            log(f"  quantum box: {len(self.lines)} lines ({nw} wall to wall), carrier {carrier}, "
                f"{sum(len(L.dd_idx) for L in self.lines)} line nodes + {len(self.extra_nodes)} interpolated")

    def update(self, state):
        """(Lambda_n, Lambda_p, q): the quantum potentials on every node for
        the state's psi, and the quantum charge per line node (cm^-3 x span)."""
        psi = state["psi"]
        phi = state["phin"] if self.carrier == "n" else state["phip"]
        pf = self.W @ psi
        lam = np.zeros(self.N)
        q = np.zeros(self.N)
        for L in self.lines:
            l_, info = quantum_potential(L, pf[L.f0:L.f0 + len(L.s)], psi[L.dd_idx], phi[L.dd_idx], self.carrier,
                                         self.mat, L.normal, self.n_states, self.statistics)
            lam[L.dd_idx] = l_
            q[L.dd_idx] = info["q"]
        for i in np.unique(self.extra_line):
            m = self.extra_line == i
            L = self.lines[i]
            lam[self.extra_nodes[m]] = np.interp(self.extra_s[m], L.s_node, lam[L.dd_idx])
        zero = np.zeros(self.N)
        return ((lam, zero) if self.carrier == "n" else (zero, lam)), q

    def solve(self, solve_dd, state0, lam0=None, max_outer=20):
        """Schrodinger-Poisson at one bias. solve_dd(quantum, guess) -> a
        newton_solve_2d result (quantum = (Lambda_n, Lambda_p)); state0 the
        start guess (psi, phin, phip). Returns the converged result with
        "lam" (the carrier's Lambda), "outer" and the summed "iters"."""
        guess = dict(state0)
        total = [0]
        first = []
        res = {}

        def dd(lam):
            sh = (lam, None) if self.carrier == "n" else (None, lam)
            r = solve_dd(sh, guess)
            total[0] += r["iters"]
            first.append(r["iters"])
            if not r.get("converged", False):
                raise _NotConverged(r)
            guess.update(psi=r["psi"], phin=r["phin"], phip=r["phip"])
            res["r"] = r
            return r

        def upd(r):
            (ln, lp), q = self.update(r)
            return (ln if self.carrier == "n" else lp), q

        if lam0 is None:
            lam0 = upd(dict(state0))[0]
        try:
            r, lam, k, hist = sp_loop(dd, upd, lam0, tol=self.tol, max_outer=max_outer)
        except _NotConverged as e:
            r = e.r
            r["converged"] = False
            r["iters"] = total[0]
            return r
        except RuntimeError:
            r = res["r"]
            r["converged"] = False
            r["iters"] = total[0]
            return r
        # step_iters: the first (predictor) solve's iterations - what the bias
        # step itself cost; the continuation's step control reads it
        r["lam"], r["outer"], r["iters"], r["step_iters"] = lam, k, total[0], first[0]
        return r


class _NotConverged(Exception):
    def __init__(self, r):
        self.r = r
