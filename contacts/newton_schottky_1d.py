"""1D quasi-Fermi Newton solve with Schottky contacts (plans/schottky_plan.md).

Wraps core/newton_solver_qf.py's residual and Jacobian without changing
it: the interior rows, Poisson and the ohmic ends are exactly that
module's. At a Schottky end:

  * psi is Dirichlet at the barrier-pinned value (contacts.schottky.boundary_state);
  * the electron/hole continuity rows become half-cell flux balances with
    the thermionic-emission flux into the metal (a Robin condition), so
    phin, phip at the interface node are free unknowns;
  * the tunneling current (contacts.schottky.path_tunneling, path = the
    mesh line from the interface inward) is booked as a carrier sink at
    the turning points.

Sign convention: sigma = +1 for the metal on the left (x[0]), -1 on the
right. Current densities are along +x like core's Jn/Jp.
"""
import warnings

import numpy as np
import scipy.sparse as sp

from core.params import Q, Material
from core import physics as ph
from core.materials import MaterialField
from core.newton_numerics import ROW_TOL, damped_newton, uniform_step_clip, componentwise_step_clip
from core.newton_solver_qf import (_residual_only, _residual_and_jacobian, poisson_row_scale,
                                   continuity_row_scale, unpack_qf, MAX_QF_STEP)
from core.solver import contact_values
from contacts import schottky as sk

PATH_LENGTH = 30e-7     # tunneling path length from the interface (cm); 30 nm
DE_PER_VT = 0.25        # tunneling energy-bin width in units of Vt


class SchottkyEnd:
    """Everything fixed for one Schottky end at one bias."""

    def __init__(self, contact, side, x, Cdop, mat, mf, V, dphi_eq=0.0):
        self.c = contact
        self.side = side
        self.sigma = 1 if side == "left" else -1
        N = len(x)
        self.b = 0 if side == "left" else N - 1          # interface node
        self.nb = 1 if side == "left" else N - 2         # its neighbor
        self.e = 0 if side == "left" else N - 2          # boundary edge
        self.h = abs(x[self.nb] - x[self.b])
        self.cv = self.h / 2.0
        self.V = V
        Vt = mf.Vt
        self.ni = mf.ni_arr[self.b]
        self.dEi = mf.delta_Ei_arr[self.b]
        majority = "n" if Cdop[self.b] >= 0 else "p"
        self.majority = majority
        st = sk.boundary_state(contact, mat.chi_eV, mat.Eg_eV, mat.Nc, mat.Nv, self.ni, Vt,
                               majority, V=V, dEi=self.dEi)
        self.psi_s, self.n_B, self.p_B = st["psi_s"], st["n_B"], st["p_B"]
        self.phi_bn, self.phi_bp = st["phi_bn"], st["phi_bp"]
        self.vn = sk.te_velocity(contact.richardson_n, mat.T, mat.Nc)
        self.vp = sk.te_velocity(contact.richardson_p, mat.T, mat.Nv)
        self.eps = mf.eps_edge[self.e]
        self.dphi_eq = dphi_eq
        # Path: mesh nodes within PATH_LENGTH of the interface, interface
        # first - never the opposite contact's node (a sink booked on its
        # Dirichlet row could not be zeroed: Newton stalled on a 30 nm device).
        dist = np.abs(x - x[self.b])
        other = N - 1 if side == "left" else 0
        nodes = np.nonzero((dist <= PATH_LENGTH) & (np.arange(N) != other))[0]
        self.path = nodes if side == "left" else nodes[::-1]
        self.s = dist[self.path]
        self.lnNc = Vt * np.log(mat.Nc / mf.ni_arr[self.path])
        self.lnNv = Vt * np.log(mat.Nv / mf.ni_arr[self.path])
        self.dEi_path = mf.delta_Ei_arr[self.path]


def _lowering(end, psi):
    """(factor, dfactor/dpsi_nb, dfactor/dpsi_b) of exp((dPhi - dPhi_eq)/Vt)."""
    if not end.c.barrier_lowering:
        return 1.0, 0.0, 0.0
    d, dd_dnb = _lowering_of_field(end, psi)
    Vt = end.Vt
    f = np.exp((d - end.dphi_eq) / Vt)
    return f, f / Vt * dd_dnb, -f / Vt * dd_dnb


E_SMOOTH = 1e3      # V/cm: width of the smooth positive part at flat band


def _lowering_of_field(end, psi):
    """Image-force lowering (V) at the interface field and its derivative
    w.r.t. psi at the neighbor node. Only a DEPLETING field (band falling
    away from the interface for the majority carrier) lowers a barrier; in
    accumulation there is none to lower (applying |E| there broke every
    forward sweep at flat band). A smooth positive part of the field keeps
    the sqrt's derivative finite at flat band; the barrier caps it."""
    E = (psi[end.nb] - psi[end.b]) / end.h * end.sigma
    if end.majority == "p":
        E = -E
    root = np.sqrt(E * E + E_SMOOTH ** 2)
    E_pos = 0.5 * (E + root)
    dEpos_dE = 0.5 * (1.0 + E / root)
    d, dd = sk.image_force_lowering(E_pos, end.eps)
    cap = end.phi_bn if end.majority == "n" else end.phi_bp
    if d > cap:
        return cap, 0.0
    dE_dnb = end.sigma / end.h * (-1.0 if end.majority == "p" else 1.0)
    return d, dd * dEpos_dE * dE_dnb


def _tunneling(end, psi, phin, phip, Vt, T, carrier):
    """Tunneling at one end for one carrier. Returns (J_tun, rows, w_sink,
    d_psi, d_qf): J_tun (A/cm^2, > 0 into the metal); the sink is
    J_tun split onto mesh nodes `rows` with weights w_sink (sum = J_tun);
    d_psi (len(rows), M) and d_qf (len(rows), M) are the derivatives of
    each node's sink w.r.t. psi and the carrier's quasi-Fermi potential at
    the path nodes."""
    P = end.path
    if carrier == "n":
        U = -psi[P] - end.dEi_path + end.lnNc
        EF = -phin[P]
        EF_m = -end.V
        m_rel, A = end.c.m_tunnel_n, end.c.richardson_n
        dU_dpsi, dEF_dqf = -1.0, -1.0
    else:
        U = psi[P] + end.dEi_path + end.lnNv
        EF = phip[P]
        EF_m = end.V
        m_rel, A = end.c.m_tunnel_p, end.c.richardson_p
        dU_dpsi, dEF_dqf = 1.0, 1.0
    dE = DE_PER_VT * Vt
    M = len(P)
    # Only energies above the band minimum beyond the interface have a
    # turning point (the emitter band edge). The bin that contains it is
    # weighted by its fraction above the edge and evaluated in the middle of
    # that part, so the integral's lower limit is continuous: with whole
    # bins, near flat band (Gamma ~ 1) each bin switched on/off as the edge
    # crossed its midpoint and Newton stalled (DEVELOPMENT_LOG).
    im = 1 + int(np.argmin(U[1:])) if M > 1 else 0
    U_min = U[im]
    E = sk.energy_grid(U[0], U_min, dE)
    if len(E) == 0 or M < 2:
        return 0.0, P, np.zeros(M), np.zeros((M, M)), np.zeros((M, M))
    lo_edge = E - dE / 2
    wgt = np.clip((E + dE / 2 - U_min) / dE, 0.0, 1.0)
    partial = lo_edge < U_min
    E = np.where(partial, 0.5 * (U_min + E + dE / 2), E)
    r = sk.path_tunneling(end.s, U, EF, EF_m, E, m_rel, A, T, Vt)
    r["j"] = r["j"] * wgt
    r["dj_dU"] = r["dj_dU"] * wgt[:, None]
    r["dj_dEF"] = r["dj_dEF"] * wgt[:, None]
    dj_dUmin = np.where(partial, -r["j"] / np.maximum(wgt, 1e-300) / dE, 0.0)
    K = len(E)
    W = np.zeros((K, M))
    W[np.arange(K), r["jt"]] = 1.0 - r["t"]
    W[np.arange(K), np.minimum(r["jt"] + 1, M - 1)] += r["t"]
    W[~r["ok"]] = 0.0
    jdE = r["j"] * dE
    w_sink = W.T @ jdE
    d_psi = W.T @ (r["dj_dU"] * dE) * dU_dpsi
    d_psi[:, im] += W.T @ (dj_dUmin * dE) * dU_dpsi
    # The split (1 - t, t) itself moves with U at the two bracketing nodes.
    jt, jt1 = r["jt"], np.minimum(r["jt"] + 1, M - 1)
    for col, dt in ((jt, r["dt_dU"][:, 0]), (jt1, r["dt_dU"][:, 1])):
        g = jdE * dt * dU_dpsi
        np.add.at(d_psi, (jt, col), -g)
        np.add.at(d_psi, (jt1, col), g)
    Pq = np.zeros((K, M))
    Pq[np.arange(K), r["jt"]] = r["dj_dEF"][:, 0]
    Pq[np.arange(K), np.minimum(r["jt"] + 1, M - 1)] += r["dj_dEF"][:, 1]
    d_qf = W.T @ (Pq * dE) * dEF_dqf
    return float(jdE.sum()), P, w_sink, d_psi, d_qf


def _boundary(end, psi, phin, phip, n, p, mf, cs, want_jac):
    """Residuals of the two continuity rows of a Schottky end (without the
    tunneling sink) and, if want_jac, their (row, col, value) entries.
    Also returns the TE current densities (along +x)."""
    N = len(psi)
    b, nb, e, sig, cv = end.b, end.nb, end.e, end.sigma, end.cv
    Vt = mf.Vt
    lo, hi = (b, nb) if sig > 0 else (nb, b)
    h = end.h
    mun, mup = mf.mu_n_edge[e], mf.mu_p_edge[e]
    n_avg = (n[lo] + n[hi]) / 2.0
    p_avg = (p[lo] + p[hi]) / 2.0
    dphin = phin[hi] - phin[lo]
    dphip = phip[hi] - phip[lo]
    Jn_e = -Q * mun * n_avg * dphin / h
    Jp_e = -Q * mup * p_avg * dphip / h

    f, df_nb, df_b = _lowering(end, psi)
    Jn_if = sig * Q * end.vn * (n[b] - end.n_B * f)
    Jp_if = -sig * Q * end.vp * (p[b] - end.p_B * f)

    ni, tn, tp = mf.ni_arr[b], mf.tau_n_arr[b], mf.tau_p_arr[b]
    denom = tp * (n[b] + ni) + tn * (p[b] + ni)
    num = n[b] * p[b] - ni ** 2
    R = num / denom
    Rn = (sig * (Jn_e - Jn_if) / cv - Q * R) / cs
    Rp = (sig * (Jp_e - Jp_if) / cv + Q * R) / cs
    if not want_jac:
        return Rn, Rp, Jn_if, Jp_if, None

    ent = []
    # d n/d psi = n/Vt, dn/dphin = -n/Vt; dp/dpsi = -p/Vt, dp/dphip = p/Vt.
    dR_dn = (p[b] * denom - num * tp) / denom ** 2
    dR_dp = (n[b] * denom - num * tn) / denom ** 2
    rn, rp = N + b, 2 * N + b
    k = sig / cv / cs
    # Edge flux Jn_e w.r.t. node m in (lo, hi).
    for m, s_m in ((lo, -1.0), (hi, 1.0)):
        dJn_dpsi = -Q * mun / h * (n[m] / Vt / 2.0) * dphin
        dJn_dphin = -Q * mun / h * ((-n[m] / Vt / 2.0) * dphin + s_m * n_avg)
        dJp_dpsi = -Q * mup / h * (-p[m] / Vt / 2.0) * dphip
        dJp_dphip = -Q * mup / h * ((p[m] / Vt / 2.0) * dphip + s_m * p_avg)
        ent += [(rn, m, k * dJn_dpsi), (rn, N + m, k * dJn_dphin),
                (rp, m, k * dJp_dpsi), (rp, 2 * N + m, k * dJp_dphip)]
    # TE flux.
    ent += [(rn, b, -k * sig * Q * end.vn * n[b] / Vt), (rn, N + b, k * sig * Q * end.vn * n[b] / Vt),
            (rp, b, k * sig * Q * end.vp * (-p[b] / Vt)), (rp, 2 * N + b, k * sig * Q * end.vp * p[b] / Vt)]
    if end.c.barrier_lowering:
        for col, dfd in ((nb, df_nb), (b, df_b)):
            ent += [(rn, col, k * sig * Q * end.vn * end.n_B * dfd),
                    (rp, col, -k * sig * Q * end.vp * end.p_B * dfd)]
    # Recombination.
    dRdpsi = dR_dn * n[b] / Vt - dR_dp * p[b] / Vt
    ent += [(rn, b, -Q * dRdpsi / cs), (rn, N + b, -Q * dR_dn * (-n[b] / Vt) / cs),
            (rn, 2 * N + b, -Q * dR_dp * p[b] / Vt / cs),
            (rp, b, Q * dRdpsi / cs), (rp, N + b, Q * dR_dn * (-n[b] / Vt) / cs),
            (rp, 2 * N + b, Q * dR_dp * p[b] / Vt / cs)]
    return Rn, Rp, Jn_if, Jp_if, ent


class Problem:
    """F(U), J(U) for one bias point."""

    def __init__(self, x, Cdop, mat, ends, V, dphi_eq=None):
        self.x, self.Cdop, self.mat = x, Cdop, mat
        self.mf = MaterialField.uniform(mat, x)
        mf = self.mf
        N = len(x)
        self.N = N
        dphi_eq = dphi_eq or {}
        Vt = mf.Vt
        psi_eq = ph.equilibrium_bulk_potential_arr(Vt, mf.ni_arr, Cdop, mf.delta_Ei_arr)
        self.psi_bc = np.zeros(2)
        self.phin_bc = np.zeros(2)
        self.phip_bc = np.zeros(2)
        self.ends = []
        for k, side in enumerate(("left", "right")):
            i = 0 if side == "left" else N - 1
            c = ends.get(side)
            if c is None:
                n_bc, p_bc = contact_values(mf, Cdop[i], ni=mf.ni_arr[i])
                self.psi_bc[k] = psi_eq[i] + V[k]
                self.phin_bc[k] = self.psi_bc[k] + mf.delta_Ei_arr[i] - Vt * np.log(n_bc / mf.ni_arr[i])
                self.phip_bc[k] = self.psi_bc[k] + mf.delta_Ei_arr[i] + Vt * np.log(p_bc / mf.ni_arr[i])
            else:
                end = SchottkyEnd(c, side, x, Cdop, mat, mf, V[k], dphi_eq.get(side, 0.0))
                end.Vt = Vt
                self.ends.append(end)
                self.psi_bc[k] = end.psi_s
                self.phin_bc[k] = self.phip_bc[k] = V[k]       # start values only
        self.psi_eq = psi_eq
        h_typ = np.min(np.diff(x))
        self.ps = poisson_row_scale(mf, h_typ)
        self.cs = continuity_row_scale(mf, h_typ)
        self.cvol = ph._control_volumes(x)

    def densities(self, U):
        psi, phin, phip = unpack_qf(U, self.N)
        mf = self.mf
        n = mf.ni_arr * np.exp((psi - phin + mf.delta_Ei_arr) / mf.Vt)
        p = mf.ni_arr * np.exp((phip - psi - mf.delta_Ei_arr) / mf.Vt)
        return psi, phin, phip, n, p

    def evaluate(self, U, want_jac=True):
        N, mf, cs = self.N, self.mf, self.cs
        args = (U, self.x, self.Cdop, mf, self.psi_bc, self.phin_bc, self.phip_bc, self.ps, cs)
        if want_jac:
            F, J = _residual_and_jacobian(*args)
        else:
            F, J = _residual_only(*args), None
        if not self.ends:
            return F, J, {}
        psi, phin, phip, n, p = self.densities(U)
        F = F.copy()
        replace = []
        ent = []
        info = {}
        for end in self.ends:
            Rn, Rp, Jn_if, Jp_if, e_b = _boundary(end, psi, phin, phip, n, p, mf, cs, want_jac)
            F[N + end.b], F[2 * N + end.b] = Rn, Rp
            replace += [N + end.b, 2 * N + end.b]
            if want_jac:
                ent += e_b
            Jt = {}
            for carrier in ("n", "p"):
                if not end.c.tunneling:
                    Jt[carrier] = 0.0
                    continue
                Jc, rows, w, d_psi, d_qf = _tunneling(end, psi, phin, phip, mf.Vt, self.mat.T, carrier)
                Jt[carrier] = Jc
                # Sink of carriers at the turning points: electron row gets
                # -(w/cv)/cs (R_n = div Jn - q(R - G), G = -w/(q cv)); the
                # hole row +(w/cv)/cs.
                sgn = -1.0 if carrier == "n" else 1.0
                off = N if carrier == "n" else 2 * N
                cv = self.cvol[rows]
                F[off + rows] += sgn * w / cv / cs
                if want_jac:
                    rr = np.repeat(off + rows, len(rows))
                    cc = np.tile(rows, len(rows))
                    scale = np.repeat(sgn / cv / cs, len(rows))
                    ent += list(zip(rr, cc, scale * d_psi.ravel()))
                    ent += list(zip(rr, off + cc, scale * d_qf.ravel()))
            info[end.side] = dict(Jn_te=Jn_if, Jp_te=Jp_if, Jn_tun=Jt["n"], Jp_tun=Jt["p"],
                                  J=Jn_if + Jp_if + end.sigma * (Jt["n"] - Jt["p"]))
        if want_jac:
            J = J.tocsr()
            keep = np.ones(3 * N)
            keep[replace] = 0.0
            J = sp.diags(keep) @ J
            if ent:
                r, c, v = (np.array(a) for a in zip(*ent))
                J = J + sp.coo_matrix((v.astype(float), (r.astype(int), c.astype(int))),
                                      shape=(3 * N, 3 * N)).tocsr()
            J = J.tocsc()
        return F, J, info


def solve(x, Cdop, mat: Material, ends, V, init=None, dphi_eq=None, f_tol=ROW_TOL, maxiter=80,
          verbose=False):
    """One bias point. ends = {"left": SchottkyContact or None (ohmic),
    "right": ...}; V = (V_left, V_right). init = (psi, phin, phip) of a
    nearby solution, or None for a cold start (equilibrium Poisson with the
    contacts' Dirichlet psi, flat quasi-Fermi levels between the two
    biases). dphi_eq: {side: image-force lowering at equilibrium} (needed
    with barrier_lowering; see equilibrium_lowering).
    Returns dict(psi, phin, phip, n, p, converged, iters, J, J_left, J_right, info)."""
    pb = Problem(x, Cdop, mat, ends, V, dphi_eq)
    N = pb.N
    Vt = pb.mf.Vt
    if init is None:
        frac = (x - x[0]) / (x[-1] - x[0])
        qf = V[0] + (V[1] - V[0]) * frac
        psi0 = pb.psi_eq + qf
        psi0[0], psi0[-1] = pb.psi_bc
        psi0, *_ = ph.solve_poisson(x, Cdop, pb.mf, qf, qf, psi0, eps=pb.mf.eps_edge, ni=pb.mf.ni_arr,
                                    delta_Ei=pb.mf.delta_Ei_arr)
        rng = np.random.default_rng(0)
        phin0 = qf + 1e-6 * Vt * rng.standard_normal(N)
        phip0 = qf + 1e-6 * Vt * rng.standard_normal(N)
    else:
        psi0, phin0, phip0 = (a.copy() for a in init)
        # Predictor: a bias change at an end shifts psi and both quasi-Fermi
        # levels together along a linear ramp, so the densities (and the
        # interface field) do not jump. Without it the image-force term saw
        # the whole bias step across the first 0.05 nm edge (~1e7 V/cm).
        frac = (x - x[0]) / (x[-1] - x[0])
        d_left, d_right = pb.psi_bc[0] - psi0[0], pb.psi_bc[1] - psi0[-1]
        ramp = d_left * (1.0 - frac) + d_right * frac
        psi0 += ramp
        phin0 += ramp
        phip0 += ramp
    psi0[0], psi0[-1] = pb.psi_bc
    for k, i in enumerate((0, N - 1)):
        if not any(e.b == i for e in pb.ends):
            phin0[i], phip0[i] = pb.phin_bc[k], pb.phip_bc[k]
    U0 = np.concatenate([psi0, phin0, phip0])
    U, merit, it, conv = damped_newton(
        U0, lambda U: pb.evaluate(U)[:2], lambda U: pb.evaluate(U, want_jac=False)[0],
        step_clip=lambda d: uniform_step_clip(d, N, max_psi=1.0, max_qf=MAX_QF_STEP),
        trial_clip=lambda d: componentwise_step_clip(d, N, max_psi=1.0, Vt=Vt),
        tol=f_tol, maxiter=maxiter, verbose=verbose, label="Newton(Schottky)")
    if not conv:
        warnings.warn(f"Schottky 1D solve did not converge at V={V} (merit {merit:.3e}, {it} iterations)")
    _, _, info = pb.evaluate(U, want_jac=False)
    psi, phin, phip, n, p = pb.densities(U)
    out = dict(psi=psi, phin=phin, phip=phip, n=n, p=p, converged=conv, iters=it, merit=merit, info=info,
               problem=pb)
    out.update(terminal_currents(pb, U, info))
    return out


def terminal_currents(pb, U, info):
    """Current density (A/cm^2, along +x) at each end. A Schottky end
    reads its own interface current (TE + tunneling); an ohmic end the
    plain-gradient flux of its boundary edge (the two agree when the
    solution is converged and recombination between them is negligible)."""
    psi, phin, phip, n, p = pb.densities(U)
    x, mf = pb.x, pb.mf
    out = {}
    for side, (lo, hi) in (("left", (0, 1)), ("right", (pb.N - 2, pb.N - 1))):
        if side in info:
            out["J_" + side] = info[side]["J"]
            continue
        e = lo
        h = x[hi] - x[lo]
        Jn = -Q * mf.mu_n_edge[e] * (n[lo] + n[hi]) / 2 * (phin[hi] - phin[lo]) / h
        Jp = -Q * mf.mu_p_edge[e] * (p[lo] + p[hi]) / 2 * (phip[hi] - phip[lo]) / h
        out["J_" + side] = Jn + Jp
    return out


def equilibrium_lowering(x, Cdop, mat, ends):
    """Image-force lowering at equilibrium per Schottky end (the reference
    that keeps equilibrium current-free when barrier_lowering is on)."""
    import dataclasses
    plain = {k: (None if v is None else dataclasses.replace(v, barrier_lowering=False, tunneling=False))
             for k, v in ends.items()}
    r = solve(x, Cdop, mat, plain, (0.0, 0.0))
    out = {}
    mf = r["problem"].mf
    for end in r["problem"].ends:
        out[end.side] = float(_lowering_of_field(end, r["psi"])[0])
    return out


def contact_grid(L, h0=0.05e-7, growth=1.05, h_max=2e-7, side="left"):
    """1D grid of length L (cm) refined at the metal: spacing h0 at the
    contact, growing geometrically to h_max."""
    xs = [0.0]
    h = h0
    while xs[-1] < L:
        xs.append(xs[-1] + h)
        h = min(h * growth, h_max)
    x = np.array(xs)
    x *= L / x[-1]
    if side == "right":
        x = L - x[::-1]
    return x


J_CAP = 1e8     # A/cm^2: far above any real contact (a S/D contact carries <~1e7)


def sweep(x, Cdop, mat, ends, V_targets, side="left", dV_max=0.1, dphi_eq=None, min_step=1e-4, J_cap=J_CAP):
    """Bias sweep of one end (the other grounded), continuation from
    equilibrium with the bias step halved on a failed point. Returns a
    list of solve() results at V_targets (in the given order, each walked
    to from the previous one). A target past the point where |J| exceeds
    J_cap is skipped (None in the list; the walk resumes from there toward
    the next target): deep forward accumulation at a degenerate contact
    carries 1e8-1e10 A/cm^2 and only gets stiffer, with no device meaning."""
    k = 0 if side == "left" else 1

    def bias(v):
        return (v, 0.0) if k == 0 else (0.0, v)

    r = solve(x, Cdop, mat, ends, (0.0, 0.0), dphi_eq=dphi_eq)
    if not r["converged"]:
        raise RuntimeError("equilibrium did not converge")
    v, out = 0.0, []
    for target in V_targets:
        step = dV_max
        capped = False
        while abs(target - v) > 1e-12:
            if max(abs(r["J_left"]), abs(r["J_right"])) > J_cap and abs(target) > abs(v) \
                    and np.sign(target) == np.sign(v):
                capped = True
                break
            v_try = v + np.clip(target - v, -step, step)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = solve(x, Cdop, mat, ends, bias(v_try), init=(r["psi"], r["phin"], r["phip"]),
                          dphi_eq=dphi_eq, maxiter=40)
            if t["converged"]:
                r, v = t, v_try
                step = min(step * 1.5, dV_max)
            else:
                step /= 2
                if step < min_step:
                    raise RuntimeError(f"bias step collapsed at V = {v_try:.4f}")
        out.append(None if capped else r)
    return out


def contact_resistivity(mat, doping, contact, L=30e-7, h0=0.05e-7, dV=1e-3):
    """Specific contact resistivity (Ohm cm^2) of `contact` on a uniformly
    doped slab of length L: dV/dJ at V -> 0, minus the same slab with an
    ohmic contact (removes the slab's own resistance)."""
    x = contact_grid(L, h0=h0, h_max=max(0.5e-7, h0))
    Cdop = np.full(len(x), doping)

    def r(ends):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r0 = solve(x, Cdop, mat, ends, (0.0, 0.0))
            init = (r0["psi"], r0["phin"], r0["phip"])
            rp = solve(x, Cdop, mat, ends, (dV, 0.0), init=init)
            rm = solve(x, Cdop, mat, ends, (-dV, 0.0), init=init)
        if not (r0["converged"] and rp["converged"] and rm["converged"]):
            raise RuntimeError(f"contact_resistivity: solve failed ({contact}, N={doping:.1e})")
        return 2 * dV / (rp["J_right"] - rm["J_right"])

    return r({"left": contact, "right": None}) - r({"left": None, "right": None})
