"""Fully coupled Newton solve, quasi-Fermi-potential unknowns and
plain-gradient current (same base formulation as newton_solver_qf.py), EXTENDED
with a field-dependent impact-ionization (avalanche) generation term in both
continuity equations.

THIS IS A SPECIAL, OPT-IN, NON-DEFAULT SOLVER, reached via
math_model="newton_avalanche" or its own driver (main_avalanche.py). It keeps
its own private copies of the QF building blocks (ARCHITECTURE.md: avalanche
stays standalone); the only shared pieces it uses are solver-agnostic
numerics in core/ (newton_numerics.py, jacobian_scaling.py, arclength.py).

THE PHYSICS: impact ionization is an extra local electron-hole-pair
generation rate added to both continuity equations (opposite sign to SRH R):

    dJn/dx = q*(R - G_ii)      dJp/dx = -q*(R - G_ii)
    G_ii = (alpha_n(F_n)*|Jn| + alpha_p(F_p)*|Jp|) / q

with the van Overstraeten-de Man coefficients (avalanche.py). F_n, F_p are
the DRIVING FORCES - see DRIVING FORCE below; G_ii's dependence on |J| (the
very quantity the continuity equations solve for) is the positive feedback
that makes the multiplication factor diverge at breakdown.

DRIVING FORCE (driving_force= "hybrid" (default) | "gradqf" | "efield"),
per edge, per carrier:
  - "efield": F = |E| = |dpsi/dx| - the textbook local-field model. Correct,
    but it evaluates alpha(E)*|J| in the thin high-field slice of a heavily
    doped side, where the carrier is majority and its plain-gradient current
    is dominated by roundoff (a huge conductance q*mu*p/h times a quasi-Fermi
    difference below double-precision resolution). At 1e20-1e21 doping that
    noise is hundreds of times the real leakage current, and the solve fails
    even at Va=-0.25V (DEVELOPMENT_LOG.md session 23).
  - "gradqf": F = |dphi/dx| of that carrier - the default of Genius-TCAD
    (src/solution/control.cc, II_Force=GradQf) and of FLOOXS/Charon's
    van Overstraeten model (Emfn/Emfp). Physically it is the force that
    actually heats the carrier; a majority carrier in quasi-equilibrium has
    |grad phi| ~ 0, so the noisy slice drops out. Its known weakness is the
    opposite limit: where a carrier's density is tiny but its current is
    large (the minority edge of a strongly multiplying depletion region),
    |grad phi| = |J|/(q*mu*c) overshoots the field and produces a spurious
    voltage snapback at ~1e-2 A/cm^2 in these diodes.
  - "hybrid": F = w*|grad phi| + (1-w)*|E|, w = c^2/(c^2 + c_ref^2), c the
    carrier's own edge-averaged density - |grad phi| where the carrier is
    dense (removes the majority-current noise), |E| where it is sparse
    (removes the low-density overshoot). The same idea as FLOOXS/Charon's
    documented n*F/(n+n0) damping of the gradient driving force.

alpha(F) is continuous down to F=0 (AvalancheModel.E_floor_V_cm is only a
guard against 0-division; the old 1.75e5 V/cm hard floor was a jump in
alpha that Newton could not settle across).

NEWTON: core/newton_numerics.damped_newton (row-normalized merit). Dirichlet
rows are unit rows (u - u_bc = 0); the Ruiz equilibration inside the linear
solve makes the old pivoting-safety diagonal scaling unnecessary, and unit
rows keep the arc-length (Va-as-unknown) extension exactly consistent.
"""
import warnings

import numpy as np
import scipy.sparse as sp

from core.params import Q, Material
from core import physics as ph
from core.solver import contact_values
from core.newton_numerics import (ROW_TOL, damped_newton, uniform_step_clip, componentwise_step_clip,
                                  edge_current_noise, resolved_current)
from avalanche.avalanche import AvalancheModel, ionization_coeffs

DRIVING_FORCES = ("hybrid", "gradqf", "efield")
# Density (cm^-3) where the hybrid driving force is half |grad phi|, half |E|.
DEFAULT_REF_DENSITY_CM3 = 1.0e16
# Same rationale as newton_solver_qf.py's MAX_QF_STEP.
_MAX_QF_STEP = 5.0
_MAX_PSI_STEP = 1.0


def _poisson_scale(mat: Material, h_typ: float) -> float:
    return mat.eps * mat.Vt / h_typ ** 2


def _continuity_scale(mat: Material, h_typ: float) -> float:
    return Q * mat.Dn * mat.ni / h_typ


def _force_weight(c, force, c_ref):
    """w and dw/dc for F = w*|grad phi| + (1-w)*|E| (see module docstring)."""
    if force == "gradqf":
        return np.ones_like(c), np.zeros_like(c)
    if force == "efield":
        return np.zeros_like(c), np.zeros_like(c)
    w = 1.0 / (1.0 + (c_ref / c) ** 2)
    return w, 2.0 * w * (1.0 - w) / c


class AvalancheProblem:
    """The discretized avalanche drift-diffusion system on one mesh, with
    the applied bias Va as a parameter. Unknowns U = [psi, phin, phip]."""

    def __init__(self, x, Cdop, mat: Material, psi_eq, ii_model: AvalancheModel = None,
                 driving_force="hybrid", ref_density_cm3=DEFAULT_REF_DENSITY_CM3):
        if driving_force not in DRIVING_FORCES:
            raise ValueError(f"driving_force must be one of {DRIVING_FORCES}, got {driving_force!r}")
        self.x, self.Cdop, self.mat, self.psi_eq = x, Cdop, mat, psi_eq
        self.ii_model = ii_model or AvalancheModel.si_von_overstraeten_de_man()
        self.force, self.c_ref = driving_force, ref_density_cm3
        self.N = N = len(x)
        self.h = np.diff(x)
        self.cvol_i = ph._control_volumes(x)[1:-1]
        h_typ = np.min(self.h)
        self.poisson_scale = _poisson_scale(mat, h_typ)
        self.cont_scale = _continuity_scale(mat, h_typ)
        Vt = mat.Vt
        n0, p0 = contact_values(mat, Cdop[0])
        nL, pL = contact_values(mat, Cdop[-1])
        # u_bc(Va) = offset + [Va at the left (biased) contact, 0 at the right]
        self.bc_rows = np.array([0, N - 1, N, 2 * N - 1, 2 * N, 3 * N - 1])
        self.bc_offset = np.array([psi_eq[0], psi_eq[-1],
                                   psi_eq[0] - Vt * np.log(n0 / mat.ni), psi_eq[-1] - Vt * np.log(nL / mat.ni),
                                   psi_eq[0] + Vt * np.log(p0 / mat.ni), psi_eq[-1] + Vt * np.log(pL / mat.ni)])
        self.bc_dVa = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    # ---- helpers -------------------------------------------------------
    def bc_values(self, Va):
        return self.bc_offset + Va * self.bc_dVa

    def apply_bc(self, U, Va):
        U = U.copy()
        U[self.bc_rows] = self.bc_values(Va)
        return U

    def densities(self, U):
        N, mat = self.N, self.mat
        psi, phin, phip = U[:N], U[N:2 * N], U[2 * N:]
        n = mat.ni * np.exp((psi - phin) / mat.Vt)
        p = mat.ni * np.exp((phip - psi) / mat.Vt)
        return psi, phin, phip, n, p

    def _edges(self, psi, phin, phip, n, p):
        mat, h = self.mat, self.h
        n_avg = 0.5 * (n[:-1] + n[1:])
        p_avg = 0.5 * (p[:-1] + p[1:])
        dpsi, dphin, dphip = np.diff(psi), np.diff(phin), np.diff(phip)
        Jn = -Q * mat.mu_n * n_avg * dphin / h
        Jp = -Q * mat.mu_p * p_avg * dphip / h
        Eabs = np.abs(dpsi) / h
        wn, dwn = _force_weight(n_avg, self.force, self.c_ref)
        wp, dwp = _force_weight(p_avg, self.force, self.c_ref)
        Fqn, Fqp = np.abs(dphin) / h, np.abs(dphip) / h
        Fn = wn * Fqn + (1.0 - wn) * Eabs
        Fp = wp * Fqp + (1.0 - wp) * Eabs
        an, _, dan, _ = ionization_coeffs(Fn, self.ii_model)
        _, ap, _, dap = ionization_coeffs(Fp, self.ii_model)
        Gii_e = (an * np.abs(Jn) + ap * np.abs(Jp)) / Q
        return dict(n_avg=n_avg, p_avg=p_avg, dpsi=dpsi, dphin=dphin, dphip=dphip, Jn=Jn, Jp=Jp,
                    Eabs=Eabs, wn=wn, dwn=dwn, wp=wp, dwp=dwp, Fqn=Fqn, Fqp=Fqp,
                    an=an, ap=ap, dan=dan, dap=dap, Gii_e=Gii_e)

    def _srh(self, n, p):
        mat = self.mat
        denom = mat.tau_p * (n + mat.ni) + mat.tau_n * (p + mat.ni)
        num = n * p - mat.ni ** 2
        return num / denom, denom, num

    # ---- residual / Jacobian (Va enters only through the BC rows) ------
    def residual(self, U, Va):
        N, mat, h = self.N, self.mat, self.h
        psi, phin, phip, n, p = self.densities(U)
        hm, hp, cv = h[:-1], h[1:], self.cvol_i
        F = np.empty(3 * N)
        F[self.bc_rows] = U[self.bc_rows] - self.bc_values(Va)
        lap_m, lap_p = mat.eps / hm / cv, mat.eps / hp / cv
        F[1:N - 1] = (lap_p * (psi[2:] - psi[1:-1]) - lap_m * (psi[1:-1] - psi[:-2])
                      - Q * (n[1:-1] - p[1:-1] - self.Cdop[1:-1])) / self.poisson_scale
        e = self._edges(psi, phin, phip, n, p)
        R, _, _ = self._srh(n, p)
        Gn = (hm * e["Gii_e"][:-1] + hp * e["Gii_e"][1:]) / (2.0 * cv)
        Jn, Jp = e["Jn"], e["Jp"]
        F[N + 1:2 * N - 1] = ((Jn[1:] - Jn[:-1]) / cv - Q * (R[1:-1] - Gn)) / self.cont_scale
        F[2 * N + 1:3 * N - 1] = ((Jp[1:] - Jp[:-1]) / cv + Q * (R[1:-1] - Gn)) / self.cont_scale
        return F

    def residual_and_jacobian(self, U, Va):
        """(F, J) with J = dF/dU (3N x 3N sparse). dF/dVa is -bc_dVa in
        the Dirichlet rows and 0 elsewhere (see dF_dVa)."""
        N, mat, h, Vt = self.N, self.mat, self.h, self.mat.Vt
        psi, phin, phip, n, p = self.densities(U)
        hm, hp, cv = h[:-1], h[1:], self.cvol_i
        F = self.residual(U, Va)
        e = self._edges(psi, phin, phip, n, p)
        R, denom, num = self._srh(n, p)
        lap_m, lap_p = mat.eps / hm / cv, mat.eps / hp / cv

        dn_dpsi, dn_dphin = n / Vt, -n / Vt
        dp_dpsi, dp_dphip = -p / Vt, p / Vt
        # Edge current derivatives; "_L"/"_R" = w.r.t. the edge's left/right node.
        kn, kp = -Q * mat.mu_n / h, -Q * mat.mu_p / h
        dJn_dpsi_L = kn * (dn_dpsi[:-1] / 2) * e["dphin"]
        dJn_dpsi_R = kn * (dn_dpsi[1:] / 2) * e["dphin"]
        dJn_dphin_L = kn * ((dn_dphin[:-1] / 2) * e["dphin"] - e["n_avg"])
        dJn_dphin_R = kn * ((dn_dphin[1:] / 2) * e["dphin"] + e["n_avg"])
        dJp_dpsi_L = kp * (dp_dpsi[:-1] / 2) * e["dphip"]
        dJp_dpsi_R = kp * (dp_dpsi[1:] / 2) * e["dphip"]
        dJp_dphip_L = kp * ((dp_dphip[:-1] / 2) * e["dphip"] - e["p_avg"])
        dJp_dphip_R = kp * ((dp_dphip[1:] / 2) * e["dphip"] + e["p_avg"])

        # Driving-force derivatives: F = w*Fq + (1-w)*|E|, w = w(c_avg).
        sE, sn, sp_ = np.sign(e["dpsi"]), np.sign(e["dphin"]), np.sign(e["dphip"])
        dE_L, dE_R = -sE / h, sE / h
        gn = (e["Fqn"] - e["Eabs"]) * e["dwn"]      # dFn/dn_avg
        gp = (e["Fqp"] - e["Eabs"]) * e["dwp"]      # dFp/dp_avg
        wn, wp = e["wn"], e["wp"]
        dFn_dpsi_L = (1 - wn) * dE_L + gn * dn_dpsi[:-1] / 2
        dFn_dpsi_R = (1 - wn) * dE_R + gn * dn_dpsi[1:] / 2
        dFn_dphin_L = wn * (-sn / h) + gn * dn_dphin[:-1] / 2
        dFn_dphin_R = wn * (sn / h) + gn * dn_dphin[1:] / 2
        dFp_dpsi_L = (1 - wp) * dE_L + gp * dp_dpsi[:-1] / 2
        dFp_dpsi_R = (1 - wp) * dE_R + gp * dp_dpsi[1:] / 2
        dFp_dphip_L = wp * (-sp_ / h) + gp * dp_dphip[:-1] / 2
        dFp_dphip_R = wp * (sp_ / h) + gp * dp_dphip[1:] / 2

        an, ap = e["an"], e["ap"]
        sJn, sJp = np.sign(e["Jn"]), np.sign(e["Jp"])
        An, Ap = e["dan"] * np.abs(e["Jn"]), e["dap"] * np.abs(e["Jp"])
        dG_dpsi_L = (An * dFn_dpsi_L + an * sJn * dJn_dpsi_L + Ap * dFp_dpsi_L + ap * sJp * dJp_dpsi_L) / Q
        dG_dpsi_R = (An * dFn_dpsi_R + an * sJn * dJn_dpsi_R + Ap * dFp_dpsi_R + ap * sJp * dJp_dpsi_R) / Q
        dG_dphin_L = (An * dFn_dphin_L + an * sJn * dJn_dphin_L) / Q
        dG_dphin_R = (An * dFn_dphin_R + an * sJn * dJn_dphin_R) / Q
        dG_dphip_L = (Ap * dFp_dphip_L + ap * sJp * dJp_dphip_L) / Q
        dG_dphip_R = (Ap * dFp_dphip_R + ap * sJp * dJp_dphip_R) / Q

        idx = np.arange(1, N - 1)
        lo, hi = idx - 1, idx          # edge indices left/right of node idx
        w_lo, w_hi = hm / (2 * cv), hp / (2 * cv)   # box weights of each edge onto node idx
        # Node-integrated G_ii derivatives w.r.t. node idx-1 (m), idx (0), idx+1 (p)
        def node(dL, dR):
            return w_lo * dL[lo], w_lo * dR[lo] + w_hi * dL[hi], w_hi * dR[hi]
        Gpsi = node(dG_dpsi_L, dG_dpsi_R)
        Gphin = node(dG_dphin_L, dG_dphin_R)
        Gphip = node(dG_dphip_L, dG_dphip_R)

        dR_dn = (p * denom - num * mat.tau_p) / denom ** 2
        dR_dp = (n * denom - num * mat.tau_n) / denom ** 2
        dR_dpsi = dR_dn * dn_dpsi + dR_dp * dp_dpsi
        cs, ps_ = self.cont_scale, self.poisson_scale

        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        # Poisson
        add(idx, idx - 1, lap_m / ps_)
        add(idx, idx, -(lap_m + lap_p) / ps_ - Q * (dn_dpsi[idx] - dp_dpsi[idx]) / ps_)
        add(idx, idx + 1, lap_p / ps_)
        add(idx, N + idx, -Q * dn_dphin[idx] / ps_)
        add(idx, 2 * N + idx, Q * dp_dphip[idx] / ps_)

        # Electron continuity: (Jn[hi]-Jn[lo])/cv - Q*(R - G), /cs
        rn = N + idx
        add(rn, idx - 1, (-dJn_dpsi_L[lo] / cv + Q * Gpsi[0]) / cs)
        add(rn, idx, ((dJn_dpsi_L[hi] - dJn_dpsi_R[lo]) / cv - Q * dR_dpsi[idx] + Q * Gpsi[1]) / cs)
        add(rn, idx + 1, (dJn_dpsi_R[hi] / cv + Q * Gpsi[2]) / cs)
        add(rn, N + idx - 1, (-dJn_dphin_L[lo] / cv + Q * Gphin[0]) / cs)
        add(rn, N + idx, ((dJn_dphin_L[hi] - dJn_dphin_R[lo]) / cv - Q * dR_dn[idx] * dn_dphin[idx] + Q * Gphin[1]) / cs)
        add(rn, N + idx + 1, (dJn_dphin_R[hi] / cv + Q * Gphin[2]) / cs)
        add(rn, 2 * N + idx - 1, Q * Gphip[0] / cs)
        add(rn, 2 * N + idx, (-Q * dR_dp[idx] * dp_dphip[idx] + Q * Gphip[1]) / cs)
        add(rn, 2 * N + idx + 1, Q * Gphip[2] / cs)

        # Hole continuity: (Jp[hi]-Jp[lo])/cv + Q*(R - G), /cs
        rp = 2 * N + idx
        add(rp, idx - 1, (-dJp_dpsi_L[lo] / cv - Q * Gpsi[0]) / cs)
        add(rp, idx, ((dJp_dpsi_L[hi] - dJp_dpsi_R[lo]) / cv + Q * dR_dpsi[idx] - Q * Gpsi[1]) / cs)
        add(rp, idx + 1, (dJp_dpsi_R[hi] / cv - Q * Gpsi[2]) / cs)
        add(rp, 2 * N + idx - 1, (-dJp_dphip_L[lo] / cv - Q * Gphip[0]) / cs)
        add(rp, 2 * N + idx, ((dJp_dphip_L[hi] - dJp_dphip_R[lo]) / cv + Q * dR_dp[idx] * dp_dphip[idx] - Q * Gphip[1]) / cs)
        add(rp, 2 * N + idx + 1, (dJp_dphip_R[hi] / cv - Q * Gphip[2]) / cs)
        add(rp, N + idx - 1, -Q * Gphin[0] / cs)
        add(rp, N + idx, (Q * dR_dn[idx] * dn_dphin[idx] - Q * Gphin[1]) / cs)
        add(rp, N + idx + 1, -Q * Gphin[2] / cs)

        add(self.bc_rows, self.bc_rows, np.ones(len(self.bc_rows)))
        J = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(3 * N, 3 * N)).tocsc()
        return F, J

    def dF_dVa(self):
        d = np.zeros(3 * self.N)
        d[self.bc_rows] = -self.bc_dVa
        return d

    # ---- currents -------------------------------------------------------
    def edge_currents(self, U):
        psi, phin, phip, n, p = self.densities(U)
        e = self._edges(psi, phin, phip, n, p)
        return e["Jn"], e["Jp"]

    def edge_noise(self, U):
        psi, phin, phip, n, p = self.densities(U)
        return edge_current_noise(phin, phip, n, p, self.h, Q * self.mat.mu_n, Q * self.mat.mu_p)

    def terminal_current(self, U):
        """(J_rep, J_std, k_best) - total current read on its best-resolved edge."""
        Jn, Jp = self.edge_currents(U)
        J_rep, J_std, k, _ = resolved_current(Jn + Jp, self.edge_noise(U))
        return J_rep, J_std, k

    def edge_current_and_grad(self, U, k):
        """Total current on edge k and its gradient w.r.t. the 6 unknowns it
        depends on - the arc-length constraint's current measure."""
        N, mat, Vt, h = self.N, self.mat, self.mat.Vt, self.h[k]
        psi, phin, phip, n, p = self.densities(U)
        nn, pp = n[k:k + 2], p[k:k + 2]
        dfn, dfp = phin[k + 1] - phin[k], phip[k + 1] - phip[k]
        a, b = -Q * mat.mu_n / h, -Q * mat.mu_p / h
        J = a * nn.mean() * dfn + b * pp.mean() * dfp
        cols = np.array([k, k + 1, N + k, N + k + 1, 2 * N + k, 2 * N + k + 1])
        vals = np.array([
            a * nn[0] / Vt / 2 * dfn - b * pp[0] / Vt / 2 * dfp,
            a * nn[1] / Vt / 2 * dfn - b * pp[1] / Vt / 2 * dfp,
            a * (-nn[0] / Vt / 2 * dfn - nn.mean()),
            a * (-nn[1] / Vt / 2 * dfn + nn.mean()),
            b * (pp[0] / Vt / 2 * dfp - pp.mean()),
            b * (pp[1] / Vt / 2 * dfp + pp.mean()),
        ])
        return J, cols, vals

    def step_clip(self, delta):
        return uniform_step_clip(delta, self.N, max_psi=_MAX_PSI_STEP, max_qf=_MAX_QF_STEP)

    def trial_clip(self, delta):
        return componentwise_step_clip(delta, self.N, max_psi=_MAX_PSI_STEP, Vt=self.mat.Vt)

    def solve(self, U0, Va, tol=ROW_TOL, maxiter=50, verbose=False):
        """Voltage-controlled solve at Va from U0. Returns (U, merit, it, converged)."""
        return damped_newton(self.apply_bc(U0, Va),
                             lambda U: self.residual_and_jacobian(U, Va),
                             lambda U: self.residual(U, Va),
                             step_clip=self.step_clip, trial_clip=self.trial_clip, tol=tol, maxiter=maxiter,
                             verbose=verbose, label="Newton(avalanche)")

    def result_dict(self, U, it, converged):
        N = self.N
        psi, phin, phip, n, p = self.densities(U)
        Jn, Jp = self.edge_currents(U)
        J_rep, J_std, _ = self.terminal_current(U)
        return {"psi": psi.copy(), "n": n, "p": p, "phin": phin.copy(), "phip": phip.copy(),
                "Jn": Jn, "Jp": Jp, "Jtot": Jn + Jp, "iters": it,
                "J_mean": J_rep, "J_std": J_std, "converged": converged}


def newton_gummel_solve(x, Cdop, mat: Material, Va, psi_eq, n_eq, p_eq,
                         psi_init=None, phin_init=None, phip_init=None,
                         f_tol=ROW_TOL, maxiter=50, verbose=False,
                         ii_model: AvalancheModel = None, driving_force="hybrid",
                         ref_density_cm3=DEFAULT_REF_DENSITY_CM3):
    """Voltage-controlled avalanche solve at one bias; same signature/return
    shape as newton_solver_qf.newton_gummel_solve (plus "converged").

    Starting point: the previous bias point's solution when given (normal
    continuation); otherwise - or if that attempt does not converge - the
    converged NO-avalanche QF solution at this same Va (get the transport
    right first, then switch on the generation feedback; a cold Gummel
    start's few-mV psi noise on a sub-nm junction mesh is an unphysical
    local field that alpha() amplifies enormously).

    Near breakdown the I(Va) curve is nearly vertical and a voltage-
    controlled step can overshoot it - use main_avalanche.py's arc-length
    trace (core/arclength.py) to follow the curve through the knee."""
    prob = AvalancheProblem(x, Cdop, mat, psi_eq, ii_model, driving_force, ref_density_cm3)

    def from_qf():
        from core.newton_solver_qf import newton_gummel_solve as qf_solve
        base = qf_solve(x, Cdop, mat, Va, psi_eq, n_eq, p_eq, maxiter=maxiter)
        return np.concatenate([base["psi"], base["phin"], base["phip"]])

    converged = False
    if psi_init is not None:
        U, merit, it, converged = prob.solve(np.concatenate([psi_init, phin_init, phip_init]),
                                             Va, f_tol, maxiter, verbose)
    if not converged:
        U2, merit2, it2, conv2 = prob.solve(from_qf(), Va, f_tol, maxiter, verbose)
        if psi_init is None or conv2 or merit2 < merit:
            U, merit, it, converged = U2, merit2, it2, conv2
    if not converged:
        warnings.warn(
            f"Newton(avalanche) solve did not converge at Va={Va} V (max|F/d|={merit:.3e} V "
            f"at iteration {it}) - check this point's self-consistency (J_std/J_mean).")
    return prob.result_dict(U, it, converged)


def trace_breakdown(x, Cdop, mat: Material, ii_model: AvalancheModel = None,
                    driving_force="hybrid", ref_density_cm3=DEFAULT_REF_DENSITY_CM3,
                    seed_V=-1.0, J_stop_A_cm2=1e3, V_limit=None, ds_max=1.0,
                    max_points=500, verbose=False):
    """Trace the reverse-bias I-V curve from near equilibrium, through the
    avalanche knee, up to |J| = J_stop_A_cm2 (A/cm^2).

    Voltage-controlled steps (bisected on failure) walk from 0 V to seed_V;
    core/arclength.trace_iv then follows the curve in the (Va, ln|J|) plane,
    which needs no bias schedule and no knowledge of where breakdown is.

    Returns dict(Va, J, iters, U, status, rejections, problem, psi_eq,
    n_eq, p_eq, seed_Va, seed_J) - Va/J cover the seeds plus the trace."""
    from core.solver import solve_equilibrium
    from core.arclength import trace_iv

    psi_eq, n_eq, p_eq, _ = solve_equilibrium(x, Cdop, mat)
    prob = AvalancheProblem(x, Cdop, mat, psi_eq, ii_model, driving_force, ref_density_cm3)
    Vt = mat.Vt
    U = np.concatenate([psi_eq, psi_eq - Vt * np.log(n_eq / mat.ni), psi_eq + Vt * np.log(p_eq / mat.ni)])

    seeds, V, dV = [], 0.0, 0.1
    while V > seed_V + 1e-12:
        V_try = max(V - dV, seed_V)
        U_try, merit, it, ok = prob.solve(U, V_try, verbose=verbose)
        if not ok:
            dV *= 0.5
            if dV < 1e-4:
                raise RuntimeError(f"trace_breakdown: could not seed the trace near Va={V_try:.4f} V "
                                   f"(max|F/d|={merit:.2e})")
            continue
        U, V = U_try, V_try
        seeds.append((U.copy(), V))
        dV = min(1.5 * dV, 0.25)

    tr = trace_iv(prob.residual_and_jacobian, prob.residual, lambda U, Va: prob.dF_dVa(),
                  prob.edge_current_and_grad, lambda U: int(np.argmin(prob.edge_noise(U))),
                  seeds[-2:], step_clip=prob.step_clip, trial_clip=prob.trial_clip, ds_max=ds_max, J_stop=J_stop_A_cm2,
                  V_limit=V_limit, max_points=max_points, verbose=verbose)

    seed_Va = np.array([s[1] for s in seeds])
    seed_J = np.array([prob.terminal_current(s[0])[0] for s in seeds])
    return dict(Va=np.concatenate([seed_Va, tr["Va"]]), J=np.concatenate([seed_J, tr["J"]]),
                iters=tr["iters"], U=[s[0] for s in seeds] + tr["U"], status=tr["status"],
                rejections=tr["rejections"], problem=prob, psi_eq=psi_eq, n_eq=n_eq, p_eq=p_eq,
                seed_Va=seed_Va, seed_J=seed_J)
