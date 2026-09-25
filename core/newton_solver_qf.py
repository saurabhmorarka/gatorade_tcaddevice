"""Fully coupled Newton solve using QUASI-FERMI POTENTIALS (phin, phip) as
the transport unknowns, instead of raw carrier densities (newton_solver.py)
or log-densities (the abandoned log-density-formulation branch).

Same overall structure as newton_solver.py (analytic sparse Jacobian, direct
sparse LU per step, backtracking line search, Gummel cold-start) - only the
CONTINUITY discretization and the choice of unknowns differ. See that
module's docstring for the shared design notes (why an analytic Jacobian +
direct solve, not Newton-Krylov).

WHY THIS EXISTS: newton_solver.py (raw n, p + Scharfetter-Gummel) fails to
converge robustly across reverse bias for a strongly asymmetric, degenerately
doped junction (p-side 1e17 / n-side 1e20-1e21 cm^-3 - see the pinned
"tcad1d-extreme-doping-convergence-limit" note). Two log-density
reformulations were tried (on the now-abandoned log-density-formulation
branch) and both failed - they kept Scharfetter-Gummel's own exponential
(Bernoulli-function) flux fitting, just reparametrized the unknowns, so the
Jacobian still carried TWO compounding exponential nonlinearities (SG's own,
plus the density-vs-potential relation).

This formulation, modeled directly on the open-source FLOOXS device
simulator's "QF" model (TclLib/Device/floods/Silicon/Equations.tcl,
Generic/Transport/continuity.tcl - FLOOXS source available locally at
~/Desktop/github_flooxs/flooxs), removes SG entirely. The unknowns are the
electron/hole quasi-Fermi potentials phin, phip - already a quantity this
codebase understands (physics.py's solve_poisson already takes phin/phip as
FIXED inputs for the MOS quasi-small-signal trick; the raw-density Newton
solver already DERIVES phin/phip post-solve, phin = psi - Vt*ln(n/ni)) - so
this promotes an existing derived quantity to a primary unknown rather than
importing a foreign concept. The current is a PLAIN gradient of the
quasi-Fermi potential (no Bernoulli function at all):

    Jn = -q * mu_n * n * grad(phin)     Jp = -q * mu_p * p * grad(phip)

(standard textbook drift-diffusion current in quasi-Fermi form - derivable
directly from Jn = q*Dn*grad(n) - q*mu_n*n*grad(psi) via the Einstein
relation and n = ni*exp((psi-phin)/Vt): grad(n) = (n/Vt)*(grad(psi)-grad(phin)),
so Jn = q*mu_n*n*(grad(psi)-grad(phin)) - q*mu_n*n*grad(psi) = -q*mu_n*n*grad(phin)).
Discretized here with the edge's carrier density taken as the arithmetic
mean of its two nodal values (the simplest standard central finite-volume
choice - if validation ever shows oscillation on a coarse mesh, a geometric
mean is the documented fallback, but this project's meshes are already fine
enough at every junction to resolve exponential variation directly, which is
the whole reason SG's exponential fitting is not needed here).

Positivity is structural (n, p = ni*exp(...) can never go negative), so
there is no density-floor clip anywhere in this module.
"""
import warnings

import numpy as np
import scipy.sparse as sp

from core.params import Q, Material
from core import physics as ph
from core.materials import MaterialField
from core.newton_numerics import (ROW_TOL, damped_newton, uniform_step_clip, componentwise_step_clip,
                                  edge_current_noise, resolved_current)
from core.solver import contact_values

# Public building blocks shared with other QF-based solvers (currently
# tat/newton_solver_tat.py) - see ARCHITECTURE.md's "reusable pure-physics
# kernel" / shared-QF-machinery notes. avalanche/newton_solver_avalanche.py
# deliberately keeps its own private copies instead of importing these -
# it is meant to stay standalone, not coupled to this module.
__all__ = ["poisson_row_scale", "continuity_row_scale", "unpack_qf", "MAX_QF_STEP",
           "newton_gummel_solve"]


def poisson_row_scale(mat: Material, h_typ: float) -> float:
    """mat may be a plain scalar Material or a MaterialField (heterojunction)
    - a mean over the eps_edge array is used as the representative scale
    (this only affects Newton's row-scaling/conditioning, not accuracy, so
    an approximate scalar is fine; reduces to mat.eps exactly for a
    homojunction MaterialField.uniform(), i.e. bit-identical to before)."""
    eps = mat.eps if not isinstance(mat, MaterialField) else float(np.mean(mat.eps_edge))
    return eps * mat.Vt / h_typ ** 2


def continuity_row_scale(mat: Material, h_typ: float) -> float:
    if isinstance(mat, MaterialField):
        Dn = float(np.mean(mat.Dn_edge))
        ni = float(np.mean(mat.ni_arr))
    else:
        Dn, ni = mat.Dn, mat.ni
    return Q * Dn * ni / h_typ


def unpack_qf(U, N):
    """U = [psi, phin, phip]."""
    return U[:N], U[N:2 * N], U[2 * N:3 * N]


def _edge_quantities(psi, phin, phip, n, p, x, mat):
    """Per-edge plain-gradient flux and recombination - shared by the
    residual-only and residual+Jacobian paths so they never disagree. mat
    must already be a MaterialField (callers normalize once via
    core.materials.MaterialField.uniform() for a plain scalar Material, see
    newton_gummel_solve) - array reads here are bit-identical to the old
    scalar mat.mu_n/mat.tau_n/mat.ni reads for a homojunction."""
    h = np.diff(x)
    n_avg = (n[:-1] + n[1:]) / 2.0
    p_avg = (p[:-1] + p[1:]) / 2.0

    Jn = -Q * mat.mu_n_edge * n_avg * (phin[1:] - phin[:-1]) / h
    Jp = -Q * mat.mu_p_edge * p_avg * (phip[1:] - phip[:-1]) / h

    ni = mat.ni_arr
    denom = mat.tau_p_arr * (n + ni) + mat.tau_n_arr * (p + ni)
    num = n * p - ni ** 2
    R = num / denom
    return h, n_avg, p_avg, Jn, Jp, R, denom, num


def _residual_only(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale):
    """Fast path: residual vector only, no Jacobian. Used for line-search
    trial evaluations, which don't need a new Jacobian until a step is
    accepted. mat must already be a MaterialField (see newton_gummel_solve,
    which normalizes a plain scalar Material once via MaterialField.uniform())."""
    N = len(x)
    psi, phin, phip = unpack_qf(U, N)
    Vt = mat.Vt
    n = mat.ni_arr * np.exp((psi - phin + mat.delta_Ei_arr) / Vt)
    p = mat.ni_arr * np.exp((phip - psi - mat.delta_Ei_arr) / Vt)
    cvol = ph._control_volumes(x)

    Rpsi = np.empty(N)
    Rn = np.empty(N)
    Rp = np.empty(N)
    Rpsi[0], Rpsi[-1] = psi[0] - psi_bc[0], psi[-1] - psi_bc[-1]
    Rn[0], Rn[-1] = phin[0] - phin_bc[0], phin[-1] - phin_bc[-1]
    Rp[0], Rp[-1] = phip[0] - phip_bc[0], phip[-1] - phip_bc[-1]

    h = np.diff(x)
    hm, hp = h[:-1], h[1:]
    cvol_i = cvol[1:-1]
    lap_m = mat.eps_edge[:-1] / hm / cvol_i
    lap_p = mat.eps_edge[1:] / hp / cvol_i
    Rpsi[1:-1] = (lap_p * (psi[2:] - psi[1:-1]) - lap_m * (psi[1:-1] - psi[:-2])
                  - Q * (n[1:-1] - p[1:-1] - Cdop[1:-1])) / poisson_scale

    _, _, _, Jn, Jp, R, _, _ = _edge_quantities(psi, phin, phip, n, p, x, mat)
    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * R[1:-1]) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * R[1:-1]) / cont_scale

    return np.concatenate([Rpsi, Rn, Rp])


def _residual_and_jacobian(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale):
    """Returns (F, J) where F is the length-3N residual vector and J is the
    3N x 3N sparse Jacobian dF/dU, for unknowns U=[psi, phin, phip]. Fully
    vectorized (no per-node Python loop). mat must already be a
    MaterialField (see newton_gummel_solve)."""
    N = len(x)
    psi, phin, phip = unpack_qf(U, N)
    Vt = mat.Vt
    n = mat.ni_arr * np.exp((psi - phin + mat.delta_Ei_arr) / Vt)
    p = mat.ni_arr * np.exp((phip - psi - mat.delta_Ei_arr) / Vt)
    cvol = ph._control_volumes(x)

    Rpsi = np.empty(N)
    Rn = np.empty(N)
    Rp = np.empty(N)
    Rpsi[0], Rpsi[-1] = psi[0] - psi_bc[0], psi[-1] - psi_bc[-1]
    Rn[0], Rn[-1] = phin[0] - phin_bc[0], phin[-1] - phin_bc[-1]
    Rp[0], Rp[-1] = phip[0] - phip_bc[0], phip[-1] - phip_bc[-1]

    h = np.diff(x)
    hm, hp = h[:-1], h[1:]
    cvol_i = cvol[1:-1]
    lap_m = mat.eps_edge[:-1] / hm / cvol_i
    lap_p = mat.eps_edge[1:] / hp / cvol_i
    Rpsi[1:-1] = (lap_p * (psi[2:] - psi[1:-1]) - lap_m * (psi[1:-1] - psi[:-2])
                  - Q * (n[1:-1] - p[1:-1] - Cdop[1:-1])) / poisson_scale

    h_e, n_avg, p_avg, Jn, Jp, R, denom, num = _edge_quantities(psi, phin, phip, n, p, x, mat)

    # dn/dpsi = n/Vt, dn/dphin = -n/Vt ; dp/dpsi = -p/Vt, dp/dphip = p/Vt
    # (delta_Ei doesn't depend on any unknown, so these derivatives are
    # exactly unchanged from the homojunction formula - see the harmonic-
    # snuggling-puddle plan's physics section)
    dn_dpsi = n / Vt
    dn_dphin = -n / Vt
    dp_dpsi = -p / Vt
    dp_dphip = p / Vt

    dphin_e = phin[1:] - phin[:-1]
    dphip_e = phip[1:] - phip[:-1]

    # Jn_e = -Q*mu_n_edge*n_avg*dphin_e/h ; n_avg = (n_i+n_{i+1})/2
    dJn_dpsi_e = -Q * mat.mu_n_edge / h_e * (dn_dpsi[:-1] / 2.0) * dphin_e
    dJn_dpsi_ep1 = -Q * mat.mu_n_edge / h_e * (dn_dpsi[1:] / 2.0) * dphin_e
    dJn_dphin_e = -Q * mat.mu_n_edge / h_e * ((dn_dphin[:-1] / 2.0) * dphin_e - n_avg)
    dJn_dphin_ep1 = -Q * mat.mu_n_edge / h_e * ((dn_dphin[1:] / 2.0) * dphin_e + n_avg)

    # Jp_e = -Q*mu_p_edge*p_avg*dphip_e/h
    dJp_dpsi_e = -Q * mat.mu_p_edge / h_e * (dp_dpsi[:-1] / 2.0) * dphip_e
    dJp_dpsi_ep1 = -Q * mat.mu_p_edge / h_e * (dp_dpsi[1:] / 2.0) * dphip_e
    dJp_dphip_e = -Q * mat.mu_p_edge / h_e * ((dp_dphip[:-1] / 2.0) * dphip_e - p_avg)
    dJp_dphip_ep1 = -Q * mat.mu_p_edge / h_e * ((dp_dphip[1:] / 2.0) * dphip_e + p_avg)

    dR_dn = (p * denom - num * mat.tau_p_arr) / denom ** 2
    dR_dp = (n * denom - num * mat.tau_n_arr) / denom ** 2

    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * R[1:-1]) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * R[1:-1]) / cont_scale
    F = np.concatenate([Rpsi, Rn, Rp])

    idx = np.arange(1, N - 1)
    k = idx - 1
    e_lo, e_hi = k, k + 1
    cv = cvol_i

    rows_list, cols_list, data_list = [], [], []

    def add(r, c, v):
        rows_list.append(r)
        cols_list.append(c)
        data_list.append(v)

    # Poisson interior rows: linear in psi, and dF_psi/dphin = dF_psi/dn*dn/dphin etc.
    add(idx, idx - 1, lap_m / poisson_scale)
    add(idx, idx, -(lap_m + lap_p) / poisson_scale
        - Q * (dn_dpsi[idx] - dp_dpsi[idx]) / poisson_scale)
    add(idx, idx + 1, lap_p / poisson_scale)
    add(idx, N + idx, (-Q / poisson_scale) * dn_dphin[idx])
    add(idx, 2 * N + idx, (-Q / poisson_scale) * (-dp_dphip[idx]))

    # Electron continuity interior rows. Row i = (Jn[e_hi] - Jn[e_lo])/cvol -
    # Q*R, all over cont_scale; e_lo connects (i-1,i), e_hi connects (i,i+1).
    # dJn_*_e is d(Jn at that edge)/d(unknown at the edge's LEFT node), so
    # e.g. column i-1 only sees Jn[e_lo] through its own left-node
    # derivative, with the row's own leading minus sign on Jn[e_lo].
    r_n = N + idx
    add(r_n, idx - 1, -dJn_dpsi_e[e_lo] / cv / cont_scale)
    add(r_n, idx, (dJn_dpsi_e[e_hi] - dJn_dpsi_ep1[e_lo]) / cv / cont_scale
        - Q * (dR_dn[idx] * dn_dpsi[idx] + dR_dp[idx] * dp_dpsi[idx]) / cont_scale)
    add(r_n, idx + 1, dJn_dpsi_ep1[e_hi] / cv / cont_scale)
    add(r_n, N + idx - 1, -dJn_dphin_e[e_lo] / cv / cont_scale)
    add(r_n, N + idx, (dJn_dphin_e[e_hi] - dJn_dphin_ep1[e_lo]) / cv / cont_scale
        - Q * dR_dn[idx] * dn_dphin[idx] / cont_scale)
    add(r_n, N + idx + 1, dJn_dphin_ep1[e_hi] / cv / cont_scale)
    add(r_n, 2 * N + idx, -Q * dR_dp[idx] * dp_dphip[idx] / cont_scale)

    # Hole continuity interior rows - same pattern, +Q*R sign (opposite
    # electron's -Q*R, same convention as newton_solver.py).
    r_p = 2 * N + idx
    add(r_p, idx - 1, -dJp_dpsi_e[e_lo] / cv / cont_scale)
    add(r_p, idx, (dJp_dpsi_e[e_hi] - dJp_dpsi_ep1[e_lo]) / cv / cont_scale
        + Q * (dR_dn[idx] * dn_dpsi[idx] + dR_dp[idx] * dp_dpsi[idx]) / cont_scale)
    add(r_p, idx + 1, dJp_dpsi_ep1[e_hi] / cv / cont_scale)
    add(r_p, 2 * N + idx - 1, -dJp_dphip_e[e_lo] / cv / cont_scale)
    add(r_p, 2 * N + idx, (dJp_dphip_e[e_hi] - dJp_dphip_ep1[e_lo]) / cv / cont_scale
        + Q * dR_dp[idx] * dp_dphip[idx] / cont_scale)
    add(r_p, 2 * N + idx + 1, dJp_dphip_ep1[e_hi] / cv / cont_scale)
    add(r_p, N + idx, Q * dR_dn[idx] * dn_dphin[idx] / cont_scale)

    interior_rows = np.concatenate(rows_list)
    interior_cols = np.concatenate(cols_list)
    interior_data = np.concatenate(data_list)

    # Dirichlet rows - same pivoting-safety scaling as newton_solver.py (see
    # that module's comment for why a bare diag=1.0 isn't always safe).
    dirichlet_idx = [0, N - 1, N + 0, N + N - 1, 2 * N + 0, 2 * N + N - 1]
    dirichlet_rows, dirichlet_cols, dirichlet_data = [], [], []
    for i in dirichlet_idx:
        col_mask = interior_cols == i
        local_max = np.max(np.abs(interior_data[col_mask])) if np.any(col_mask) else 0.0
        dirichlet_rows.append(i)
        dirichlet_cols.append(i)
        dirichlet_data.append(max(1.0, local_max))

    rows = np.concatenate([interior_rows, dirichlet_rows])
    cols = np.concatenate([interior_cols, dirichlet_cols])
    data = np.concatenate([interior_data, dirichlet_data])
    J = sp.coo_matrix((data, (rows, cols)), shape=(3 * N, 3 * N)).tocsc()
    return F, J


# Cap on a single Newton step's raw |delta_phin|, |delta_phip| (volts) - see
# the comment at its use site for why this is needed (a near-zero carrier
# density leaves its quasi-Fermi potential almost unconstrained by the
# residual, letting a raw step send it hundreds of volts off in one shot).
MAX_QF_STEP = 5.0


def newton_gummel_solve(x, Cdop, mat: Material, Va, psi_eq, n_eq, p_eq,
                         psi_init=None, phin_init=None, phip_init=None,
                         f_tol=ROW_TOL, maxiter=50, verbose=False):
    """Same signature/return shape as solver.gummel_solve and
    newton_solver.newton_gummel_solve, but solves for quasi-Fermi potentials
    (phin, phip) instead of raw densities - see module docstring.

    mat may be a plain scalar Material (today's exact behavior) or a
    core.materials.MaterialField (heterojunction) - normalized to a
    MaterialField once here (`mf`), then used everywhere below and in every
    helper this function calls (_residual_only/_residual_and_jacobian
    require a MaterialField, per their own docstrings)."""
    N = len(x)
    mf = mat if isinstance(mat, MaterialField) else MaterialField.uniform(mat, x)
    ni0, niL = mf.ni_arr[0], mf.ni_arr[-1]
    dEi0, dEiL = mf.delta_Ei_arr[0], mf.delta_Ei_arr[-1]
    n_bc0, p_bc0 = contact_values(mf, Cdop[0], ni=ni0)
    n_bcL, p_bcL = contact_values(mf, Cdop[-1], ni=niL)
    Vt = mf.Vt

    # phin = psi + delta_Ei - Vt*ln(n/ni), phip = psi + delta_Ei + Vt*ln(p/ni)
    # (inverting n=ni*exp((psi-phin+delta_Ei)/Vt), p=ni*exp((phip-psi-delta_Ei)/Vt)
    # at each contact - delta_Ei=0 for a homojunction, reducing to today's
    # exact formula bit-for-bit; verified by hand this gives phin_bc=phip_bc=Va
    # at the biased contact and 0 at the grounded one regardless of which
    # material each contact sits in, see the harmonic-snuggling-puddle plan).
    psi_bc = np.array([psi_eq[0] + Va, psi_eq[-1]])
    phin_bc = np.array([psi_bc[0] - Vt * np.log(n_bc0 / ni0) + dEi0,
                         psi_bc[-1] - Vt * np.log(n_bcL / niL) + dEiL])
    phip_bc = np.array([psi_bc[0] + Vt * np.log(p_bc0 / ni0) + dEi0,
                         psi_bc[-1] + Vt * np.log(p_bcL / niL) + dEiL])

    h_typ = np.min(np.diff(x))
    poisson_scale = poisson_row_scale(mf, h_typ)
    cont_scale = continuity_row_scale(mf, h_typ)

    def _gummel_start():
        from core.solver import gummel_solve
        warm = gummel_solve(x, Cdop, mf, Va, psi_eq, n_eq, p_eq, max_gummel=15)
        return (warm["psi"].copy(),
                warm["psi"] + mf.delta_Ei_arr - Vt * np.log(warm["n"] / mf.ni_arr),
                warm["psi"] + mf.delta_Ei_arr + Vt * np.log(warm["p"] / mf.ni_arr))

    def _run_newton(psi0, phin0, phip0):
        """One full Newton attempt from a given starting point. Returns
        (U, merit, it, converged) - see core/newton_numerics.py for the
        row-normalized merit (and why the old raw-residual max-norm test
        stalled on roundoff in heavily doped majority-carrier rows)."""
        psi0 = psi0.copy(); phin0 = phin0.copy(); phip0 = phip0.copy()
        psi0[0], psi0[-1] = psi_bc
        phin0[0], phin0[-1] = phin_bc
        phip0[0], phip0[-1] = phip_bc
        U0 = np.concatenate([psi0, phin0, phip0])
        # Ruiz row/column equilibration happens inside damped_newton's
        # linear solve (core/jacobian_scaling.py) - needed for real
        # heterojunctions, where a substantially different ni (e.g. SiGe's
        # ~150x larger ni) spreads the dJn/dphin entries over many more
        # orders of magnitude than any homojunction.
        #
        # The step is shortened by ONE scalar so no phin/phip entry moves
        # more than MAX_QF_STEP: wherever a carrier's density is near zero
        # (electrons deep in the p-side bulk, say) its quasi-Fermi potential
        # barely affects the residual and a raw step could send it hundreds
        # of volts off (seen: phip swinging to -612V from an equilibrium
        # warm start). A uniform rescale keeps the Newton direction, which
        # the line search's descent guarantee relies on.
        return damped_newton(
            U0,
            lambda U: _residual_and_jacobian(U, x, Cdop, mf, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale),
            lambda U: _residual_only(U, x, Cdop, mf, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale),
            step_clip=lambda d: uniform_step_clip(d, N, max_psi=1.0, max_qf=MAX_QF_STEP),
            trial_clip=lambda d: componentwise_step_clip(d, N, max_psi=1.0, Vt=Vt),
            tol=f_tol, maxiter=maxiter, verbose=verbose, label="Newton(QF)")

    if psi_init is None:
        U, merit, it, converged = _run_newton(*_gummel_start())
    else:
        # phin_init/phip_init are already this solver's own native unknowns
        # (or, if warm-starting from the raw-density Newton solver, are
        # already provided in exactly this phin/phip convention too - both
        # solvers derive/use the identical phin = psi - Vt*ln(n/ni) relation).
        U, merit, it, converged = _run_newton(psi_init, phin_init, phip_init)
        if not converged:
            # A warm start can inherit a structurally degenerate Jacobian
            # from its source point - most notably right after equilibrium
            # (Va=0), where phin=phip=psi_eq is exactly FLAT across the
            # interior, making every edge's flux-vs-psi coupling (which is
            # proportional to that edge's own delta-phin) vanish identically.
            # Retry from a fresh Gummel-derived start, whose decoupled
            # iteration doesn't share that failure mode.
            U_r, merit_r, it_r, conv_r = _run_newton(*_gummel_start())
            if conv_r or merit_r < merit:
                U, merit, it, converged = U_r, merit_r, it_r, conv_r

    if not converged:
        warnings.warn(
            f"Newton(QF) solve did not converge at Va={Va} V "
            f"(max|F/d|={merit:.3e} V at iteration {it}) even after a Gummel-restart retry - "
            "check this point's self-consistency (J_std/J_mean) before trusting it.")

    psi, phin, phip = unpack_qf(U, N)
    n = mf.ni_arr * np.exp((psi - phin + mf.delta_Ei_arr) / Vt)
    p = mf.ni_arr * np.exp((phip - psi - mf.delta_Ei_arr) / Vt)

    # Report currents from THIS solver's own plain-gradient flux (not
    # physics.edge_currents' Scharfetter-Gummel formula), and read the
    # terminal current only where it is numerically resolved - in a heavily
    # doped majority region the edge current is roundoff (see
    # core/newton_numerics.resolved_current).
    h_e, _, _, Jn, Jp, _, _, _ = _edge_quantities(psi, phin, phip, n, p, x, mf)
    Jtot = Jn + Jp
    noise = edge_current_noise(phin, phip, n, p, h_e, Q * mf.mu_n_edge, Q * mf.mu_p_edge)
    J_rep, J_std, _, _ = resolved_current(Jtot, noise)

    return {
        "psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
        "Jn": Jn, "Jp": Jp, "Jtot": Jtot, "iters": it,
        "J_mean": J_rep, "J_std": J_std, "converged": converged,
    }
