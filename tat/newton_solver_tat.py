"""Fully coupled QF Newton solve (core.newton_solver_qf.py's formulation)
extended with reverse-bias junction leakage: Kane band-to-band tunneling
(purely additive generation) and Hurkx trap-assisted tunneling (a
field-enhancement of the SRH recombination/generation term this project
already implements). See tat/tat.py's module docstring for the physics and
plans/tat_btbt_plan.md for the full design rationale.

New module rather than a flag threaded into newton_solver_qf.py - matches
this project's own precedent (newton_solver_qf.py itself was added
alongside newton_solver.py, avalanche/newton_solver_avalanche.py alongside
that) - keeps the plain QF solve's code and behavior byte-for-byte
untouched.

UNLIKE avalanche/newton_solver_avalanche.py, this solver needs no
arc-length continuation. Avalanche's G_ii depends on |Jn|, |Jp| - the very
quantities the continuity equations solve for - a positive feedback (more
current -> more generation -> more current) that makes I(Va) near-vertical
at breakdown. Both G_btbt and G_tat here depend only on the local field and
local densities n, p - never on Jn/Jp - so there is no such self-reinforcing
loop, and an ordinary voltage-controlled sweep with the shared damped Newton
(core/newton_numerics.py, same as newton_solver_qf.py) suffices; this is
checked, not merely assumed (see the finite-difference Jacobian check and
the example sweep's own self-consistency in main_tat.py).

Both generation terms plug in like a REPLACEMENT of the existing R term
in newton_solver_qf.py's continuity rows, not an addition alongside it:
the trap-assisted term R_trap(n, p, F) - Hurkx (default) or Schenk, see
tat/tat.py, selected via `trap_generation_fn`/`trap_model` - IS the
(field-enhanced) SRH-like term, collapsing exactly (Hurkx) or closely
(Schenk) onto ordinary SRH behavior at F=0, so it takes over that term's
role rather than sitting beside it. Kane's G_btbt is a genuinely separate
mechanism (no SRH trap at all - direct electron-hole-pair creation) and is
purely additive, subtracted from the effective recombination rate the
same way avalanche/avalanche.py's G_ii is:

    Reff(n, p, psi) = R_trap(n, p, F_node(psi)) - G_btbt(F_node(psi))
    Rn[1:-1] = (dJn/cvol - Q*Reff) / cont_scale   (replaces plain R)
    Rp[1:-1] = (dJp/cvol + Q*Reff) / cont_scale
"""
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from core.params import Q, KB, Material
from core import physics as ph
from core.materials import MaterialField
from core.newton_numerics import (ROW_TOL, damped_newton, uniform_step_clip, componentwise_step_clip,
                                  edge_current_noise, resolved_current)
from core.solver import contact_values
from core.newton_solver_qf import poisson_row_scale, continuity_row_scale, unpack_qf, MAX_QF_STEP
from tat.tat import KaneBTBTModel, HurkxTATModel, btbt_generation, hurkx_tat_generation

# Default trap-assisted generation function - swappable per plans/tat_btbt_plan.md's
# "Hurkx first, Schenk second, same call signature" design (see tat.tat's
# hurkx_tat_generation/schenk_tat_generation docstrings). Passed as
# `trap_generation_fn` through _reff_and_derivs -> _residual_only /
# _residual_and_jacobian -> newton_gummel_solve, paired with whichever
# model object (HurkxTATModel or SchenkTATModel) matches it.
_DEFAULT_TRAP_GENERATION_FN = hurkx_tat_generation


def _node_field(psi, x):
    """Edge field E_e = -(psi[1:]-psi[:-1])/h (this project's existing sign
    convention, matches avalanche/avalanche's own doc), and a length-
    weighted average of the two edges flanking each interior node onto that
    node - same box-averaging style as avalanche's Gii_node, but averaging
    the field value itself rather than integrating a flux-divergence-like
    quantity, since G_btbt/G_tat depend on nodal n, p directly (like
    srh_recombination already does), not on edge-averaged currents.

    Returns (E_e, Eabs_e, F_node, dF_node_dpsi_im1, dF_node_dpsi_i,
    dF_node_dpsi_ip1) - the last three each length N-2 (interior nodes
    1..N-2), giving d(F_node[i])/d(psi at node i-1, i, i+1) respectively.
    """
    h = np.diff(x)
    E_e = -(psi[1:] - psi[:-1]) / h
    Eabs_e = np.abs(E_e)
    sgn_e = np.sign(E_e)

    hm, hp = h[:-1], h[1:]           # h[i-1], h[i] for interior node i
    e_lo, e_hi = slice(0, -1), slice(1, None)  # edge indices (i-1,i) and (i,i+1)
    Eabs_lo, Eabs_hi = Eabs_e[e_lo], Eabs_e[e_hi]
    sgn_lo, sgn_hi = sgn_e[e_lo], sgn_e[e_hi]

    denom = hm + hp
    F_node = (hm * Eabs_lo + hp * Eabs_hi) / denom

    dF_dpsi_im1 = sgn_lo / denom
    dF_dpsi_i = (sgn_hi - sgn_lo) / denom
    dF_dpsi_ip1 = -sgn_hi / denom

    return F_node, dF_dpsi_im1, dF_dpsi_i, dF_dpsi_ip1


class _NodeMaterialView:
    """Lightweight per-node material view exposing the SAME attribute names
    tat.tat's generation functions read as plain scalars (mat.ni, mat.Vt,
    mat.T, mat.tau_n, mat.tau_p) - built from a MaterialField's arrays so
    hurkx_tat_generation/schenk_tat_generation work UNCHANGED (duck-typed,
    no edits to tat/tat.py needed) whether the device is a homojunction
    (arrays are a repeated constant - elementwise arithmetic bit-identical
    to the old scalar path) or a real heterojunction (arrays genuinely vary
    node to node, e.g. across a SiGe/Si junction, so trap-assisted
    generation correctly sees the LOCAL ni/tau at each interior node)."""
    def __init__(self, ni, Vt, T, tau_n, tau_p):
        self.ni, self.Vt, self.T, self.tau_n, self.tau_p = ni, Vt, T, tau_n, tau_p


def _interior_node_view(mf: MaterialField) -> _NodeMaterialView:
    """mf's arrays restricted to interior nodes (1..N-2), matching the
    shape of n_i, p_i, F_node in _reff_and_derivs below. T is uniform
    across the device (MaterialField only carries one Vt), recovered from
    Vt=KB*T/Q."""
    return _NodeMaterialView(
        ni=mf.ni_arr[1:-1], Vt=mf.Vt, T=mf.Vt * Q / KB,
        tau_n=mf.tau_n_arr[1:-1], tau_p=mf.tau_p_arr[1:-1],
    )


def _reff_and_derivs(n, p, psi, x, mat, kane_model, trap_model, trap_generation_fn):
    """Interior-node (length N-2) effective recombination rate
    Reff = R_trap - G_btbt and its partial derivatives w.r.t. n[i], p[i]
    (local, diagonal) and psi[i-1], psi[i], psi[i+1] (via the field).

    trap_generation_fn is either hurkx_tat_generation or
    schenk_tat_generation (tat.tat) - both share the same
    (n, p, F_abs, mat, model) -> (G, dG_dn, dG_dp, dG_dF) signature, paired
    with trap_model being the matching HurkxTATModel or SchenkTATModel. mat
    must already be a MaterialField (see newton_gummel_solve) - trap_generation_fn
    is handed an _interior_node_view of it instead (per-node ni/Vt/T/tau_n/
    tau_p, matching n_i/p_i/F_node's interior-node shape), so it correctly
    sees the LOCAL material at a heterojunction without any change to
    tat.tat's own generation-rate formulas."""
    F_node, dF_im1, dF_i, dF_ip1 = _node_field(psi, x)
    n_i, p_i = n[1:-1], p[1:-1]

    mat_node = _interior_node_view(mat)
    G_tat, dGtat_dn, dGtat_dp, dGtat_dF = trap_generation_fn(n_i, p_i, F_node, mat_node, trap_model)
    G_btbt, dGbtbt_dF = btbt_generation(F_node, kane_model)

    R_trap = -G_tat
    dRh_dn, dRh_dp, dRh_dF = -dGtat_dn, -dGtat_dp, -dGtat_dF

    Reff = R_trap - G_btbt
    dReff_dn = dRh_dn
    dReff_dp = dRh_dp
    dReff_dF = dRh_dF - dGbtbt_dF

    dReff_dpsi_im1 = dReff_dF * dF_im1
    dReff_dpsi_i = dReff_dF * dF_i
    dReff_dpsi_ip1 = dReff_dF * dF_ip1

    return Reff, dReff_dn, dReff_dp, dReff_dpsi_im1, dReff_dpsi_i, dReff_dpsi_ip1


def _edge_quantities(psi, phin, phip, n, p, x, mat):
    """Plain-gradient flux only (no R here - Reff is computed separately
    above, unlike newton_solver_qf.py's _edge_quantities which bundles
    both). mat must already be a MaterialField (see newton_gummel_solve)."""
    h = np.diff(x)
    n_avg = (n[:-1] + n[1:]) / 2.0
    p_avg = (p[:-1] + p[1:]) / 2.0
    Jn = -Q * mat.mu_n_edge * n_avg * (phin[1:] - phin[:-1]) / h
    Jp = -Q * mat.mu_p_edge * p_avg * (phip[1:] - phip[:-1]) / h
    return h, n_avg, p_avg, Jn, Jp


def _add_fixed_generation(Rn, Rp, G_ext, cont_scale):
    """G_ext = (Gn, Gp): per-node generation rates (cm^-3 s^-1) held FIXED
    during this Newton solve - a nonlocal BTBT source (btbt/paths1d.py)
    whose electrons and holes are generated at different nodes, lagged by
    the caller's outer iteration (btbt/main_btbt_1d.py). Constant w.r.t.
    U, so no Jacobian term."""
    Gn, Gp = G_ext
    Rn[1:-1] += Q * Gn[1:-1] / cont_scale
    Rp[1:-1] -= Q * Gp[1:-1] / cont_scale


def _residual_only(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale,
                    kane_model, trap_model, trap_generation_fn=_DEFAULT_TRAP_GENERATION_FN, G_ext=None):
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

    _, _, _, Jn, Jp = _edge_quantities(psi, phin, phip, n, p, x, mat)
    Reff, *_ = _reff_and_derivs(n, p, psi, x, mat, kane_model, trap_model, trap_generation_fn)
    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * Reff) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * Reff) / cont_scale
    if G_ext is not None:
        _add_fixed_generation(Rn, Rp, G_ext, cont_scale)

    return np.concatenate([Rpsi, Rn, Rp])


def _residual_and_jacobian(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale,
                            kane_model, trap_model, trap_generation_fn=_DEFAULT_TRAP_GENERATION_FN, G_ext=None):
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

    h_e, n_avg, p_avg, Jn, Jp = _edge_quantities(psi, phin, phip, n, p, x, mat)

    dn_dpsi = n / Vt
    dn_dphin = -n / Vt
    dp_dpsi = -p / Vt
    dp_dphip = p / Vt

    dphin_e = phin[1:] - phin[:-1]
    dphip_e = phip[1:] - phip[:-1]

    dJn_dpsi_e = -Q * mat.mu_n_edge / h_e * (dn_dpsi[:-1] / 2.0) * dphin_e
    dJn_dpsi_ep1 = -Q * mat.mu_n_edge / h_e * (dn_dpsi[1:] / 2.0) * dphin_e
    dJn_dphin_e = -Q * mat.mu_n_edge / h_e * ((dn_dphin[:-1] / 2.0) * dphin_e - n_avg)
    dJn_dphin_ep1 = -Q * mat.mu_n_edge / h_e * ((dn_dphin[1:] / 2.0) * dphin_e + n_avg)

    dJp_dpsi_e = -Q * mat.mu_p_edge / h_e * (dp_dpsi[:-1] / 2.0) * dphip_e
    dJp_dpsi_ep1 = -Q * mat.mu_p_edge / h_e * (dp_dpsi[1:] / 2.0) * dphip_e
    dJp_dphip_e = -Q * mat.mu_p_edge / h_e * ((dp_dphip[:-1] / 2.0) * dphip_e - p_avg)
    dJp_dphip_ep1 = -Q * mat.mu_p_edge / h_e * ((dp_dphip[1:] / 2.0) * dphip_e + p_avg)

    (Reff, dReff_dn, dReff_dp,
     dReff_dpsi_im1, dReff_dpsi_i, dReff_dpsi_ip1) = _reff_and_derivs(
        n, p, psi, x, mat, kane_model, trap_model, trap_generation_fn)

    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * Reff) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * Reff) / cont_scale
    if G_ext is not None:
        _add_fixed_generation(Rn, Rp, G_ext, cont_scale)
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

    # Poisson interior rows - identical to newton_solver_qf.py (Reff/G_btbt
    # don't touch Poisson's own charge term).
    add(idx, idx - 1, lap_m / poisson_scale)
    add(idx, idx, -(lap_m + lap_p) / poisson_scale
        - Q * (dn_dpsi[idx] - dp_dpsi[idx]) / poisson_scale)
    add(idx, idx + 1, lap_p / poisson_scale)
    add(idx, N + idx, (-Q / poisson_scale) * dn_dphin[idx])
    add(idx, 2 * N + idx, (-Q / poisson_scale) * (-dp_dphip[idx]))

    # Electron continuity interior rows.
    r_n = N + idx
    add(r_n, idx - 1, -dJn_dpsi_e[e_lo] / cv / cont_scale
        - Q * dReff_dpsi_im1 / cont_scale)
    add(r_n, idx, (dJn_dpsi_e[e_hi] - dJn_dpsi_ep1[e_lo]) / cv / cont_scale
        - Q * (dReff_dn * dn_dpsi[idx] + dReff_dp * dp_dpsi[idx] + dReff_dpsi_i) / cont_scale)
    add(r_n, idx + 1, dJn_dpsi_ep1[e_hi] / cv / cont_scale
        - Q * dReff_dpsi_ip1 / cont_scale)
    add(r_n, N + idx - 1, -dJn_dphin_e[e_lo] / cv / cont_scale)
    add(r_n, N + idx, (dJn_dphin_e[e_hi] - dJn_dphin_ep1[e_lo]) / cv / cont_scale
        - Q * dReff_dn * dn_dphin[idx] / cont_scale)
    add(r_n, N + idx + 1, dJn_dphin_ep1[e_hi] / cv / cont_scale)
    add(r_n, 2 * N + idx, -Q * dReff_dp * dp_dphip[idx] / cont_scale)

    # Hole continuity interior rows - same pattern, +Q*Reff sign.
    r_p = 2 * N + idx
    add(r_p, idx - 1, -dJp_dpsi_e[e_lo] / cv / cont_scale
        + Q * dReff_dpsi_im1 / cont_scale)
    add(r_p, idx, (dJp_dpsi_e[e_hi] - dJp_dpsi_ep1[e_lo]) / cv / cont_scale
        + Q * (dReff_dn * dn_dpsi[idx] + dReff_dp * dp_dpsi[idx] + dReff_dpsi_i) / cont_scale)
    add(r_p, idx + 1, dJp_dpsi_ep1[e_hi] / cv / cont_scale
        + Q * dReff_dpsi_ip1 / cont_scale)
    add(r_p, 2 * N + idx - 1, -dJp_dphip_e[e_lo] / cv / cont_scale)
    add(r_p, 2 * N + idx, (dJp_dphip_e[e_hi] - dJp_dphip_ep1[e_lo]) / cv / cont_scale
        + Q * dReff_dp * dp_dphip[idx] / cont_scale)
    add(r_p, 2 * N + idx + 1, dJp_dphip_ep1[e_hi] / cv / cont_scale)
    add(r_p, N + idx, Q * dReff_dn * dn_dphin[idx] / cont_scale)

    interior_rows = np.concatenate(rows_list)
    interior_cols = np.concatenate(cols_list)
    interior_data = np.concatenate(data_list)

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


def newton_gummel_solve(x, Cdop, mat: Material, Va, psi_eq, n_eq, p_eq,
                         psi_init=None, phin_init=None, phip_init=None,
                         kane_model=None, trap_model=None,
                         trap_generation_fn=_DEFAULT_TRAP_GENERATION_FN,
                         f_tol=ROW_TOL, maxiter=50, verbose=False, G_ext=None):
    """Same signature/return shape as newton_solver_qf.newton_gummel_solve,
    plus optional kane_model/trap_model (default to the standard Si
    constructors in tat/tat.py if not given) and trap_generation_fn -
    hurkx_tat_generation (default) or schenk_tat_generation from tat.tat,
    paired with a matching HurkxTATModel or SchenkTATModel `trap_model`
    (see plans/tat_btbt_plan.md's Hurkx-then-Schenk design).

    mat may be a plain scalar Material (today's exact behavior) or a
    core.materials.MaterialField (heterojunction) - normalized once here
    (`mf`); see newton_solver_qf.newton_gummel_solve's own docstring for
    the same pattern this mirrors.

    G_ext: optional (Gn, Gp) fixed per-node generation arrays (see
    _add_fixed_generation) - None (default) leaves the solve unchanged."""
    kane_model = kane_model or KaneBTBTModel.si_kane_quadratic()
    if trap_model is None:
        trap_model = HurkxTATModel() if trap_generation_fn is _DEFAULT_TRAP_GENERATION_FN else trap_model
    if trap_model is None:
        raise ValueError("trap_model must be given when trap_generation_fn is overridden "
                          "(e.g. schenk_tat_generation needs a matching SchenkTATModel)")

    N = len(x)
    mf = mat if isinstance(mat, MaterialField) else MaterialField.uniform(mat, x)
    ni0, niL = mf.ni_arr[0], mf.ni_arr[-1]
    dEi0, dEiL = mf.delta_Ei_arr[0], mf.delta_Ei_arr[-1]
    n_bc0, p_bc0 = contact_values(mf, Cdop[0], ni=ni0)
    n_bcL, p_bcL = contact_values(mf, Cdop[-1], ni=niL)
    Vt = mf.Vt

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
        psi0 = psi0.copy(); phin0 = phin0.copy(); phip0 = phip0.copy()
        psi0[0], psi0[-1] = psi_bc
        phin0[0], phin0[-1] = phin_bc
        phip0[0], phip0[-1] = phip_bc
        U0 = np.concatenate([psi0, phin0, phip0])
        # Same row-normalized damped Newton as newton_solver_qf.py (see
        # core/newton_numerics.py). Its linear solve keeps the Ruiz
        # row/column equilibration this solver already relied on for
        # Schenk's large, rapidly field-varying generation (an
        # un-equilibrated Schenk solve stalled at |F|~5e8 from a 1e11 cold
        # start; with equilibration it converged in 12 full steps).
        args = (x, Cdop, mf, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale,
                kane_model, trap_model, trap_generation_fn, G_ext)
        return damped_newton(
            U0,
            lambda U: _residual_and_jacobian(U, *args),
            lambda U: _residual_only(U, *args),
            step_clip=lambda d: uniform_step_clip(d, N, max_psi=1.0, max_qf=MAX_QF_STEP),
            trial_clip=lambda d: componentwise_step_clip(d, N, max_psi=1.0, Vt=Vt),
            tol=f_tol, maxiter=maxiter, verbose=verbose, label="Newton(TAT)")

    if psi_init is None:
        U, merit, it, converged = _run_newton(*_gummel_start())
    else:
        U, merit, it, converged = _run_newton(psi_init, phin_init, phip_init)
        if not converged:
            U_r, merit_r, it_r, conv_r = _run_newton(*_gummel_start())
            if conv_r or merit_r < merit:
                U, merit, it, converged = U_r, merit_r, it_r, conv_r

    if not converged:
        warnings.warn(
            f"Newton(TAT) solve did not converge at Va={Va} V "
            f"(max|F/d|={merit:.3e} V at iteration {it}) even after a Gummel-restart retry - "
            "check this point's self-consistency (J_std/J_mean) before trusting it.")

    psi, phin, phip = unpack_qf(U, N)
    n = mf.ni_arr * np.exp((psi - phin + mf.delta_Ei_arr) / Vt)
    p = mf.ni_arr * np.exp((phip - psi - mf.delta_Ei_arr) / Vt)

    h_e, _, _, Jn, Jp = _edge_quantities(psi, phin, phip, n, p, x, mf)
    Jtot = Jn + Jp
    noise = edge_current_noise(phin, phip, n, p, h_e, Q * mf.mu_n_edge, Q * mf.mu_p_edge)
    # Jtot_resolved: the total current on every edge, read where it is best
    # resolved (core/newton_numerics.resolved_current) and carried to every
    # other edge exactly. Summing the converged electron and hole rows at
    # node i gives Jtot[i] - Jtot[i-1] = q*cvol_i*(Gp_ext - Gn_ext)_i - the
    # local generation/recombination cancels - so the total current is
    # edge-constant without a nonlocal source, and otherwise changes by the
    # known injected charge. The raw Jtot next to a heavily doped contact is
    # roundoff (a huge conductance times a sub-resolution quasi-Fermi step).
    k = int(np.argmin(noise))
    if G_ext is None:
        offset = np.zeros_like(Jtot)
    else:
        Gn_ext, Gp_ext = G_ext
        cvol_i = ph._control_volumes(x)[1:-1]
        offset = np.concatenate([[0.0], np.cumsum(Q * cvol_i * (Gp_ext[1:-1] - Gn_ext[1:-1]))])
    Jtot_resolved = Jtot[k] + offset - offset[k]
    _, J_std, _, _ = resolved_current(Jtot - offset, noise)
    # Terminal current: the (edge-constant) resolved current, or with a
    # nonlocal source the right-contact value.
    J_rep = float(Jtot_resolved[-1])

    return {
        "psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
        "Jn": Jn, "Jp": Jp, "Jtot": Jtot, "Jtot_resolved": Jtot_resolved, "iters": it,
        "J_mean": J_rep, "J_std": J_std, "converged": converged,
    }
