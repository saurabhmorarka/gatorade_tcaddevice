"""Equilibrium solve and Gummel-map drift-diffusion bias sweep."""
import time

import numpy as np

from core.params import Q, Material, Device
from core import physics as ph
from core.materials import MaterialField


def solve_equilibrium(x, Cdop, mat: Material):
    """Full nonlinear (non-depletion-approximation) equilibrium solve. mat
    may be a plain scalar Material (today's exact behavior, delta_Ei=0
    everywhere) or a MaterialField (heterojunction - per-node ni_arr and
    the delta_Ei_arr band-offset correction, see MaterialField's docstring)."""
    mf = mat if isinstance(mat, MaterialField) else MaterialField.uniform(mat, x)
    psi_bc = ph.equilibrium_bulk_potential_arr(mf.Vt, mf.ni_arr, Cdop, mf.delta_Ei_arr)
    # Good initial guess: exact neutral bulk potential at every point (ignores
    # depletion smoothing, corrected by Newton iterations below).
    psi0 = psi_bc.copy()
    phin = np.zeros_like(x)
    phip = np.zeros_like(x)
    psi, n, p, iters = ph.solve_poisson(x, Cdop, mf, phin, phip, psi0,
                                         eps=mf.eps_edge, ni=mf.ni_arr, delta_Ei=mf.delta_Ei_arr)
    return psi, n, p, iters


def contact_values(mat: Material, Cdop_end: float, ni: float = None):
    """Equilibrium (n, p) at an ohmic contact with net doping Cdop_end, held
    fixed at that value regardless of applied bias (ideal ohmic contact).

    This is a purely local mass-action/neutrality relation (n0*p0=ni^2,
    n0-p0=Cdop_end) with no dependence on any global reference or
    delta_Ei, so no heterojunction-specific generalization of the FORMULA
    is needed (see the harmonic-snuggling-puddle plan) - only which ni
    value it uses. ni=None (default) uses mat.ni (today's exact behavior,
    a plain scalar Material). For a heterojunction, pass the LOCAL ni at
    that contact (e.g. MaterialField.ni_arr[0] or [-1]) explicitly; `mat`
    then only needs to expose .Vt (a MaterialField satisfies this too, so
    the caller can pass either a scalar Material or a MaterialField here
    when overriding ni)."""
    ni = mat.ni if ni is None else ni
    psi0 = ph.equilibrium_bulk_potential(mat, Cdop_end, ni=ni)
    n0 = ni * np.exp(psi0 / mat.Vt)
    p0 = ni * np.exp(-psi0 / mat.Vt)
    return n0, p0


def gummel_solve(x, Cdop, mat: Material, Va, psi_eq, n_eq, p_eq,
                  psi_init=None, phin_init=None, phip_init=None,
                  max_gummel=300, tol=1e-10, verbose=False):
    """Solve the coupled Poisson + continuity equations at applied bias Va
    (forward-positive on the p-side / x<0 contact... wait — see note below)
    using Gummel (decoupled) iteration.

    Boundary convention: x[0] is the p-side contact, x[-1] is the n-side
    contact. The n-side contact is grounded; Va is applied to the p-side
    contact (Va>0 = forward bias).

    mat may be a plain scalar Material (today's exact behavior) or a
    MaterialField (heterojunction) - contact_values() itself only ever
    needs the LOCAL ni at each contact (mf.ni_arr[0]/[-1]), a purely local
    relation unaffected by delta_Ei (see contact_values' own docstring).
    """
    N = len(x)
    mf = mat if isinstance(mat, MaterialField) else MaterialField.uniform(mat, x)
    Vt = mf.Vt
    n_bc0, p_bc0 = contact_values(mf, Cdop[0], ni=mf.ni_arr[0])     # p-side contact
    n_bcL, p_bcL = contact_values(mf, Cdop[-1], ni=mf.ni_arr[-1])  # n-side contact

    psi_bc0 = psi_eq[0] + Va
    phi_bc0 = Va
    psi_bcL = psi_eq[-1]
    phi_bcL = 0.0

    if psi_init is None:
        # linear ramp of the quasi-Fermi split as the initial guess
        frac = np.linspace(1.0, 0.0, N)
        phin = phi_bc0 * frac
        phip = phi_bc0 * frac
        psi = psi_eq + (psi_bc0 - psi_eq[0]) * frac
    else:
        psi, phin, phip = psi_init.copy(), phin_init.copy(), phip_init.copy()

    psi[0], psi[-1] = psi_bc0, psi_bcL
    phin[0], phin[-1] = phi_bc0, phi_bcL
    phip[0], phip[-1] = phi_bc0, phi_bcL

    n = mf.ni_arr * np.exp((psi - phin + mf.delta_Ei_arr) / Vt)
    p = mf.ni_arr * np.exp((phip - psi - mf.delta_Ei_arr) / Vt)
    n[0], n[-1] = n_bc0, n_bcL
    p[0], p[-1] = p_bc0, p_bcL

    # Under-relaxation factor for the quasi-Fermi-level update: a plain
    # (undamped) Gummel map overshoots and can drive the continuity solve to
    # unphysical negative densities, especially at higher forward bias where
    # the exponential nonlinearity is strong. Blending a fraction alpha of
    # each new update in keeps the outer iteration stable.
    alpha = 0.35
    floor = 1.0  # cm^-3, absolute density floor (avoids log(0) without distorting physics)

    for it in range(max_gummel):
        R = ph.srh_recombination(mf, n, p)

        n_new = np.clip(ph.solve_continuity_n(x, psi, mf, R, n_bc0, n_bcL), floor, None)
        p_new = np.clip(ph.solve_continuity_p(x, psi, mf, R, p_bc0, p_bcL), floor, None)

        # phin = psi + delta_Ei - Vt*ln(n/ni), phip = psi + delta_Ei + Vt*ln(p/ni)
        # (invert n=ni*exp((psi-phin+delta_Ei)/Vt), p=ni*exp((phip-psi-delta_Ei)/Vt) -
        # delta_Ei=0 everywhere for a homojunction, reducing to today's exact formula)
        phin_raw = psi + mf.delta_Ei_arr - Vt * np.log(n_new / mf.ni_arr)
        phip_raw = psi + mf.delta_Ei_arr + Vt * np.log(p_new / mf.ni_arr)

        phin_new = phin + alpha * (phin_raw - phin)
        phip_new = phip + alpha * (phip_raw - phip)
        phin_new[0], phin_new[-1] = phi_bc0, phi_bcL
        phip_new[0], phip_new[-1] = phi_bc0, phi_bcL

        psi_guess = psi.copy()
        psi_guess[0], psi_guess[-1] = psi_bc0, psi_bcL
        psi_new, n_from_psi, p_from_psi, _ = ph.solve_poisson(
            x, Cdop, mf, phin_new, phip_new, psi_guess, damping_cap=0.2,
            eps=mf.eps_edge, ni=mf.ni_arr, delta_Ei=mf.delta_Ei_arr)

        d_psi = np.max(np.abs(psi_new - psi))
        d_phin = np.max(np.abs(phin_new - phin))
        d_phip = np.max(np.abs(phip_new - phip))

        psi, phin, phip = psi_new, phin_new, phip_new
        n = np.clip(n_from_psi, floor, None)
        p = np.clip(p_from_psi, floor, None)
        n[0], n[-1] = n_bc0, n_bcL
        p[0], p[-1] = p_bc0, p_bcL

        if verbose:
            print(f"  Gummel it {it}: d_psi={d_psi:.3e} d_phin={d_phin:.3e} d_phip={d_phip:.3e}")

        if d_psi < tol and d_phin < 1e-7 and d_phip < 1e-7:
            break

    Jn, Jp, Jtot = ph.edge_currents(x, psi, n, p, mat)
    # The two edges touching the Dirichlet contact nodes are not covered by an
    # interior continuity equation (the boundary row just pins the density),
    # so the SG flux there can show a small boundary-layer mismatch even at
    # full convergence. Report the terminal current from the interior edges,
    # where dJ/dx=0 should (and does) hold to high precision as a
    # self-consistency check on the converged solution.
    J_interior = Jtot[1:-1] if len(Jtot) > 2 else Jtot
    J_rep = float(np.median(J_interior))
    return {
        "psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
        "Jn": Jn, "Jp": Jp, "Jtot": Jtot, "iters": it + 1,
        "J_mean": J_rep, "J_std": float(np.std(J_interior)),
    }


def voltage_sweep(x, Cdop, mat: Material, dev: Device, Va_list, verbose=False, method="gummel"):
    """Sweep applied bias with solution continuation (each point initialized
    from the previous converged solution for robustness/speed).

    method: "gummel" (decoupled Gummel iteration), "newton" (fully coupled
    Newton with analytic Jacobian in raw densities + Scharfetter-Gummel, see
    newton_solver.py), "newton_qf" (fully coupled Newton in quasi-Fermi
    potentials with a plain-gradient current instead of Scharfetter-Gummel,
    see newton_solver_qf.py - aimed at the strongly asymmetric/degenerate
    doping cases where "newton" doesn't converge), or "newton_avalanche"
    (newton_solver_qf.py's formulation EXTENDED with a field-dependent
    impact-ionization generation term, see newton_solver_avalanche.py;
    voltage-controlled here, so use avalanche/main_avalanche.py's
    arc-length trace to go through the breakdown knee - a special, opt-in mode for modeling
    avalanche breakdown under high reverse bias; not used by any normal
    CMOS-flow example), or "newton_tat" (newton_solver_qf.py's formulation
    extended with reverse-bias junction leakage - Kane band-to-band
    tunneling plus Hurkx trap-assisted tunneling, see tat/newton_solver_tat.py
    and plans/tat_btbt_plan.md - a special, opt-in mode for modeling
    drain-to-substrate leakage; not used by any normal CMOS-flow example) -
    all five share this same continuation/bookkeeping wrapper so they're
    directly comparable point-by-point.
    """
    if method == "gummel":
        solve_fn = gummel_solve
    elif method == "newton":
        from core.newton_solver import newton_gummel_solve
        solve_fn = newton_gummel_solve
    elif method == "newton_qf":
        from core.newton_solver_qf import newton_gummel_solve
        solve_fn = newton_gummel_solve
    elif method == "newton_avalanche":
        from avalanche.newton_solver_avalanche import newton_gummel_solve
        solve_fn = newton_gummel_solve
    elif method == "newton_tat":
        from tat.newton_solver_tat import newton_gummel_solve
        solve_fn = newton_gummel_solve
    else:
        raise ValueError(
            "method must be 'gummel', 'newton', 'newton_qf', 'newton_avalanche', "
            f"or 'newton_tat', got {method!r}")

    psi_eq, n_eq, p_eq, _ = solve_equilibrium(x, Cdop, mat)

    results = []
    # None for the first point, so each solver falls back to its own
    # from-scratch initial guess (for Newton, that includes a Gummel warm
    # start - see newton_solver.py). Passing an explicit equilibrium-based
    # guess even for the first point (as this used to do) skips that
    # fallback and can be a substantially worse starting point, especially
    # far from equilibrium (e.g. the first point of a reverse-bias-heavy
    # sweep).
    psi_prev = phin_prev = phip_prev = None

    for Va in Va_list:
        t0 = time.perf_counter()
        res = solve_fn(x, Cdop, mat, Va, psi_eq, n_eq, p_eq,
                        psi_init=psi_prev, phin_init=phin_prev, phip_init=phip_prev,
                        verbose=verbose)
        res["solve_time_s"] = time.perf_counter() - t0
        res["Va"] = Va
        res["I"] = res["J_mean"] * dev.area
        results.append(res)
        psi_prev, phin_prev, phip_prev = res["psi"], res["phin"], res["phip"]
        if verbose:
            print(f"Va={Va:+.3f} V  I={res['I']:.6e} A  t={res['solve_time_s']:.4f}s "
                  f"({method} iters={res['iters']}, J std/mean={res['J_std']/max(abs(res['J_mean']),1e-30):.2e})")

    return psi_eq, n_eq, p_eq, results
