"""Pseudo-arclength continuation of a device I-V curve in the (Va, ln|J|)
plane - solver-agnostic numerics (no physics), the 1D analogue of Genius-
TCAD's dynamic load-line trace (src/solver/ddm_common/ddm_solver.cc,
solve_iv_trace) and Silvaco Atlas' CURVETRACE.

WHY: a voltage-controlled sweep fixes Va and solves for the state U. Near
avalanche breakdown dI/dVa grows by orders of magnitude within millivolts
(and past it the curve can turn back to lower |Va|), so any fixed Va step
either overshoots the curve or needs absurdly fine hand-tuned spacing.
Here Va is an extra unknown, and each step instead advances a fixed
distance ds along the curve itself, measured in
    s^2 = (dVa / V_scale)^2 + (d ln|J|)^2,
which stays well-behaved whether the curve is flat (leakage: steps are
mostly in Va), vertical (breakdown: steps are mostly decades of current)
or folding back.

One step = secant predictor from the last two points, then Newton
(core/newton_numerics.damped_newton) on the bordered system
    F(U, Va) = 0
    t_V*(Va - Va_k)/V_scale + t_L*(ln|J(U)| - ln|J_k|) - ds = 0
with (t_V, t_L) the unit secant direction. ds shrinks on a failed step and
grows after easy ones.
"""
import numpy as np
import scipy.sparse as sp

from core.newton_numerics import ROW_TOL, damped_newton


def trace_iv(residual_and_jacobian, residual, dF_dVa, current_and_grad, pick_edge,
             seeds, step_clip=None, trial_clip=None, V_scale=1.0, ds0=0.3, ds_max=1.0, ds_min=1e-4,
             J_stop=1e3, V_limit=None, max_points=500, tol=ROW_TOL, maxiter=40,
             verbose=False):
    """Trace the I-V curve from `seeds` until |J| >= J_stop.

    residual_and_jacobian(U, Va) -> (F, J sparse n x n); residual(U, Va) -> F;
    dF_dVa(U, Va) -> dense length-n vector.
    current_and_grad(U, k) -> (J_k, cols, vals): the current used in the
        arc-length metric (e.g. the total current on edge k) and its
        nonzero gradient entries w.r.t. U.
    pick_edge(U) -> k: which measure to use for the next step (e.g. the
        edge where the current is best resolved); held fixed within a step.
    seeds: >= 2 converged (U, Va) points, in tracing order.
    step_clip / trial_clip (delta_aug) -> delta_aug: optional step limiters
        for the augmented unknown [U, Va] (see damped_newton).
    V_limit: stop once |Va| exceeds it (None = no limit).

    Returns dict(Va, J, iters (arrays over the traced points, seeds
    excluded), U (list of states), status, rejections)."""
    pts = []
    for U, Va in seeds:
        J, _, _ = current_and_grad(U, pick_edge(U))
        pts.append((U.copy(), float(Va), float(np.log(abs(J)))))
    n = len(pts[0][0])
    out_V, out_J, out_it, out_U = [], [], [], []
    ds, rejections, status = ds0, 0, "max_points"

    while len(out_V) < max_points:
        (U1, V1, L1), (U2, V2, L2) = pts[-2], pts[-1]
        sec = np.array([(V2 - V1) / V_scale, L2 - L1])
        sec_len = np.linalg.norm(sec)
        t = sec / sec_len
        a = ds / sec_len
        Z0 = np.append(U2 + a * (U2 - U1), V2 + a * (V2 - V1))
        k = pick_edge(U2)

        def aug(Z, need_J):
            U, Va = Z[:n], Z[n]
            Jk, cols, vals = current_and_grad(U, k)
            g = t[0] * (Va - V2) / V_scale + t[1] * (np.log(abs(Jk)) - L2) - ds
            if not need_J:
                return np.append(residual(U, Va), g)
            F, Jm = residual_and_jacobian(U, Va)
            col = sp.csc_matrix(dF_dVa(U, Va).reshape(-1, 1))
            row = sp.csr_matrix((t[1] * vals / Jk, (np.zeros_like(cols), cols)), shape=(1, n))
            Jaug = sp.bmat([[Jm, col], [row, sp.csr_matrix([[t[0] / V_scale]])]], format="csc")
            return np.append(F, g), Jaug

        try:
            Z, merit, it, ok = damped_newton(Z0, lambda Z: aug(Z, True), lambda Z: aug(Z, False),
                                              step_clip=step_clip, trial_clip=trial_clip, tol=tol,
                                              maxiter=maxiter)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError, RuntimeError):
            ok, it = False, maxiter
        if not ok:
            rejections += 1
            ds *= 0.5
            if verbose:
                print(f"  arclength: step rejected, ds -> {ds:.3g}")
            if ds < ds_min:
                status = "step_collapse"
                break
            continue

        U, Va = Z[:n], float(Z[n])
        J, _, _ = current_and_grad(U, pick_edge(U))
        pts.append((U.copy(), Va, float(np.log(abs(J)))))
        out_V.append(Va); out_J.append(float(J)); out_it.append(it); out_U.append(U.copy())
        if verbose:
            print(f"  arclength: Va={Va:+.5f} V  J={J:.4e}  it={it}  ds={ds:.3g}")
        if it <= 4:
            ds = min(1.5 * ds, ds_max)
        if abs(J) >= J_stop:
            status = "reached_J_stop"
            break
        if V_limit is not None and abs(Va) > V_limit:
            status = "V_limit"
            break

    return {"Va": np.array(out_V), "J": np.array(out_J), "iters": np.array(out_it),
            "U": out_U, "status": status, "rejections": rejections}
