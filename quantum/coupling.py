"""The Schrodinger-Poisson outer loop (predictor-corrector, Trellakis et
al., J. Appl. Phys. 81, 7880 (1997)).

  predictor: the DD/Poisson Newton solve with Lambda fixed. The density
             stays n = n_cl(psi) exp(-Lambda/Vt), i.e. the quantum density
             of the last line solve times exp(dpsi/Vt) - Trellakis' first-
             order estimate of how n_QM responds to the potential. This is
             what keeps the loop stable where plain density mixing
             oscillates.
  corrector: the line solves at the new psi give the new Lambda.

Converged when Lambda changes by less than `tol` (V) at every node that
holds a non-negligible share of the quantum charge (Lambda of an empty node
is irrelevant and can wander).
"""
import numpy as np


def sp_loop(solve_dd, update_lambda, lam0, tol=1e-3, max_outer=40, mix=1.0, q_floor=1e-8):
    """solve_dd(lam) -> state; update_lambda(state) -> (lam_new, q) with q
    the quantum charge per node. Returns (state, lam, n_outer, history)."""
    lam = lam0.copy()
    hist = []
    for k in range(1, max_outer + 1):
        state = solve_dd(lam)
        lam_new, q = update_lambda(state)
        sig = q > q_floor * q.max() if q.max() > 0 else np.ones(len(q), bool)
        d = float(np.max(np.abs(lam_new - lam)[sig])) if sig.any() else 0.0
        hist.append(d)
        if d < tol:
            return state, lam, k, hist
        lam = lam + mix * (lam_new - lam)
    raise RuntimeError(f"Schrodinger-Poisson loop did not converge in {max_outer} iterations "
                       f"(last max |dLambda| = {hist[-1]:.2e} V)")
