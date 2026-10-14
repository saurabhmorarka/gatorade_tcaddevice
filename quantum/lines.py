"""A Schrodinger line through the channel body, and the quantum potential
Lambda it gives on the drift-diffusion (DD) nodes it crosses.

A line is a fine uniform grid (default 0.1 nm) along the confinement
direction. It starts a little inside the gate oxide (barrier penetration)
and ends either
  - at another oxide wall (double gate, FinFET fin, nanosheet: the two
    walls give volume inversion by themselves), or
  - at a depth D_q in the bulk (planar), where the artificial wall of the
    line is hidden by a taper: over the last `taper_frac` of the line the
    density blends from the quantum to the classical one.

Coupling (plans/quantum_plan.md 4.3). The DD density at node i is
n_i = n_cl,i exp(-Lambda_i/Vt), n_cl,i = ni exp((psi_i - phi_n,i)/Vt), and
its charge is n_i V_i (V_i: the volume the DD solver gives the node's
charge). Lambda is chosen so that this equals the quantum charge of the
node's span [lo_i, hi_i] on the line:

  Lambda_i = -Vt ln( w_i r_i + 1 - w_i ),   r_i = int_{lo_i}^{hi_i} n_QM ds / (n_cl,i V_i)

so the total charge is conserved on any DD mesh, however coarse (the 0.5 nm
fab mesh included); w_i is the taper weight. The first semiconductor node
also takes the charge that penetrates the oxide (the DD has no carriers
there).

With Boltzmann subband statistics r_i does not depend on the quasi-Fermi
level (it cancels between n_QM and n_cl), so Lambda depends on psi alone.
With 2D Fermi-Dirac the line uses one Fermi level, that of its first
semiconductor node (the inversion carriers'), and Lambda also carries the
degeneracy of the inversion layer.
"""
from dataclasses import dataclass

import numpy as np

from quantum import bands
from quantum.schrodinger1d import nc3d, subband_density

LAMBDA_CLIP = (-0.5, 1.0)       # V


@dataclass
class Line:
    s: np.ndarray            # fine grid along the line, cm
    is_semi: np.ndarray      # per fine point
    h: float                 # fine spacing, cm
    dd_idx: np.ndarray       # DD nodes the line owns
    s_node: np.ndarray       # their position along the line, cm
    lo: np.ndarray           # their span on the line, cm
    hi: np.ndarray
    V: np.ndarray            # their DD charge volume (cm in 1D, cm^3 in 3D)
    w: np.ndarray            # taper weight per node (1: fully quantum)


def taper(s, s_semi0, s_end, wall_end, taper_frac):
    if wall_end:
        return np.ones_like(s)
    s_t = s_end - taper_frac * (s_end - s_semi0)
    t = np.clip((s - s_t) / (s_end - s_t), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def line_1d(x, is_semi_node, V_node, semi_lo, semi_hi, depth=20e-7, t_pen=1e-7, h=1e-8,
            taper_frac=0.25):
    """A line along a 1D DD grid x (cm) through the semiconductor body
    [semi_lo, semi_hi]. If the body is thinner than `depth`, the line runs
    wall to wall (+ t_pen into both oxides); otherwise it ends at
    semi_lo + depth with a taper (bulk)."""
    wall_end = semi_hi - semi_lo <= depth
    s_end = semi_hi + t_pen if wall_end else semi_lo + depth
    n = int(round((s_end - (semi_lo - t_pen)) / h))
    s = semi_lo - t_pen + h * np.arange(n + 1)
    is_semi = (s >= semi_lo - 1e-3 * h) & (s <= semi_hi + 1e-3 * h)
    idx = np.flatnonzero(is_semi_node & (x >= semi_lo) & (x <= min(semi_hi, s_end)))
    xs = x[idx]
    mid = 0.5 * (xs[1:] + xs[:-1])
    lo = np.concatenate([[s[0]], mid])
    hi = np.concatenate([mid, [s[-1] if wall_end else min(0.5 * (xs[-1] + x[min(idx[-1] + 1, len(x) - 1)]), s[-1])]])
    w = taper(xs, semi_lo, s_end, wall_end, taper_frac)
    return Line(s=s, is_semi=is_semi, h=h, dd_idx=idx, s_node=xs, lo=lo, hi=np.minimum(hi, s[-1]),
                V=np.asarray(V_node)[idx], w=w)


def _span_integral(s, f, lo, hi):
    """int_lo^hi f ds for piecewise-linear f on the grid s, exact (the span
    ends are inserted into the grid, so a DD cell smaller than the line
    spacing gets the linear density, not a staircase)."""
    t = np.union1d(s, np.concatenate([lo, hi]))
    ft = np.interp(t, s, f)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (ft[1:] + ft[:-1]) * np.diff(t))])
    return F[np.searchsorted(t, hi)] - F[np.searchsorted(t, lo)]


def quantum_potential(line, psi_fine, psi_node, phi_node, carrier, mat, normal=(0.0, 1.0, 0.0),
                      n_states=20, statistics="boltzmann"):
    """Lambda on line.dd_idx (V), plus diagnostics.

    psi_fine: potential on line.s; psi_node, phi_node: potential and the
    carrier's quasi-Fermi potential on line.dd_idx (DD values)."""
    Vt, T = mat.Vt, mat.T
    vals = bands.valleys(carrier, normal)
    Nc = nc3d(vals, T)
    sgn = -1.0 if carrier == "n" else 1.0
    U = sgn * psi_fine + np.where(line.is_semi, 0.0, bands.OXIDE_BARRIER_EV[carrier])
    U_node = sgn * psi_node
    mu_node = -phi_node if carrier == "n" else phi_node        # carrier Fermi level in the U scale
    U_ref = U[line.is_semi].min()
    if statistics == "boltzmann":
        eta0 = 0.0
    else:                                                     # one Fermi level per line (see docstring)
        eta0 = (mu_node[0] - U_ref) / Vt - np.log(Nc / mat.ni)
    nq, levels = subband_density(U - U_ref, line.is_semi, vals, bands.OXIDE_MASS[carrier], line.h,
                                 n_states, T, eta0, statistics)
    q = _span_integral(line.s, nq, line.lo, line.hi)          # quantum charge per node, scaled
    if statistics == "boltzmann":
        # nq is n_QM * Nc/ni * exp((U_ref - mu)/Vt): divide the same factor out of n_cl V
        ncl_V = Nc * np.exp(-(U_node - U_ref) / Vt) * line.V
        scale = mat.ni / Nc * np.exp((mu_node - U_ref) / Vt)
    else:
        ncl_V = mat.ni * np.exp((mu_node - U_node) / Vt) * line.V
        scale = np.ones_like(U_node)
    r = q / np.maximum(ncl_V, 1e-300)
    lam = -Vt * np.log(np.maximum(line.w * r + 1.0 - line.w, 1e-300))
    lam = np.clip(lam, *LAMBDA_CLIP)
    # the point value along the whole line (n_QM / n_cl at each fine point,
    # same taper): Lambda for DD nodes off the line's own column
    w_f = getattr(line, "w_fine", None)
    lam_fine = None
    if w_f is not None and statistics == "boltzmann":
        with np.errstate(over="ignore"):
            rf = np.where(line.is_semi, nq / (Nc * np.exp(-(U - U_ref) / Vt)), 1.0)
        lam_fine = np.clip(-Vt * np.log(np.maximum(w_f * rf + 1.0 - w_f, 1e-300)), *LAMBDA_CLIP)
    return lam, dict(idx=line.dd_idx, q=q * scale, n_fine=nq * (mat.ni / Nc * np.exp((mu_node[0] - U_ref) / Vt)
                                                  if statistics == "boltzmann" else 1.0),
                     levels=[(v, E + U_ref) for v, E, _ in levels], U_ref=U_ref, lam_fine=lam_fine)
