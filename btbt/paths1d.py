"""1D nonlocal BTBT path search (see btbt/kernel.py for the physics): on a 1D
grid the "field line" through a node is just the x axis, so the search is:
from node i walk uphill in psi until psi has risen by Eg/q.

Used to validate the kernel and the lagged (outer-iteration) coupling on the
fast 1D drain-substrate diode (btbt/main_btbt_1d.py) before the 2D
field-line version (btbt/paths2d.py) uses the same kernel.
"""
import numpy as np

from btbt.kernel import path_rate, occupation_factor


def find_paths_1d(x, psi, Eg_eV, psi_tol=1e-12, level=None, Eg_nodes=None):
    """For every interior node i: the path length l[i] (cm, inf = no path),
    the end point as (j_lo[i], w_hi[i]) - between nodes j_lo and j_lo+1 at
    fraction w_hi from j_lo - and the bandgap averaged along the path.
    The uphill direction comes from the centered difference of psi at i. The
    energy test uses `level` (default psi; -Ec at a heterojunction): the
    path is complete once level has risen by Eg_eV (scalar or per node =
    Eg at the start, i.e. Ec(end) = Ev(start)). It fails if level stops
    rising first or the grid ends. Eg_nodes (default Eg_eV) is the local gap
    averaged along the path (trapezoid rule)."""
    N = len(x)
    lev = psi if level is None else level
    Eg_start = np.broadcast_to(np.asarray(Eg_eV, dtype=float), (N,))
    Egn = Eg_start if Eg_nodes is None else np.asarray(Eg_nodes, dtype=float)
    l = np.full(N, np.inf)
    j_lo = np.zeros(N, dtype=int)
    w_hi = np.zeros(N)
    Eg_path = Eg_start.copy()
    for i in range(1, N - 1):
        s = 1 if psi[i + 1] - psi[i - 1] > 0 else -1
        target = lev[i] + Eg_start[i]
        seg = lev[i::s] if s > 0 else lev[i::-1]
        k = np.argmax(seg >= target)
        if seg[k] < target:
            continue
        if np.any(np.diff(seg[:k + 1]) < -psi_tol):
            continue
        a, b = seg[k - 1], seg[k]
        t = (target - a) / (b - a)
        ja, jb = i + s * (k - 1), i + s * k
        xe = x[ja] + t * (x[jb] - x[ja])
        l[i] = abs(xe - x[i])
        lo, hi = min(ja, jb), max(ja, jb)
        j_lo[i] = lo
        w_hi[i] = (xe - x[lo]) / (x[hi] - x[lo])
        idx = np.arange(i, ja + s, s) if k > 1 else np.array([i])
        xs = np.concatenate([x[idx], [xe]])
        es = np.concatenate([Egn[idx], [Egn[ja] + t * (Egn[jb] - Egn[ja])]])
        Eg_path[i] = np.trapezoid(es, xs) / (xs[-1] - xs[0]) if len(xs) > 1 and xs[-1] != xs[0] else Egn[i]
    return l, j_lo, w_hi, Eg_path


def nonlocal_generation_1d(x, psi, phin, phip, cvol, Eg_eV, Vt, model, level=None, Eg_ref_eV=None):
    """Per-node (Gn, Gp) in cm^-3 s^-1: holes generated at each path's start
    node, electrons at its end (split linearly onto the two bracketing
    nodes, interior nodes only - a share landing on a contact node is moved
    onto its interior neighbour so no generated carrier is lost). Also
    returns the per-node path length and F_eff for diagnostics.
    Heterojunction: Eg_eV per node, level = -Ec, Eg_ref_eV = the Kane
    fit's gap (see btbt/kernel.py::path_rate)."""
    N = len(x)
    l, j_lo, w_hi, Eg_path = find_paths_1d(x, psi, Eg_eV, level=level,
                                           Eg_nodes=None if np.ndim(Eg_eV) == 0 else Eg_eV)
    G, F_eff = path_rate(l, Eg_path, model, Eg_ref_eV=Eg_ref_eV)
    ok = G > 0
    j_hi = np.minimum(j_lo + 1, N - 1)
    phin_end = (1 - w_hi) * phin[j_lo] + w_hi * phin[j_hi]
    G = np.where(ok, G * occupation_factor(phin_end, phip, Vt), 0.0)

    Gp = G.copy()
    Gp[0] = Gp[-1] = 0.0
    pair = Gp * cvol                          # pairs / (s * cm^2) per node
    elec = np.zeros(N)
    np.add.at(elec, np.clip(j_lo, 1, N - 2), pair * (1 - w_hi))
    np.add.at(elec, np.clip(j_hi, 1, N - 2), pair * w_hi)
    Gn = elec / cvol
    return Gn, Gp, l, F_eff
