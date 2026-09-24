"""Nonlocal band-to-band tunneling: the pure-physics part, independent of
dimension and mesh (the path SEARCH is dimension-specific - btbt/paths1d.py,
btbt/paths2d.py - but every path, however found, is turned into a rate here).

Local vs nonlocal. The local Kane model (tat/tat.py::btbt_generation) rates
each point by the field AT that point: G = A F^P exp(-B/F). That silently
assumes the field is constant over the whole tunneling distance and that
the bands bend by at least Eg somewhere nearby. Neither holds in general:
  * The band bending can be smaller than Eg (e.g. a depleted drain surface
    under the gate at modest Vdg): there is then NO state to tunnel into at
    any distance, yet the local field may already be > 1e6 V/cm.
  * The field varies over the ~5-50 nm tunneling distance (a junction's
    triangular field profile, a gate-edge corner).

Nonlocal (the "dynamic nonlocal path" idea): an electron at energy E in the
valence band at r0 tunnels at constant energy to the conduction band at r1.
With homogeneous bands, Ev = -q*psi - Eg/2 and Ec = -q*psi + Eg/2, so
Ev(r0) = Ec(r1) means

    psi(r1) - psi(r0) = Eg/q                                        (1)

- the path runs UPHILL in psi (against E) until the potential has risen by
exactly Eg/q. Its length l defines the path-averaged field

    F_eff = Eg / (q l)                                              (2)

and the Kane expression is evaluated at F_eff with the SAME A, B, P as the
local model:

    G = A F_eff^P exp(-B/F_eff)                                     (3)

(Kane's exponent is the WKB integral through a triangular barrier of height
Eg and length l, B/F = B*l/Eg, so (3) is the natural nonlocal form; in a
uniform field F_eff = F and (3) is exactly the local model - checked in
btbt/main_btbt_1d.py.) If (1) cannot be satisfied - the path leaves the
semiconductor, or the potential stops rising - there is no tunneling from
r0 at all.

Where the carriers go. The hole is left behind at r0 (valence side), the
electron appears at r1 (conduction side). Pair conservation: the pair rate
G(r0)*dV(r0) is removed from r0's hole equation and added to the electron
equation at r1 (spread onto the nodes around r1).

Occupation. Net transitions need a filled valence state at r0 and an empty
conduction state at r1 (and the reverse process). With quasi-Fermi
potentials (Ef = -q*phi), the standard two-reservoir net factor is

    D = 1 - exp(-(phin(r1) - phip(r0)) / Vt)                         (4)

D -> 1 under reverse bias (phin at the n side above phip at the p side),
D = 0 in equilibrium (so, unlike the local model, no spurious BTBT current
at zero bias), D < 0 (net tunneling recombination) in forward bias; it is
clipped to [-1, 1].
"""
import numpy as np

D_MIN = -1.0


def path_rate(l_cm, Eg_eV, model):
    """G (cm^-3 s^-1) for tunneling paths of length l_cm (array; np.inf or
    nan = no path -> 0), from (2)-(3). `model` is a tat.tat.KaneBTBTModel;
    its F_sat_V_cm cap is NOT applied - the cap exists to stop the local
    model extrapolating F^P exp(-B/F) past validated fields, while here the
    path length itself bounds F_eff physically (see module docstring)."""
    l = np.asarray(l_cm, dtype=float)
    ok = np.isfinite(l) & (l > 0)
    F = np.where(ok, Eg_eV / np.where(ok, l, 1.0), 1.0)
    G = model.A * F ** model.P * np.exp(-model.B / F)
    return np.where(ok, G, 0.0), np.where(ok, F, 0.0)


def occupation_factor(phin_end, phip_start, Vt):
    """(4), clipped to [D_MIN, 1]."""
    arg = np.clip(-(np.asarray(phin_end) - np.asarray(phip_start)) / Vt, None, np.log(1.0 - D_MIN))
    return 1.0 - np.exp(arg)
