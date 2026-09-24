"""Trap-assisted tunneling (Hurkx field-enhanced SRH) as an ADDITION to the
plain SRH term the 2D solver already carries - pure physics, no mesh.

In silicon (indirect gap) band-to-band leakage is mostly trap-assisted: an
electron reaches a mid-gap trap and tunnels from it to the conduction band
(or a hole to the valence band) through half the gap, instead of crossing
the whole gap in one go. Hurkx, Klaassen & Knuvers (IEEE TED 39(2), 1992)
write this as the SRH rate with field-enhanced capture:

    R = (n p - ni^2) / ( tau_p/(1+Gp) (n+ni) + tau_n/(1+Gn) (p+ni) )

with Gamma from tat/tat.py::hurkx_gamma. The 2D solver already has plain
SRH (Gamma=0), so only the enhancement is added through its generation
hook:

    dG_tat = G_hurkx(n, p, Gn, Gp) - G_srh(n, p),   G = -R            (1)

tat/tat.py::hurkx_tat_generation uses one Gamma for both carriers. Here
Gamma_n and Gamma_p are kept separate, because the nonlocal model (btbt/paths2d.py)
gets them from different paths: uphill for the electron's trap-to-conduction-band
tunneling, downhill for the hole's. With Gamma_n = Gamma_p = Gamma(F) it is
exactly tat/'s local model. Trap at midgap (n1 = p1 = ni), matching the
solver's SRH.
"""
import numpy as np


def trap_generation(n, p, ni, tau_n, tau_p, Gam_n, Gam_p):
    """Full field-enhanced trap generation G = -R (not just the enhancement)
    and dG/dn, dG/dp, with R = (np - ni^2) / (tau_p/(1+Gp)(n+ni) +
    tau_n/(1+Gn)(p+ni)). With tau = 1/s (s: surface recombination velocity,
    cm/s) this is a SURFACE rate in cm^-2 s^-1 - the interface-trap term of
    btbt/paths2d.py. Zero wherever ni == 0."""
    ok = ni > 0
    num = n * p - ni ** 2
    tpe, tne = tau_p / (1.0 + Gam_p), tau_n / (1.0 + Gam_n)
    den = np.where(ok, tpe * (n + ni) + tne * (p + ni), 1.0)
    G = -num / den
    d_dn = -(p * den - num * tpe) / den ** 2
    d_dp = -(n * den - num * tne) / den ** 2
    z = lambda a: np.where(ok, a, 0.0)
    return z(G), z(d_dn), z(d_dp)


def hurkx_enhancement(n, p, ni, tau_n, tau_p, Gam_n, Gam_p):
    """(dG, d/dn, d/dp, d/dGam_n, d/dGam_p) of (1). Zero wherever ni == 0
    (insulator nodes)."""
    ok = ni > 0
    num = n * p - ni ** 2
    tpe, tne = tau_p / (1.0 + Gam_p), tau_n / (1.0 + Gam_n)
    den = tpe * (n + ni) + tne * (p + ni)
    den0 = tau_p * (n + ni) + tau_n * (p + ni)
    den = np.where(ok, den, 1.0)
    den0 = np.where(ok, den0, 1.0)
    dG = -num / den + num / den0
    d_dn = -(p * den - num * tpe) / den ** 2 + (p * den0 - num * tau_p) / den0 ** 2
    d_dp = -(n * den - num * tne) / den ** 2 + (n * den0 - num * tau_n) / den0 ** 2
    # dG/dGam = (num/den^2) * dden/dGam, dden/dGam_n = -tne (p+ni)/(1+Gam_n)
    d_dGn = num / den ** 2 * (-tne * (p + ni) / (1.0 + Gam_n))
    d_dGp = num / den ** 2 * (-tpe * (n + ni) / (1.0 + Gam_p))
    z = lambda a: np.where(ok, a, 0.0)
    return z(dG), z(d_dn), z(d_dp), z(d_dGn), z(d_dGp)
