"""Effective-mass Schrodinger equation on a uniform 1D line, and the
subband density it gives.

  -(hbar^2/2) d/ds [ 1/m(s) dpsi/ds ] + U(s) psi = E psi,   psi = 0 beyond the ends

BenDaniel-Duke finite differences: the mass sits on the half-points, so a
mass step at the Si/SiO2 wall keeps the Hamiltonian symmetric and the
current 1/m dpsi/ds continuous. The matrix is tridiagonal; only the lowest
n_states eigenpairs are computed (LAPACK stebz/stein via
scipy.linalg.eigh_tridiagonal).

Units: s in cm, U and E in eV, |psi|^2 in 1/cm (sum |psi|^2 h = 1).
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal

from core.params import KB, Q

HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
HB2M_EV_NM2 = HBAR ** 2 / (2.0 * M0) / Q * 1e18      # hbar^2/2m0 = 0.0381 eV nm^2


def dos2d_per_mass(T):
    """kT m0 / (pi hbar^2) in cm^-2: the 2D sheet density of one spin-
    degenerate subband of mass m0 at E_F = E_k (Boltzmann)."""
    return KB * T * M0 / (np.pi * HBAR ** 2) * 1e-4


def nc3d(valleys, T):
    """Classical 3D effective density of states of a band, cm^-3:
    sum_v g_v * 2 (m3d kT / 2 pi hbar^2)^(3/2), split-off offsets included."""
    kT = KB * T
    out = 0.0
    for v in valleys:
        out += v.g * 2.0 * (v.m3d * M0 * kT / (2.0 * np.pi * HBAR ** 2)) ** 1.5 * 1e-6 \
            * np.exp(-v.offset_eV * Q / kT)
    return out


def solve_line(U_eV, m_node, h_cm, n_states):
    """Lowest n_states levels of one valley. U_eV, m_node (m0) per node.
    Returns E (k,) in eV and psi (k, N) with sum |psi|^2 h = 1."""
    N = len(U_eV)
    k = min(n_states, N)
    h_nm = h_cm * 1e7
    m_half = 2.0 / (1.0 / m_node[:-1] + 1.0 / m_node[1:])      # harmonic: exact for a step at a half-point
    t = HB2M_EV_NM2 / (m_half * h_nm ** 2)                   # hopping on each half-point
    t_lo = np.concatenate([[HB2M_EV_NM2 / (m_node[0] * h_nm ** 2)], t])
    t_hi = np.concatenate([t, [HB2M_EV_NM2 / (m_node[-1] * h_nm ** 2)]])
    E, V = eigh_tridiagonal(t_lo + t_hi + U_eV, -t, select="i", select_range=(0, k - 1))
    return E, V.T / np.sqrt(h_cm)


def occupancy(eta, statistics):
    """Sheet-density factor of a subband: e^eta (Boltzmann) or ln(1+e^eta)
    (2D Fermi-Dirac), eta = (E_F - E_k)/kT."""
    if statistics == "boltzmann":
        return np.exp(eta)
    return np.logaddexp(0.0, eta)


def subband_density(U_eV, is_semi, valleys, oxide_mass, h_cm, n_states, T, eta_offset, statistics="boltzmann"):
    """Carrier density on the line (cm^-3), the levels and the wavefunctions.

    eta_offset: (E_F - 0)/kT in the energy scale of U, i.e. the subband
    with E_k has eta = eta_offset - E_k/kT (a scalar; the band edge's own
    Nc/ni reference is folded in by the caller, see lines.py)."""
    kT_eV = KB * T / Q
    n = np.zeros(len(U_eV))
    levels = []
    for v in valleys:
        m = np.where(is_semi, v.mz, oxide_mass)
        E, psi = solve_line(U_eV + v.offset_eV, m, h_cm, n_states)
        w = v.g * v.md * dos2d_per_mass(T) * occupancy(eta_offset - E / kT_eV, statistics)
        n += w @ (psi ** 2)
        levels.append((v, E, psi))
    return n, levels
