"""Field-dependent impact-ionization (avalanche generation) coefficients.

Pure physics, no solver/mesh dependency (mirrors physics.py's own
stateless-function style) - so this can be sanity-checked standalone before
being wired into any Newton solve.

Model: van Overstraeten & de Man (Solid-State Electronics 13 (1970) 583),
the standard local-field Chynoweth-law avalanche model for silicon,
alpha(E) = a*exp(-b/E), with electrons using a single (a,b) pair across
their whole validated field range and holes split into two field regions
(the model's own historical fit). NOT valid outside the tabulated field
range or far from T=300K (no temperature scaling is implemented here - if
Material.T deviates significantly from 300K, alpha(E) below is still
evaluated at the 300K-fit coefficients, which is a known, documented
limitation, not a bug).

This is intentionally the classic *local* field model (as opposed to a
non-local "lucky electron" or hydrodynamic-energy model): alpha depends only
on the field AT that point, not on carrier history along its path. That's
the standard choice for a self-consistent generation term inside a
drift-diffusion continuity equation (as opposed to a post-processing
ionization-integral breakdown estimate, which analytic.ionization_integral
computes separately as an independent cross-check).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class AvalancheModel:
    """Chynoweth-law coefficients alpha(E) = a*exp(-b/E), E in V/cm, alpha
    in cm^-1.

    E_floor_V_cm is only a guard against dividing by zero: below it alpha
    is set to exactly 0, which at 1e4 V/cm is already exp(-123) ~ 1e-54 of
    the prefactor, so alpha stays continuous for any practical purpose.
    The fit's own validated range starts at 1.75e5 V/cm, and an earlier
    version hard-floored alpha to 0 right there - a JUMP (alpha_p falls from
    ~14 cm^-1 to 0) that Newton could never settle across once some edge's
    driving force sat near 1.75e5 V/cm, e.g. at a depletion edge during
    breakdown (DEVELOPMENT_LOG.md session 23). Below 1.75e5 V/cm the formula
    is an extrapolation, but a smooth and physically negligible one."""
    a_n: float = 7.03e5      # electrons, single field region
    b_n: float = 1.231e6
    a_p_lo: float = 1.582e6  # holes, E in [E_floor_V_cm, E_split_V_cm)
    b_p_lo: float = 2.036e6
    a_p_hi: float = 6.71e5   # holes, E >= E_split_V_cm
    b_p_hi: float = 1.693e6
    E_floor_V_cm: float = 1.0e4
    E_split_V_cm: float = 4.0e5

    @staticmethod
    def si_von_overstraeten_de_man() -> "AvalancheModel":
        return AvalancheModel()


def ionization_coeffs(E_abs: np.ndarray, model: AvalancheModel):
    """Vectorized alpha_n(E), alpha_p(E) and their field derivatives.

    E_abs: electric field MAGNITUDE (cm/V... i.e. V/cm), any shape, >= 0.
    Returns (alpha_n, alpha_p, dalpha_n_dE, dalpha_p_dE), each the same
    shape as E_abs.

    d/dE [a*exp(-b/E)] = a*exp(-b/E) * (b/E^2) = alpha * b/E^2.
    """
    E = np.asarray(E_abs, dtype=float)
    below_floor = E < model.E_floor_V_cm
    # Avoid 0-division / warnings from the floored region; the result there
    # is overwritten by np.where below regardless of what's computed here.
    E_safe = np.where(below_floor, model.E_floor_V_cm, E)

    alpha_n = model.a_n * np.exp(-model.b_n / E_safe)
    dalpha_n_dE = alpha_n * model.b_n / E_safe ** 2

    hi = E_safe >= model.E_split_V_cm
    a_p = np.where(hi, model.a_p_hi, model.a_p_lo)
    b_p = np.where(hi, model.b_p_hi, model.b_p_lo)
    alpha_p = a_p * np.exp(-b_p / E_safe)
    dalpha_p_dE = alpha_p * b_p / E_safe ** 2

    alpha_n = np.where(below_floor, 0.0, alpha_n)
    alpha_p = np.where(below_floor, 0.0, alpha_p)
    dalpha_n_dE = np.where(below_floor, 0.0, dalpha_n_dE)
    dalpha_p_dE = np.where(below_floor, 0.0, dalpha_p_dE)

    return alpha_n, alpha_p, dalpha_n_dE, dalpha_p_dE


def ionization_integrals_from_field(x, psi, model: AvalancheModel):
    """Classical breakdown criterion evaluated on a SOLVED potential profile:
    the electron- and hole-initiated ionization integrals

        I_n = int alpha_n * exp(-int_x0^x (alpha_n - alpha_p) dx') dx
        I_p = int alpha_p * exp(-int_x^xL (alpha_p - alpha_n) dx') dx

    over the whole device, with alpha evaluated at the local field
    |dpsi/dx| of each edge (x increasing from the p-side contact; electrons
    drift toward +x, holes toward -x). Either integral reaching 1 is the
    local-field condition for the multiplication factor to diverge, so at
    the numerically traced breakdown voltage both should be ~1 - an
    independent check that the PDE solve's breakdown is consistent with the
    ionization coefficients it was given. Uses |E| everywhere, so in the thin
    high-field slice of a heavily doped side (where the solver's hybrid
    driving force deliberately does not, see newton_solver_avalanche.py) it
    slightly overcounts. Returns (I_n, I_p)."""
    h = np.diff(x)
    an, ap, _, _ = ionization_coeffs(np.abs(np.diff(psi)) / h, model)
    d_n = np.cumsum((an - ap) * h)
    d_p = np.cumsum(((ap - an) * h)[::-1])[::-1]
    return float(np.sum(an * np.exp(-d_n) * h)), float(np.sum(ap * np.exp(-d_p) * h))
