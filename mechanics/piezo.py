"""Piezoresistance: mobility change from stress (linear, small-stress).

    d(rho)/rho = sum_k pi_k sigma_k   ->   d(mu)/mu ~ -d(rho)/rho

with the cubic coefficients pi11, pi12, pi44 (Smith, Phys. Rev. 94, 42
(1954); lightly doped Si, 300 K, in 1/GPa) rotated to the device frame.
For a (001) wafer and a <110> channel (x = [110], y = [001] vertical,
z = [-110] width), the coefficients for current along x are
  longitudinal  pi_l = (pi11 + pi12 + pi44) / 2   (sigma_xx)
  vertical      pi_v = pi12                       (sigma_yy)
  transverse    pi_t = (pi11 + pi12 - pi44) / 2   (sigma_zz)

This is a FIRST-ORDER estimate. Real inversion-layer mobility gains
saturate at GPa stress (valley/subband repopulation completes), and depend
on doping, field and confinement. Phase 4 puts the full tensor Pi:sigma
into the device solver's per-edge mobility.
"""
import numpy as np

# Smith (1954), 1e-11 / Pa = 1e-2 / GPa
SMITH = {"n": (-102.2e-2, 53.4e-2, -13.6e-2), "p": (6.6e-2, -1.1e-2, 138.1e-2)}


def pi_110_on_001(carrier):
    p11, p12, p44 = SMITH[carrier]
    return dict(l=(p11 + p12 + p44) / 2, v=p12, t=(p11 + p12 - p44) / 2)


def mobility_change(sigma_GPa, carrier):
    """Linear d(mu)/mu (fraction) for current along x from Voigt stress
    (GPa) in the (001)/<110> device frame."""
    c = pi_110_on_001(carrier)
    s = np.asarray(sigma_GPa, float)
    return -(c["l"] * s[0] + c["v"] * s[1] + c["t"] * s[2])
