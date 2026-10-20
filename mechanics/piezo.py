"""Piezoresistance: mobility change from stress.

    d(rho)_ij / rho = Pi_ijkl sigma_kl      ->   mu_t / mu_0 = 1 - t.(Pi:sigma).t

for a current along the unit vector t (first order: d(mu)/mu = -d(rho)/rho).
Pi is the cubic 4th-order piezoresistance tensor, given in crystal axes by
pi11, pi12, pi44 (Smith, Phys. Rev. 94, 42 (1954); lightly doped Si, 300 K;
1e-11 / Pa = 1e-2 / GPa) and rotated to the device frame (mechanics/
elastic.py: x along the channel, y depth, z width; default (001) wafer,
<110> channel). For current along x = [110] that gives the familiar
  longitudinal  pi_l = (pi11 + pi12 + pi44) / 2   (sigma_xx)
  vertical      pi_v = pi12                       (sigma_yy)
  transverse    pi_t = (pi11 + pi12 - pi44) / 2   (sigma_zz)
holes: pi_l = +71.8, pi_t = -66.3; electrons: pi_l = -31.2, pi_t = -17.6
(1e-2 / GPa) - Thompson et al., IEEE TED 53, 1010 (2006), Table I.

The device solver (solver2d/stress_mobility.py) uses the full tensor per
mesh edge. Two modes:
  "linear"      f = 1 + x, x = -t.(Pi:sigma).t, floored at F_FLOOR - the
                textbook small-stress law (Smith), exact at low stress;
  "saturating"  the same slope at zero stress, saturating smoothly
                (tanh) at the gain limit f_max for x > 0 and at the loss
                limit f_min for x < 0:
                    f = 1 + (f_max - 1) tanh(x / (f_max - 1))   x >= 0
                    f = 1 - (1 - f_min) tanh(-x / (1 - f_min))  x <  0
                Holes f_max = 4: the maximum Si inversion-layer hole
                enhancement for uniaxial stress on (001) (Thompson 2006,
                6-band k.p, "~4x"). Electrons f_max = 1.8: the full valley-
                repopulation limit of the electron mobility enhancement
                (Takagi et al., JAP 80, 1567 (1996); Ungersboeck, TU Wien
                thesis 2007, ch. 6: the enhancement saturates near 1%
                strain). f_min = 0.5 for both (a stated default, not a fit).
Doping factor (optional, on by default): Pi(N) = P(N) Pi(low doping),
  electrons: Kanda's P(N, T) = (300/T) F_-1/2(eta) / F_1/2(eta), eta the
  reduced Fermi level at the doping N (Kanda, IEEE TED 29, 64 (1982));
  holes: Richter et al.'s fit, which Kanda under-reads in p-Si (J. Appl.
  Phys. 104, 023715 (2008), Eq. 20 / Table IV):
  P = Th^-0.9 / (1 + (N/6e19)^0.43 Th^-0.1 + (N/7e20)^1.6 Th^-3), Th = T/300.
"""
import numpy as np

# Smith (1954), 1e-11 / Pa = 1e-2 / GPa: (pi11, pi12, pi44)
SMITH = {"n": (-102.2e-2, 53.4e-2, -13.6e-2), "p": (6.6e-2, -1.1e-2, 138.1e-2)}
COEFFS = {"smith": SMITH}

MODES = ("linear", "saturating")
SATURATION = {"n": dict(f_max=1.8, f_min=0.5), "p": dict(f_max=4.0, f_min=0.5)}
F_FLOOR = 0.05          # the linear mode's floor: a mobility never goes below 5% of mu0


def pi_110_on_001(carrier):
    p11, p12, p44 = SMITH[carrier]
    return dict(l=(p11 + p12 + p44) / 2, v=p12, t=(p11 + p12 - p44) / 2)


def mobility_change(sigma_GPa, carrier):
    """Linear d(mu)/mu (fraction) for current along x from Voigt stress
    (GPa) in the (001)/<110> device frame (the flow reports' channel-mean
    preview)."""
    c = pi_110_on_001(carrier)
    s = np.asarray(sigma_GPa, float)
    return -(c["l"] * s[0] + c["v"] * s[1] + c["t"] * s[2])


# --- the full tensor ---------------------------------------------------------------
def cubic_pi_tensor(p11, p12, p44):
    """(3,3,3,3) piezoresistance tensor of a cubic crystal in its own axes:
    Pi_iiii = pi11, Pi_iijj = pi12, Pi_ijij = Pi_ijji = pi44/2 (i != j), so
    that d(rho)_23/rho = pi44 sigma_23 (the Voigt pi44)."""
    P = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                P[i, i, i, i] = p11
            else:
                P[i, i, j, j] = p12
                P[i, j, i, j] = P[i, j, j, i] = p44 / 2
    return P


def pi_tensor(carrier, R=None, coeffs="smith"):
    """Pi (3,3,3,3, 1/GPa) of `carrier` ('n' or 'p') in the device frame R
    (rows = device x, y, z in crystal axes; default (001) wafer, <110>
    channel, mechanics.elastic.FRAME_001_110). coeffs: a COEFFS name or a
    {carrier: (pi11, pi12, pi44)} dict."""
    from mechanics.elastic import FRAME_001_110, frame
    c = COEFFS[coeffs] if isinstance(coeffs, str) else coeffs
    R = frame(FRAME_001_110 if R is None else R)
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, cubic_pi_tensor(*c[carrier]))


_VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def edge_weights(t, Pi):
    """(E, 6) weights a such that t.(Pi:sigma).t = a . sigma_voigt for unit
    directions t (E, 3) and Voigt stress (xx, yy, zz, yz, xz, xy; plain
    shear components, so the shear weights carry the factor 2)."""
    t = np.asarray(t, float)
    A = np.einsum("ei,ej,ijkl->ekl", t, t, Pi)
    out = np.empty((len(t), 6))
    for k, (a, b) in enumerate(_VOIGT):
        out[:, k] = A[:, a, b] if a == b else A[:, a, b] + A[:, b, a]
    return out


def linear_change(t, sigma_voigt, Pi):
    """x = -t.(Pi:sigma).t per edge: the first-order d(mu)/mu."""
    return -np.einsum("ek,ek->e", edge_weights(t, Pi), np.asarray(sigma_voigt, float))


def factor(x, carrier, mode="saturating", f_max=None, f_min=None):
    """mu / mu0 from the first-order change x (see the module docstring)."""
    x = np.asarray(x, float)
    if mode == "linear":
        return np.maximum(1.0 + x, F_FLOOR)
    if mode != "saturating":
        raise ValueError(f"piezo mode must be one of {MODES}, not {mode!r}")
    s = SATURATION[carrier]
    g = (s["f_max"] if f_max is None else f_max) - 1.0
    l_ = 1.0 - (s["f_min"] if f_min is None else f_min)
    if g <= 0 or not 0 < l_ < 1:
        raise ValueError("saturating piezo needs f_max > 1 and 0 < f_min < 1")
    return np.where(x >= 0, 1.0 + g * np.tanh(np.maximum(x, 0) / g), 1.0 - l_ * np.tanh(np.maximum(-x, 0) / l_))


# --- doping factor ------------------------------------------------------------------
def _fd_half_pair(eta):
    """Normalized Fermi-Dirac integrals (F_-1/2, F_1/2) at eta (array), by
    quadrature in u = sqrt(energy): F_j = 2/Gamma(j+1) int u^(2j+1)/(1+exp(u^2-eta)) du."""
    eta = np.atleast_1d(np.asarray(eta, float))
    umax = np.sqrt(np.maximum(eta, 0) + 40.0).max()
    u = np.linspace(0.0, umax, 4001)
    occ = 1.0 / (1.0 + np.exp(np.clip(u[None, :] ** 2 - eta[:, None], -700, 700)))
    fm = 2.0 / np.sqrt(np.pi) * np.trapezoid(occ, u, axis=1)                 # Gamma(1/2) = sqrt(pi)
    fp = 2.0 / (np.sqrt(np.pi) / 2) * np.trapezoid(occ * u[None, :] ** 2, u, axis=1)   # Gamma(3/2)
    return fm, fp


def kanda_factor(N, T=300.0, Nc=2.8e19):
    """Kanda (1982) P(N, T) = (300/T) F_-1/2(eta)/F_1/2(eta), eta from
    N = Nc(T) F_1/2(eta) (Nc at T: Nc(300) (T/300)^1.5)."""
    N = np.asarray(N, float)
    NcT = Nc * (T / 300.0) ** 1.5
    # tabulate eta -> N once, invert by interpolation in log N
    eta = np.linspace(-30.0, 60.0, 1801)
    fm, fp = _fd_half_pair(eta)
    logn = np.log(NcT * fp)
    ratio = fm / fp
    r = np.interp(np.log(np.maximum(N, 1.0)), logn, ratio, left=1.0, right=ratio[-1])
    return (300.0 / T) * r


RICHTER = dict(Nb=6e19, Nc=7e20, theta=0.9, alpha=0.43, beta=0.1, gamma=1.6, eta=3.0)


def richter_factor(N, T=300.0):
    """Richter et al. (2008) Eq. 20: p-type pi44 doping/temperature factor."""
    c = RICHTER
    th = T / 300.0
    N = np.asarray(N, float)
    return th ** -c["theta"] / (1.0 + (N / c["Nb"]) ** c["alpha"] * th ** -c["beta"]
                                + (N / c["Nc"]) ** c["gamma"] * th ** -c["eta"])


def doping_factor(N, carrier, T=300.0):
    """P(N, T) of the coefficients: Kanda for electrons, Richter for holes."""
    return kanda_factor(N, T) if carrier == "n" else richter_factor(N, T)
