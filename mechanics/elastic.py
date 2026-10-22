"""Linear elastic constants for the stress solver.

Voigt order everywhere: (xx, yy, zz, yz, xz, xy). Stresses are plain
components, strains are ENGINEERING strains (gamma = 2 eps for shear), so
sigma = C @ eps with the usual 6x6 stiffness C. Units: GPa for C and stress,
nm for lengths (strain is dimensionless).

Device frame (the solver's x, y, z):
  x - along the channel (current flow),
  y - depth into the wafer (positive DOWN, the fab's convention),
  z - across the width.
The default wafer is (001) with a <110> channel, so in crystal axes
  x = [110]/sqrt2, y = [00-1], z = [-110]/sqrt2  (right handed).
A cubic crystal's stiffness is rotated into that frame with the Bond
matrix; amorphous films and metals are isotropic (E, nu).

Material data come from core.material_db (E, nu, C11/C12/C44, lattice
constant, typical intrinsic stress). SiGe uses Vegard's law for C_ij and
Dismukes' lattice constant a(x) = 5.431 + 0.20 x + 0.027 x^2 A.
"""
from dataclasses import dataclass, field

import numpy as np

from core import material_db

# crystal axes of the device frame's x, y, z (rows), (001) wafer, <110> channel
FRAME_001_110 = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
FRAME_001_100 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def channel_frame(channel_dir=110):
    """Device frame of a (001) wafer with the channel along <channel_dir>
    (110: the standard, FinFET side walls (110); 100: a 45-degree rotated
    layout, side walls (100))."""
    d = int(round(float(channel_dir)))
    if d == 110:
        return FRAME_001_110
    if d == 100:
        return FRAME_001_100
    raise ValueError(f"channel_dir must be 110 or 100 (a (001) wafer), not {channel_dir!r}")


def frame(rows):
    """Orthonormal rotation matrix whose rows are the device x, y, z axes
    given in crystal coordinates (any lengths)."""
    R = np.asarray(rows, float)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-9) or np.linalg.det(R) < 0:
        raise ValueError("frame rows must be orthogonal and right handed")
    return R


def bond_matrix(R):
    """6x6 Bond matrix M of rotation R (new axes = rows of R, in old
    coordinates): sigma' = M sigma and C' = M C M^T for Voigt stress and
    engineering strain."""
    R = np.asarray(R, float)
    M = np.zeros((6, 6))
    pairs = [(1, 2), (0, 2), (0, 1)]                 # yz, xz, xy
    for i in range(3):
        for j in range(3):
            M[i, j] = R[i, j] ** 2
        for k, (a, b) in enumerate(pairs):
            M[i, 3 + k] = 2.0 * R[i, a] * R[i, b]
    for r, (p, q) in enumerate(pairs):
        for j in range(3):
            M[3 + r, j] = R[p, j] * R[q, j]
        for k, (a, b) in enumerate(pairs):
            M[3 + r, 3 + k] = R[p, a] * R[q, b] + R[p, b] * R[q, a]
    return M


def isotropic_C(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    return C


def cubic_C(c11, c12, c44):
    C = np.zeros((6, 6))
    C[:3, :3] = c12
    C[np.arange(3), np.arange(3)] = c11
    C[np.arange(3, 6), np.arange(3, 6)] = c44
    return C


def rotate_C(C, R):
    M = bond_matrix(R)
    return M @ C @ M.T


def strain_to_voigt(e):
    """3x3 tensor strain -> Voigt engineering strain."""
    e = np.asarray(e, float)
    return np.array([e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])


def sige_lattice_A(x_ge):
    """Relaxed Si(1-x)Ge(x) lattice constant (Dismukes et al. 1964)."""
    return 5.431 + 0.20 * x_ge + 0.027 * x_ge ** 2


def sige_mismatch(x_ge, x_sub=0.0):
    """Lattice mismatch f = (a_epi - a_sub)/a_sub of Si(1-x)Ge(x) grown on
    Si(1-x_sub)Ge(x_sub): the epi's stress-free strain relative to the
    substrate lattice (positive: the epi is bigger and grows compressed).
    x = 0.5 on Si: +1.97%."""
    a, a0 = sige_lattice_A(x_ge), sige_lattice_A(x_sub)
    return (a - a0) / a0


@dataclass
class ElasticMaterial:
    """One material as the solver sees it: C (6x6, device frame, GPa) and
    its compliance S."""
    name: str
    C: np.ndarray
    crystalline: bool = False
    S: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.S = np.linalg.inv(self.C)

    def eigenstrain_from_stress(self, sigma0):
        """The stress-free strain that gives intrinsic stress sigma0 (Voigt,
        GPa) when the film is held at zero strain: eps* = -S sigma0."""
        return -self.S @ np.asarray(sigma0, float)


def material(name, R=FRAME_001_110, x_ge=None):
    """ElasticMaterial for a fab / catalog material name. Cubic crystals
    (C11/C12/C44 in the catalog, or SiGe with x_ge) are rotated to the
    device frame R; everything else is isotropic E, nu."""
    if name == "SiGe" or (x_ge is not None and name in ("Silicon", "Si")):
        x = 0.5 if x_ge is None else x_ge
        si, ge = material_db.get("Silicon"), material_db.get("Germanium")
        c = [(1 - x) * getattr(si, k) + x * getattr(ge, k) for k in ("c11_GPa", "c12_GPa", "c44_GPa")]
        return ElasticMaterial(f"Si{1 - x:.2f}Ge{x:.2f}", rotate_C(cubic_C(*c), frame(R)), True)
    m = material_db.get(material_db.FAB_ALIASES.get(name, name))
    if m.c11_GPa is not None:
        return ElasticMaterial(m.name, rotate_C(cubic_C(m.c11_GPa, m.c12_GPa, m.c44_GPa), frame(R)), True)
    if m.youngs_modulus_GPa is None or m.poisson_ratio is None:
        raise ValueError(f"material {name!r} has no elastic constants in core/material_db.py")
    return ElasticMaterial(m.name, isotropic_C(m.youngs_modulus_GPa, m.poisson_ratio))


def film_eigenstrain(m, sigma0):
    """Isotropic stress-free strain eps0*(1,1,1,0,0,0) of a film of
    ElasticMaterial m (isotropic) that, held flat by its substrate, carries
    the in-plane intrinsic stress sigma0 (GPa, + tensile): sigma0 =
    -E eps0 / (1 - nu). The usual way to load a deposited film's measured
    (wafer-bow) stress into a stress solver."""
    C = m.C
    nu = C[0, 1] / (C[0, 0] + C[0, 1])
    E = C[0, 0] * (1 + nu) * (1 - 2 * nu) / (1 - nu)
    e0 = -sigma0 * (1 - nu) / E
    return np.array([e0, e0, e0, 0.0, 0.0, 0.0])


def intrinsic_stress_GPa(name):
    m = material_db.get(material_db.FAB_ALIASES.get(name, name))
    return m.intrinsic_stress_GPa
