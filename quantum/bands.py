"""Subband masses of the silicon bands for a confinement direction.

Electrons: six <100> ellipsoids (m_l, m_t). For a valley with inverse mass
tensor W in crystal axes and the unit confinement normal n (Stern & Howard,
Phys. Rev. 163, 816 (1967)):

  quantization mass   m_z = 1 / (n . W . n)
  density-of-states   m_d = sqrt((n . W . n) / det W)

(the in-plane inverse mass after eliminating k_z is W_ij - W_i3 W_j3 / W_33,
whose determinant is det W / W_33). Valleys with the same (m_z, m_d) are
merged, their degeneracies summed: (100) gives Delta2 (0.916, 0.19, g=2)
and Delta4 (0.19, 0.417, g=4); (110), a FinFET sidewall, gives
(0.315, 0.324, g=4) and (0.19, 0.417, g=2); (111) gives (0.258, 0.358,
g=6). Computed, not tabulated, so any orientation is the same code.

Holes: parabolic, isotropic-in-plane heavy/light/split-off bands (first
pass; no k.p). Masses are the commonly used effective-mass values for
quantization in (100) Si inversion layers; they are approximate and
orientation-independent here.

Each band also carries its 3D density-of-states mass, so the classical
limit of the subband sum is the band's own Nc / Nv (quantum/lines.py
references the quantum density to it).
"""
from dataclasses import dataclass

import numpy as np

M_L, M_T = 0.916, 0.19          # Si conduction-band ellipsoid masses (m0)
OXIDE_BARRIER_EV = {"n": 3.15, "p": 4.5}   # Si/SiO2 band offsets
OXIDE_MASS = {"n": 0.5, "p": 0.5}          # tunneling mass in SiO2 (m0)


@dataclass(frozen=True)
class Valley:
    mz: float          # quantization mass (m0)
    md: float          # 2D density-of-states mass per valley (m0)
    g: int             # number of valleys (spin not included)
    offset_eV: float = 0.0   # band-edge offset (split-off band)

    @property
    def m3d(self):
        """3D DOS mass of one valley: (m_z m_d^2)^(1/3)."""
        return (self.mz * self.md ** 2) ** (1.0 / 3.0)


def electron_valleys(normal=(0.0, 1.0, 0.0), ml=M_L, mt=M_T, tol=1e-6):
    """Electron valleys for confinement along `normal` (crystal axes)."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    out = []
    for a in np.eye(3):
        W = np.eye(3) / mt + (1.0 / ml - 1.0 / mt) * np.outer(a, a)
        w33 = n @ W @ n
        mz, md = 1.0 / w33, np.sqrt(w33 / np.linalg.det(W))
        for i, v in enumerate(out):
            if abs(v.mz - mz) < tol and abs(v.md - md) < tol:
                out[i] = Valley(mz, md, v.g + 2)
                break
        else:
            out.append(Valley(mz, md, 2))
    return sorted(out, key=lambda v: -v.mz)


def hole_valleys():
    """Heavy, light and split-off holes (parabolic; (100) quantization
    masses, see the module docstring)."""
    return [Valley(0.29, 0.433, 1), Valley(0.20, 0.169, 1), Valley(0.29, 0.29, 1, offset_eV=0.044)]


def valleys(carrier, normal=(0.0, 1.0, 0.0)):
    return electron_valleys(normal) if carrier == "n" else hole_valleys()
