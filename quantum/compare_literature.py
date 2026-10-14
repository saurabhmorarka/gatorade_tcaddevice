"""Quantum Vt shift vs substrate doping against a published
Schrodinger-Poisson curve.

Reference: Jallepalli et al., IEEE TED 44, 297 (1997), as replotted in
Asenov, Brown, Watling, "The use of quantum potentials for confinement in
semiconductor devices", Nanotech/ICCN 2002, Fig. 1 (values read off the
plot, +-5 mV). The figure does not state tox or the threshold criterion;
the best match is tox ~ 5 nm with a hard wall (no oxide penetration, the
usual choice of 1990s Schrodinger-Poisson solvers), at a constant
inversion charge of 3e11 cm^-2 (DEVELOPMENT_LOG 70).

  python3 -m quantum.compare_literature
"""
import numpy as np

from core.mesh import build_mos_grid
from core.params import Material
from mos.mos_params import MOSDevice
from quantum.mos1d import QuantumConfig, sweep_mos

JALLEPALLI_DVT = {1e17: 0.033, 2.5e17: 0.057, 5e17: 0.082, 7.5e17: 0.104, 1e18: 0.125,
                  2e18: 0.190, 3e18: 0.245, 5e18: 0.330}         # V


def vg_at(res, N):
    Ni = np.array([r["Ninv"] for r in res])
    return float(np.interp(np.log(N), np.log(Ni), [r["VG"] for r in res]))


def dvt(Na, tox=5e-7, N=3e11, hard_wall=True, mat=None):
    mat = mat or Material()
    dev = MOSDevice(Na=Na, t_ox=tox)
    g = build_mos_grid(mat, dev, -Na)
    VG = np.arange(0.3, 4.5, 0.02)
    cl = sweep_mos(g, mat, dev, -Na, VG)
    qm = sweep_mos(g, mat, dev, -Na, VG, QuantumConfig(hard_wall=hard_wall))
    return vg_at(qm, N) - vg_at(cl, N)


def main():
    print("Na        Jallepalli   ours (tox 5 nm, hard wall)   ours (penetration)")
    for Na, ref in JALLEPALLI_DVT.items():
        a, b = dvt(Na), dvt(Na, hard_wall=False)
        print(f"{Na:8.1e}  {ref * 1e3:6.0f} mV   {a * 1e3:6.0f} mV ({(a / ref - 1) * 100:+4.0f}%)"
              f"           {b * 1e3:6.0f} mV")


if __name__ == "__main__":
    main()
