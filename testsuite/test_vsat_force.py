"""Velocity-saturation driving forces (solver2d/newton_solver_qf_2d.py
edge_mobility / edge_currents): the analytic Jacobian of the edge currents
against central finite differences for every force, and the physics each
force is meant to have (DEVELOPMENT_LOG 49).

  .venv/bin/python -m unittest testsuite.test_vsat_force
"""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.params import Material                                    # noqa: E402
from solver2d.newton_solver_qf_2d import edge_currents, edge_mobility  # noqa: E402


def _mesh(n_edges=40, seed=0):
    """Random edges between random points (cm), random doping: enough to
    exercise every term; geometry need not be a real mesh."""
    rng = np.random.default_rng(seed)
    N = n_edges + 1
    pts = rng.uniform(0, 20e-7, size=(N, 3))
    edges = np.stack([np.arange(n_edges), np.arange(1, N)], axis=1)
    return SimpleNamespace(points=pts, edges=edges, facet_length=rng.uniform(0.5, 2, n_edges) * 1e-14,
                           facet_length_semi=None, cv_area=np.ones(N) * 1e-21, cv_area_semi=None,
                           Cdop=rng.choice([1e15, -1e18, 1e20], size=N), dEi_arr=None, ni_arr=None)


def _state(mat, N, seed):
    """psi/phin/phip spanning dense and sparse carriers on both sides of the
    hybrid reference density, with fields from ~1e3 to ~1e6 V/cm."""
    rng = np.random.default_rng(seed)
    psi = rng.uniform(-0.6, 0.6, N)
    phin = psi - mat.Vt * np.log(10 ** rng.uniform(12, 20, N) / mat.ni)      # n from 1e12 to 1e20
    phip = psi + mat.Vt * np.log(10 ** rng.uniform(12, 20, N) / mat.ni)
    return psi, phin, phip


def _currents(mesh, mat, psi, phin, phip, force):
    n = mat.ni * np.exp((psi - phin) / mat.Vt)
    p = mat.ni * np.exp((phip - psi) / mat.Vt)
    return edge_currents(mesh, mat, psi, n, p, phin, phip, force, jacobian=True)


class TestVsatForce(unittest.TestCase):
    def setUp(self):
        self.mat = Material()

    def test_jacobian_matches_finite_differences(self):
        mat = self.mat
        for force in (False, True, "efield", "hybrid", "gradqf"):
            for seed in (1, 2, 3):
                mesh = _mesh(seed=seed)
                psi, phin, phip = _state(mat, len(mesh.points), seed)
                In, Ip, J = _currents(mesh, mat, psi, phin, phip, force)
                ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
                h = 1e-7
                # (unknown array, node side, carrier) for the 8 partials, in edge_currents' order
                order = [("psi", ii, 0), ("psi", jj, 0), ("phin", ii, 0), ("phin", jj, 0),
                         ("psi", ii, 1), ("psi", jj, 1), ("phip", ii, 1), ("phip", jj, 1)]
                for k, (var, nodes, carrier) in enumerate(order):
                    vals = {"psi": psi, "phin": phin, "phip": phip}
                    # a node is shared by two consecutive edges of the chain: perturb
                    # every other edge's node at a time so each edge sees one change
                    for parity in (0, 1):
                        sel = (np.arange(len(nodes)) % 2) == parity
                        u = {v: a.copy() for v, a in vals.items()}
                        d = {v: a.copy() for v, a in vals.items()}
                        u[var][nodes[sel]] += h
                        d[var][nodes[sel]] -= h
                        Iu = _currents(mesh, mat, u["psi"], u["phin"], u["phip"], force)[carrier]
                        Id = _currents(mesh, mat, d["psi"], d["phin"], d["phip"], force)[carrier]
                        fd = ((Iu - Id) / (2 * h))[sel]
                        an = J[k][sel]
                        scale = np.maximum(np.abs(fd), np.abs(an)).max() + 1e-300
                        err = np.abs(fd - an).max() / scale
                        self.assertLess(err, 1e-5, f"force={force!r} seed={seed} partial {k}: rel err {err:.2e}")

    def test_force_physics(self):
        """gradqf ignores a current-free field (built-in junction field or a
        vertical gate field); efield does not; hybrid follows gradqf where
        carriers are dense and efield where they are sparse."""
        mat = self.mat
        mesh = SimpleNamespace(points=np.array([[0, 0, 0], [1e-7, 0, 0.0]]), edges=np.array([[0, 1]]),
                               Cdop=np.array([1e17, 1e17]))
        psi = np.array([0.0, 0.1])                                  # 1e6 V/cm across 1 nm
        for dens in (1e19, 1e10):
            phin = np.full(2, psi[0] - mat.Vt * np.log(dens / mat.ni))   # flat phin: no current, n(0) = dens
            n = mat.ni * np.exp((psi - phin) / mat.Vt)
            p = np.zeros(2)
            phip = np.zeros(2)
            mu0 = edge_mobility(mesh, mat, psi, False)[0][0]
            mu_e = edge_mobility(mesh, mat, psi, "efield", n, p, phin, phip)[0][0]
            mu_g = edge_mobility(mesh, mat, psi, "gradqf", n, p, phin, phip)[0][0]
            mu_h = edge_mobility(mesh, mat, psi, "hybrid", n, p, phin, phip)[0][0]
            self.assertLess(mu_e, 0.5 * mu0)                         # 1e6 V/cm saturates under efield
            self.assertAlmostEqual(mu_g / mu0, 1.0, places=6)        # no quasi-Fermi gradient: no saturation
            c = 0.5 * (n[0] + n[1])
            if c > 1e18:
                self.assertAlmostEqual(mu_h / mu0, 1.0, places=3)
            else:
                self.assertAlmostEqual(mu_h / mu_e, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
