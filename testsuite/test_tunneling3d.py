"""3D local tunneling (btbt/local3d.py) on the octree mesh: the node field,
the Jacobian, and the occupation factor.

  ~/miniconda3/envs/tcad/bin/python -m unittest testsuite.test_tunneling3d
"""
import os
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestLocalTunneling3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fab import flow
            from fab.simulate import build_device, tunneling_model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = build_device(flow.load(os.path.join(ROOT, "configs", "fab_planar_tiny.yaml")),
                                 log=lambda *a: None, mesher="octree")
        except ImportError as e:              # the octree mesher needs the conda env
            raise unittest.SkipTest(str(e))
        cls.d = d
        cls.model = tunneling_model(d, "local")
        cls.P = np.asarray(d["mesh"].points)

    def test_linear_field_exact(self):
        """psi linear in x, y, z: the field is exact at every free silicon
        node, without its normal component on the half-width symmetry plane."""
        a = np.array([3.0e5, -2.0e5, 1.5e5])
        F, _ = self.model.node_field(self.P @ a)
        on = np.abs(self.P[:, 2] - self.P[:, 2].max()) < 1e-12
        want = np.where(on, np.hypot(a[0], a[1]), np.linalg.norm(a))
        free = self.model.free
        self.assertGreater((free & on).sum(), 0)
        np.testing.assert_allclose(F[free], want[free], rtol=1e-10)

    def _state(self):
        """A non-equilibrium state: an applied-like potential and split
        quasi-Fermi levels (depleted and accumulated regions)."""
        P, N = self.P, len(self.P)
        ni = self.model.ni
        rng = np.random.default_rng(0)
        Vt = self.d["mat"].Vt
        psi = Vt * np.log(np.maximum(np.abs(self.d["mesh"].Cdop), 1e10) / np.maximum(ni, 1.0)) \
            * np.sign(self.d["mesh"].Cdop) + 2e5 * P[:, 1]
        phin = 0.3 * (P[:, 0] - P[:, 0].min()) / np.ptp(P[:, 0]) + 1e-3 * rng.standard_normal(N)
        phip = -0.2 * (P[:, 1] - P[:, 1].min()) / np.ptp(P[:, 1]) + 1e-3 * rng.standard_normal(N)
        return psi, phin, phip

    def test_jacobian_matches_finite_differences(self):
        psi, phin, phip = self._state()
        N = len(psi)
        G, _, dU, _ = self.model(psi, phin, phip, jacobian=True)
        U = np.concatenate([psi, phin, phip])
        rng = np.random.default_rng(1)
        for sl in (slice(0, N), slice(N, 2 * N), slice(2 * N, 3 * N)):
            v = np.zeros(3 * N)
            v[sl] = 1e-3 * rng.standard_normal(N)
            eps = 1e-3
            Up, Um = U + eps * v, U - eps * v
            fd = (self.model(Up[:N], Up[N:2 * N], Up[2 * N:])[0] - self.model(Um[:N], Um[N:2 * N], Um[2 * N:])[0]) \
                / (2 * eps)
            an = dU @ v
            self.assertLess(np.linalg.norm(fd - an) / np.linalg.norm(fd), 1e-6)

    def test_no_generation_in_equilibrium(self):
        """np = ni^2 everywhere (phin = phip = 0): both rates vanish, however
        high the field (Hurkx occupation factor on Kane, SRH form for TAT)."""
        psi, _, _ = self._state()
        N = len(psi)
        G, _, _, _ = self.model(psi, np.zeros(N), np.zeros(N))
        F, _ = self.model.node_field(psi)
        from tat.tat import btbt_generation
        self.assertGreater(F[self.model.free].max(), 1e6)          # Kane alone would fire here
        Gk_raw = np.where(self.model.free, btbt_generation(F, self.model._kane())[0], 0.0)
        self.assertGreater(Gk_raw.max(), 1e20)
        # what is left is the round-off of n p - ni^2 (~1e-16 relative)
        self.assertLess(np.abs(G).max(), 1e-12 * Gk_raw.max())


if __name__ == "__main__":
    unittest.main()
