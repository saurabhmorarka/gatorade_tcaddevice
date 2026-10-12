"""Doping-dependent low-field mobility (solver2d/newton_solver_qf_2d.py):
the Masetti model (the default) and the Caughey-Thomas alternative.

    python3 -m unittest testsuite.test_mobility
"""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.params import Material                                          # noqa: E402
from solver2d.newton_solver_qf_2d import (_mesh_mobility_nodal, mobility_doping,  # noqa: E402
                                          mobility_masetti)


class TestMasetti(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_limits_and_shape(self):
        m = self.m
        N = np.logspace(10, 22, 200)
        for coef in (m.masetti_n, m.masetti_p):
            mu = mobility_masetti(N, coef)
            self.assertTrue(np.all(np.diff(mu) < 0))                 # falls with doping
            self.assertAlmostEqual(float(mobility_masetti(np.array(0.0), coef)), coef[8], places=6)
        # undoped: the lattice mobility
        self.assertAlmostEqual(float(mobility_masetti(np.array(1e10), m.masetti_n)), 1417.0, delta=0.5)

    def test_agrees_with_caughey_thomas_when_light(self):
        """Up to ~1e18 both fit the same data: within 5%."""
        m = self.m
        for N in (1e14, 1e15, 1e16, 1e17, 1e18):
            ct_n = mobility_doping(N, m.mu_n_max, m.mu_n_min, m.N_ref_n, m.alpha_n)
            ct_p = mobility_doping(N, m.mu_p_max, m.mu_p_min, m.N_ref_p, m.alpha_p)
            self.assertAlmostEqual(float(mobility_masetti(np.array(N), m.masetti_n)) / ct_n, 1.0, delta=0.05)
            self.assertAlmostEqual(float(mobility_masetti(np.array(N), m.masetti_p)) / ct_p, 1.0, delta=0.05)

    def test_heavy_doping_values(self):
        """Masetti's heavy-doping values (worked by hand from the formula;
        DEVELOPMENT_LOG 55): electrons 108 / 61 cm^2/Vs at 1e19 / 1e20 (As),
        holes 73 / 50 (B)."""
        m = self.m
        mu = lambda N, c: float(mobility_masetti(np.array(N), c))
        self.assertAlmostEqual(mu(1e19, m.masetti_n), 108.0, delta=0.5)
        self.assertAlmostEqual(mu(1e20, m.masetti_n), 60.9, delta=0.5)
        self.assertAlmostEqual(mu(1e19, m.masetti_p), 73.2, delta=0.5)
        self.assertAlmostEqual(mu(1e20, m.masetti_p), 49.9, delta=0.5)

    def test_model_switch(self):
        mesh = SimpleNamespace(Cdop=np.array([1e20, -1e17, 0.0]))
        m = Material()
        mn, mp = _mesh_mobility_nodal(mesh, m)
        np.testing.assert_allclose(mn, mobility_masetti(np.abs(mesh.Cdop), m.masetti_n))
        m.doping_mobility = "caughey_thomas"
        mn, mp = _mesh_mobility_nodal(mesh, m)
        np.testing.assert_allclose(mn, mobility_doping(np.abs(mesh.Cdop), m.mu_n_max, m.mu_n_min, m.N_ref_n,
                                                       m.alpha_n))
        m.doping_mobility = "arora"
        with self.assertRaises(ValueError):
            _mesh_mobility_nodal(mesh, m)


if __name__ == "__main__":
    unittest.main()
