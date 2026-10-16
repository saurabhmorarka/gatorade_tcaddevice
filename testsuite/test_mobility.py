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
from solver2d.newton_solver_qf_2d import (_edge_vsat, _mesh_mobility_nodal, mobility_doping,  # noqa: E402
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


class TestPerMaterialMobility(unittest.TestCase):
    """Each node takes its own material's mobility parameters
    (mesh.node_materials / node_material_index)."""

    def _mesh(self, other):
        Cdop = np.array([1e20, -5e20, -1e17, 0.0, -5e20])
        return SimpleNamespace(Cdop=Cdop, node_materials=[Material(), other],
                               node_material_index=np.array([0, 1, 0, 0, 1]),
                               edges=np.array([[0, 1], [1, 4], [2, 3]]))

    def test_nodes_take_their_material(self):
        base, other = Material(), Material()
        other.masetti_p = tuple(2.0 * c if k in (0, 2, 8) else c for k, c in enumerate(base.masetti_p))
        other.mu_n_max = 999.0                       # unused by masetti: must not matter
        mesh = self._mesh(other)
        mn, mp = _mesh_mobility_nodal(mesh, base)
        N = np.abs(mesh.Cdop)
        on = mesh.node_material_index == 1
        np.testing.assert_allclose(mp[on], mobility_masetti(N[on], other.masetti_p))
        np.testing.assert_allclose(mp[~on], mobility_masetti(N[~on], base.masetti_p))
        np.testing.assert_allclose(mn, mobility_masetti(N, base.masetti_n))

    def test_same_parameters_is_the_single_material_path(self):
        base = Material()
        mesh = self._mesh(Material())
        single = SimpleNamespace(Cdop=mesh.Cdop)
        for a, b in zip(_mesh_mobility_nodal(mesh, base), _mesh_mobility_nodal(single, base)):
            np.testing.assert_array_equal(a, b)

    def test_edge_vsat(self):
        base, other = Material(), Material()
        mesh = self._mesh(other)
        self.assertEqual(_edge_vsat(mesh, base), (base.vsat_n, base.vsat_p))      # equal: the scalars
        other.vsat_p = 6.0e6
        vn, vp = _edge_vsat(mesh, base)
        np.testing.assert_allclose(vp, [0.5 * (base.vsat_p + 6e6), 6e6, base.vsat_p])
        other.beta_p = 2.0
        with self.assertRaises(NotImplementedError):
            _edge_vsat(mesh, base)


class TestMobilityBlock(unittest.TestCase):
    def test_block_sets_fields(self):
        from core.config import _resolve_material_block
        m = _resolve_material_block({"mobility": {"masetti_p": [1, 2, 3, 4, 5, 6, 7, 8, 9], "vsat_p": 7e6}})
        self.assertEqual(m.masetti_p, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        self.assertEqual(m.vsat_p, 7e6)
        with self.assertRaises(ValueError):
            _resolve_material_block({"mobility": {"mu_p_typo": 1}})
        with self.assertRaises(ValueError):
            _resolve_material_block({"mobility": {"masetti_p": [1, 2]}})


if __name__ == "__main__":
    unittest.main()
