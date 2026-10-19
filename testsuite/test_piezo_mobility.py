"""Stress -> mobility (piezoresistance) in the device solver
(mechanics/piezo.py, mechanics/stress_field.py, solver2d/stress_mobility.py,
DEVELOPMENT_LOG 82): the rotated tensor against the textbook (001)/<110>
coefficients, edge-direction invariance, uniform stress against the
analytic d(mu)/mu, the saturating law, the doping factors, the masked
stress sampling and the frame check, zero stress = bit-identical currents,
the SiGe Schottky table material and the PMOS metrics mirror. No I-V
sweeps (one 2D bias point).

  .venv/bin/python -m unittest testsuite.test_piezo_mobility
"""
import os
import sys
import tempfile
import unittest
import warnings
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mechanics import elastic as el                                  # noqa: E402
from mechanics import piezo                                          # noqa: E402
from mechanics.stress_field import StressField                       # noqa: E402

CFG_N = os.path.join(ROOT, "configs", "input_mosfet_2d_stress_nmos.yaml")


def _voigt(S):
    return np.array([S[0, 0], S[1, 1], S[2, 2], S[1, 2], S[0, 2], S[0, 1]])


def _tensor(v):
    xx, yy, zz, yz, xz, xy = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


class TestTensor(unittest.TestCase):
    def test_110_coefficients(self):
        for c in "np":
            P = piezo.pi_tensor(c)
            ref = piezo.pi_110_on_001(c)
            t = np.array([[1.0, 0, 0]])
            for k, comp in ((0, "l"), (1, "v"), (2, "t")):
                s = np.zeros((1, 6))
                s[0, k] = 1.0
                self.assertAlmostEqual(-piezo.linear_change(t, s, P)[0], ref[comp], places=12)
        # holes: the textbook 71.8 / -66.3 (1e-11/Pa), Thompson et al. 2006 Table I
        self.assertAlmostEqual(piezo.pi_110_on_001("p")["l"], 0.718, places=12)
        self.assertAlmostEqual(piezo.pi_110_on_001("p")["t"], -0.663, places=12)

    def test_100_frame_gives_pi11(self):
        P = piezo.pi_tensor("n", el.FRAME_001_100)
        s = np.array([[1.0, 0, 0, 0, 0, 0]])
        self.assertAlmostEqual(-piezo.linear_change(np.array([[1.0, 0, 0]]), s, P)[0], piezo.SMITH["n"][0], 12)

    def test_edge_direction_invariance(self):
        """x depends on the physics only: t -> -t changes nothing, and turning
        the current AND the stress together by a rotation of the device
        frame gives the same x as the unrotated pair (Pi rotated into the
        new frame)."""
        rng = np.random.default_rng(1)
        t = rng.normal(size=(50, 3))
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        sig = rng.normal(size=(50, 6))
        for c in "np":
            P = piezo.pi_tensor(c)
            x = piezo.linear_change(t, sig, P)
            np.testing.assert_allclose(piezo.linear_change(-t, sig, P), x, atol=1e-13)
            # a rotation Q of the device axes: t' = Q t, sigma' = Q sigma Q^T, Pi' = Pi in frame Q R
            a = 0.7
            Q = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
            R = el.frame(el.FRAME_001_110)
            P2 = piezo.pi_tensor(c, Q @ R)
            t2 = t @ Q.T
            s2 = np.array([_voigt(Q @ _tensor(v) @ Q.T) for v in sig])
            np.testing.assert_allclose(piezo.linear_change(t2, s2, P2), x, atol=1e-12)

    def test_hydrostatic_is_isotropic(self):
        """Hydrostatic pressure has no preferred direction: every current
        direction sees -(pi11 + 2 pi12) p."""
        rng = np.random.default_rng(2)
        t = rng.normal(size=(20, 3))
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        for c in "np":
            p11, p12, _ = piezo.SMITH[c]
            x = piezo.linear_change(t, np.tile([1.0, 1, 1, 0, 0, 0], (20, 1)), piezo.pi_tensor(c))
            np.testing.assert_allclose(x, -(p11 + 2 * p12), atol=1e-12)


class TestLaw(unittest.TestCase):
    def test_saturating_slope_and_limits(self):
        for c in "np":
            s = piezo.SATURATION[c]
            h = 1e-6
            d = (piezo.factor(h, c) - piezo.factor(-h, c)) / (2 * h)
            self.assertAlmostEqual(float(d), 1.0, places=6)            # Smith's slope at zero stress
            self.assertAlmostEqual(float(piezo.factor(50.0, c)), s["f_max"], places=9)
            self.assertAlmostEqual(float(piezo.factor(-50.0, c)), s["f_min"], places=9)
            self.assertEqual(float(piezo.factor(0.0, c)), 1.0)
        self.assertEqual(float(piezo.factor(-3.0, "p", "linear")), piezo.F_FLOOR)

    def test_doping_factors(self):
        np.testing.assert_allclose(piezo.kanda_factor([1e14, 1e15]), 1.0, atol=1e-3)
        k = piezo.kanda_factor([1e18, 1e19, 1e20])
        self.assertTrue(np.all(np.diff(k) < 0) and 0.3 < k[-1] < 0.6)
        r = piezo.richter_factor([1e18, 1e20])            # Richter Eq. 20 at 300 K, by hand
        self.assertAlmostEqual(r[0], 1 / (1 + (1e18 / 6e19) ** 0.43 + (1e18 / 7e20) ** 1.6), places=12)
        self.assertTrue(0.4 < r[1] < 0.47)


class TestField(unittest.TestCase):
    def _field(self):
        """2 x 1 pixels: Si (label 1) left with sigma_xx = -1, oxide right with +5."""
        sig = np.zeros((2, 1, 6))
        sig[0, 0, 0], sig[1, 0, 0] = -1.0, 5.0
        return StressField(sig, (0.0, 0.0), 1.0, np.array([[1], [2]]), {1: "Si", 2: "SiO2"})

    def test_masked_sampling(self):
        f = self._field()
        s = f.sample(np.array([[0.5, 0.5], [1.0, 0.5], [1.5, 0.5]]))
        np.testing.assert_allclose(s[:, 0], -1.0)        # only the silicon pixel, even inside the oxide
        s = f.sample(np.array([[1.0, 0.5]]), materials=("Si", "SiO2"))
        self.assertAlmostEqual(s[0, 0], 2.0)              # both: plain interpolation

    def test_save_load(self):
        f = self._field()
        with tempfile.TemporaryDirectory() as d:
            g = StressField.load(f.save(os.path.join(d, "s.npz")))
        np.testing.assert_allclose(g.sigma, f.sigma)
        self.assertEqual(g.materials, f.materials)


def _context_2d(stress=True):
    import main2d_mosfet_sweep as m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m._context(CFG_N, None, None if stress else "off")


class TestDevice2D(unittest.TestCase):
    """The 30 nm planar NMOS of configs/input_mosfet_2d_stress_nmos.yaml."""

    def test_uniform_stress_analytic(self):
        from solver2d.stress_mobility import PiezoModel, edge_geometry, stress_factors
        mesh, mat = _context_2d(False)[:2]
        sig = np.array([-1.0, 0.3, 0.2, 0.0, 0.0, 0.1])            # GPa, uniform
        lab = np.ones((400, 400), np.int32)
        f = StressField(np.broadcast_to(sig, (400, 400, 6)), (-100.0, -100.0), 1.0, lab, {1: "Si"})
        fn, fp, rep = stress_factors(mesh, mat, f, PiezoModel(mode="linear", doping_factor=False))
        on = fn != 1.0
        _, t = edge_geometry(mesh)
        tx = np.abs(t[:, 0]) > 0.999
        self.assertGreater(rep["n_si_edges"], 1000)
        self.assertEqual(rep["frame_mismatch"], 0.0)
        # x edges: 1 - (pi_l sxx + pi_v syy + pi_t szz) - and the in-plane shear drops out
        for c, ff in (("n", fn), ("p", fp)):
            k = piezo.pi_110_on_001(c)
            want = 1 - (k["l"] * sig[0] + k["v"] * sig[1] + k["t"] * sig[2])
            np.testing.assert_allclose(ff[on & tx], max(want, piezo.F_FLOOR), rtol=1e-12)
        # every edge: the tensor law directly
        x = piezo.linear_change(t[on], np.tile(sig, (on.sum(), 1)), piezo.pi_tensor("p"))
        np.testing.assert_allclose(fp[on], np.maximum(1 + x, piezo.F_FLOOR), rtol=1e-12)

    def test_frame_check_catches_a_shift(self):
        from solver2d.stress_mobility import field_from_config, stress_factors
        import yaml
        from mesh2d.config2d import build_domain_from_config
        mesh, mat = _context_2d(False)[:2]
        with open(CFG_N) as fh:
            cfg = yaml.safe_load(fh)
        f = field_from_config(cfg["stress"], build_domain_from_config(cfg), log=None)
        stress_factors(mesh, mat, f)                                  # aligned: passes
        with self.assertRaises(ValueError):
            stress_factors(mesh, mat, f.shifted([0.0, 25.0]))         # 25 nm too deep: silicon edges in the gate

    def test_zero_stress_bit_identical_and_gain(self):
        from main2d_mosfet_sweep import solve_mosfet
        from solver2d.current import contact_current
        from solver2d.newton_solver_qf_2d import edge_currents, set_edge_mobility_factor
        from solver2d.stress_mobility import PiezoModel, apply_stress_mobility
        mesh, mat, dev, Cs, eq, vsat = _context_2d(False)
        rng = np.random.default_rng(0)
        N = len(mesh.points)
        psi = eq["psi"] + rng.normal(scale=1e-3, size=N)
        phin = rng.normal(scale=1e-2, size=N)
        phip = rng.normal(scale=1e-2, size=N)
        n, p = eq["n"], eq["p"]
        ref = edge_currents(mesh, mat, psi, n, p, phin, phip, True)
        f = StressField(np.zeros((300, 300, 6)), (-100.0, -100.0), 1.0, np.ones((300, 300), np.int32), {1: "Si"})
        apply_stress_mobility(mesh, mat, f, PiezoModel(), log=None)
        try:
            self.assertTrue(np.all(mesh.edge_mu0_factor[0] == 1.0))
            got = edge_currents(mesh, mat, psi, n, p, phin, phip, True)
            for a, b in zip(ref, got):
                if a is None:
                    continue
                for u, v in zip(a if isinstance(a, tuple) else (a,), b if isinstance(b, tuple) else (b,)):
                    self.assertTrue(np.array_equal(np.asarray(u), np.asarray(v)))
        finally:
            set_edge_mobility_factor(mesh, "stress", None)
        self.assertIsNone(mesh.edge_mu0_factor)
        # the planar_flow NMOS stress (tensile channel): Id up at one linear bias point
        mesh_s = _context_2d(True)[0]
        self.assertGreater(mesh_s.stress_report["f_n"][1], 1.2)

        def idlin(mesh_):
            r = solve_mosfet(mesh_, mat, dev, Cs, 0.0, 0.05, psi_init=eq["psi"], phin_init=eq["phin"],
                             phip_init=eq["phip"], velocity_saturation=vsat)
            r = solve_mosfet(mesh_, mat, dev, Cs, 0.4, 0.05, psi_init=r["psi"], phin_init=r["phin"],
                             phip_init=r["phip"], velocity_saturation=vsat)
            self.assertTrue(r["converged"])
            return contact_current(mesh_, mat, r, "drain")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = idlin(mesh_s) / idlin(mesh)
        self.assertTrue(1.1 < g < 1.4, g)


class TestPlumbing(unittest.TestCase):
    def test_pmos_metrics_mirror(self):
        from solver2d.mosfet_metrics import mosfet_metrics, mosfet_metrics_signed
        v = np.linspace(0, 1, 21)
        lo = dict(Vds=0.05, Vgs=v, Id=1e-9 * np.exp(v / 0.026 / 1.3) / (1 + np.exp((v - 0.3) / 0.03)) + 1e-5 * v)
        hi = dict(Vds=0.7, Vgs=v, Id=3 * lo["Id"])
        mn = mosfet_metrics(lo, hi, 0.03)
        flip = lambda c: dict(Vds=-c["Vds"], Vgs=-c["Vgs"][::-1], Id=-c["Id"][::-1])  # noqa: E731
        mp = mosfet_metrics_signed(flip(lo), flip(hi), 0.03)
        self.assertTrue(mp["pmos"])
        for k in ("Ion", "Ioff", "SS_lo", "DIBL_mV_per_V"):
            self.assertAlmostEqual(mp[k], mn[k], places=12)
        self.assertAlmostEqual(mp["Vt_cc_lo"], -mn["Vt_cc_lo"], places=12)

    def test_schottky_contact_material(self):
        from contacts.subgrid import contact_material
        from core.config import _resolve_material_block
        from fab.simulate import semiconductor_material
        si = _resolve_material_block({})
        sige = semiconductor_material("SiGe")
        si2 = semiconductor_material("Si")
        mesh = SimpleNamespace(node_materials=[si, si2, sige], node_material_index=np.array([1, 1, 2, 2, 2]))
        self.assertIs(contact_material(mesh, si, np.array([2, 3, 4])), sige)
        self.assertIs(contact_material(mesh, si, np.array([0, 1])), si)       # Si nodes: the base object
        self.assertIs(contact_material(SimpleNamespace(), si, np.array([0])), si)


if __name__ == "__main__":
    unittest.main()
