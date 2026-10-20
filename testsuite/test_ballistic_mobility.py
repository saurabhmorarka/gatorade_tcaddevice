"""Ballistic mobility (solver2d/ballistic_mobility.py, DEVELOPMENT_LOG 84,
docs/ballistic.md): the thermal velocity and k = mu_bal/L against hand
numbers, Matthiessen limits, the identity with Lundstrom's linear-regime
backscattering model, the channel mask, the stress gamma^s scaling, the
edge-current Jacobian with the term on (finite differences), off =
bit-identical, and the 2D device: the linear current drops as Matthiessen
says. No I-V sweeps (a few 2D bias points).

  ~/miniconda3/envs/tcad/bin/python -m unittest testsuite.test_ballistic_mobility
"""
import os
import sys
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.params import Material                                             # noqa: E402
from solver2d import ballistic_mobility as bm                                # noqa: E402
from solver2d.newton_solver_qf_2d import edge_currents, low_field_mobility  # noqa: E402
from testsuite.test_vsat_force import _currents, _mesh, _state               # noqa: E402

CFG_N = os.path.join(ROOT, "configs", "input_mosfet_2d_stress_nmos.yaml")


class TestFormulas(unittest.TestCase):
    def test_thermal_velocity_and_k(self):
        # v_T = sqrt(2kT/(pi m*)): 1.234e7 cm/s for m_t = 0.19 at 300 K
        self.assertAlmostEqual(bm.thermal_velocity(0.19) / 1e7, 1.2343, places=3)
        # k = q v_T/(2kT): 23.9 (electrons), 16.5 (holes 0.40), 20.4 for 0.26 (Sentaurus' 20)
        self.assertAlmostEqual(bm.ballistic_k(0.19), 23.87, places=1)
        self.assertAlmostEqual(bm.ballistic_k(0.40), 16.45, places=1)
        self.assertAlmostEqual(bm.ballistic_k(0.26), 20.4, places=1)
        # v_T ~ T^(1/2), so k ~ T^(-1/2)
        self.assertAlmostEqual(bm.ballistic_k(0.19, 400.0) / bm.ballistic_k(0.19), np.sqrt(300 / 400), places=12)
        self.assertAlmostEqual(bm.ballistic_mobility(15.0, 0.19), 15 * bm.ballistic_k(0.19), places=12)

    def test_matthiessen_limits(self):
        mu = np.array([50.0, 250.0, 1400.0])
        np.testing.assert_array_equal(bm.matthiessen(mu, np.inf), mu)          # long channel: unchanged
        np.testing.assert_array_equal(bm.matthiessen(mu, 300.0, 0.0), mu)      # off the channel mask
        self.assertTrue(np.all(bm.matthiessen(mu, 300.0) < np.minimum(mu, 300.0)))   # never above either
        self.assertLess(bm.matthiessen(250.0, bm.ballistic_mobility(1e-3, 0.19)), 0.03)   # L -> 0: -> 0
        np.testing.assert_allclose(bm.matthiessen(250.0, 250.0), 125.0)

    def test_lundstrom_identity(self):
        """Linear regime, backscattering r = L/(L+lambda), lambda = 2 Vt mu / v_T:
        G = W Q v_T (1-r)/(2 Vt) and mu_app = G L / (W Q) is exactly
        1/(1/mu + 1/mu_bal) with mu_bal = v_T L/(2 Vt)."""
        Vt = bm.KB * 300 / bm.Q
        for m in (0.19, 0.40):
            vT = bm.thermal_velocity(m)
            for mu in (80.0, 250.0, 600.0):
                for L_nm in (10.0, 15.0, 30.0, 100.0, 1000.0):
                    L = L_nm * 1e-7
                    lam = 2 * Vt * mu / vT
                    r = L / (L + lam)
                    mu_app = vT * (1 - r) / (2 * Vt) * L
                    self.assertAlmostEqual(mu_app / bm.matthiessen(mu, bm.ballistic_mobility(L_nm, m)), 1.0,
                                           places=10)

    def test_model_options(self):
        m = bm.BallisticModel.from_options({"enabled": True, "k": 20})
        self.assertEqual((m.k("n"), m.k("p")), (20.0, 20.0))
        m = bm.BallisticModel.from_options({})
        self.assertAlmostEqual(m.k("n"), bm.ballistic_k(0.19))
        self.assertAlmostEqual(m.k("p"), bm.ballistic_k(0.40))
        with self.assertRaises(ValueError):
            bm.BallisticModel.from_options({"kk": 1})


class TestEdgeTerm(unittest.TestCase):
    def test_off_is_bit_identical(self):
        mat = Material()
        mesh = _mesh(seed=4)
        psi, phin, phip = _state(mat, len(mesh.points), 4)
        ref = _currents(mesh, mat, psi, phin, phip, True)
        mesh.edge_mu_ballistic = None
        got = _currents(mesh, mat, psi, phin, phip, True)
        np.testing.assert_array_equal(ref[0], got[0])
        np.testing.assert_array_equal(ref[1], got[1])

    def test_total_placement(self):
        """placement "total" (Sentaurus Eq. 367): no velocity saturation = the
        same as "lowfield"; with it, 1/mu = 1/mu_DD(F) + w/mu_bal, so the
        high-field velocity mu*F still tends to vsat, and the term cuts
        the moderate-field velocity more than the low-field placement."""
        from solver2d.newton_solver_qf_2d import edge_mobility
        mat = Material()
        mesh = _mesh(seed=7)
        E = len(mesh.edges)
        psi, phin, phip = _state(mat, len(mesh.points), 7)
        n = mat.ni * np.exp((psi - phin) / mat.Vt)
        p = mat.ni * np.exp((phip - psi) / mat.Vt)
        w = np.full(E, 1.0)
        res = {}
        for pl in ("lowfield", "total"):
            mesh.edge_mu_ballistic = dict(inv_n=w / 300.0, inv_p=w / 200.0, s=0.5, placement=pl)
            res[pl, False] = edge_mobility(mesh, mat, psi, False, n, p, phin, phip)
            res[pl, True] = edge_mobility(mesh, mat, psi, "efield", n, p, phin, phip)
        np.testing.assert_allclose(res["lowfield", False][0], res["total", False][0], rtol=1e-14)
        mesh.edge_mu_ballistic = None
        mu_dd = edge_mobility(mesh, mat, psi, "efield", n, p, phin, phip)[0]
        np.testing.assert_allclose(res["total", True][0], 1 / (1 / mu_dd + 1 / 300.0), rtol=1e-13)
        self.assertTrue(np.all(res["total", True][0] <= res["lowfield", True][0] * (1 + 1e-12)))

    def test_stress_scaling(self):
        """mu_bal is scaled by gamma^s on each edge: 1/mu0 = 1/(gamma mu) + w/(gamma^s mu_bal)."""
        mat = Material()
        mesh = _mesh(seed=5)
        E = len(mesh.edges)
        mu_n, mu_p = low_field_mobility(mesh, mat)
        g = np.linspace(0.5, 3.0, E)
        mesh.edge_mu0_factors = {"stress": (g, g[::-1].copy())}
        mesh.edge_mu0_factor = (g, g[::-1].copy())
        w = np.linspace(0.0, 1.0, E)
        for s in (0.0, 0.5, 1.0):
            mesh.edge_mu_ballistic = dict(inv_n=w / 300.0, inv_p=w / 200.0, s=s)
            a, b = low_field_mobility(mesh, mat)
            np.testing.assert_allclose(a, 1 / (1 / (g * mu_n) + w / (g ** s * 300.0)), rtol=1e-13)
            np.testing.assert_allclose(b, 1 / (1 / (g[::-1] * mu_p) + w / (g[::-1] ** s * 200.0)), rtol=1e-13)

    def test_jacobian_with_ballistic(self):
        """The ballistic term is static per edge: the analytic edge-current
        Jacobian (velocity saturation on, every force) still matches finite
        differences."""
        mat = Material()
        for force, placement in ((False, "lowfield"), ("hybrid", "lowfield"), ("mqf", "lowfield"),
                                 ("efield", "lowfield"), (False, "total"), ("hybrid", "total"), ("mqf", "total")):
            mesh = _mesh(seed=6, contacts=(0, 9))
            E = len(mesh.edges)
            rng = np.random.default_rng(6)
            mesh.edge_mu0_factor = (rng.uniform(0.6, 2.0, E), rng.uniform(0.6, 2.0, E))
            mesh.edge_mu0_factors = {"stress": mesh.edge_mu0_factor}
            mesh.edge_mu_ballistic = dict(inv_n=rng.uniform(0, 1 / 100, E), inv_p=rng.uniform(0, 1 / 80, E), s=0.5,
                                          placement=placement)
            psi, phin, phip = _state(mat, len(mesh.points), 6)
            In, Ip, J = _currents(mesh, mat, psi, phin, phip, force)
            ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
            h = 1e-7
            order = [("psi", ii, 0), ("psi", jj, 0), ("phin", ii, 0), ("phin", jj, 0),
                     ("psi", ii, 1), ("psi", jj, 1), ("phip", ii, 1), ("phip", jj, 1)]
            for k, (var, nodes, carrier) in enumerate(order):
                vals = {"psi": psi, "phin": phin, "phip": phip}
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
                    err = np.abs(fd - an).max() / (np.maximum(np.abs(fd), np.abs(an)).max() + 1e-300)
                    self.assertLess(err, 1e-5, f"force={force!r} {placement} partial {k}: rel err {err:.2e}")


def _context_2d(ballistic):
    import main2d_mosfet_sweep as m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m._context(CFG_N, None, "off", ballistic=ballistic)


class TestDevice2D(unittest.TestCase):
    """The 30 nm planar NMOS of configs/input_mosfet_2d_stress_nmos.yaml."""

    def test_channel_mask(self):
        mesh = _context_2d("on")[0]
        rep = mesh.ballistic_report
        self.assertAlmostEqual(rep["L_nm"], 30.0, places=6)
        self.assertAlmostEqual(rep["mu_bal_n"], 30 * bm.ballistic_k(0.19), places=6)
        b = mesh.edge_mu_ballistic
        w = b["inv_n"] * rep["mu_bal_n"]
        P = mesh.points * 1e7
        ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
        xm, ym = 0.5 * (P[ii, 0] + P[jj, 0]), 0.5 * (P[ii, 1] + P[jj, 1])
        semi = ~(mesh.is_insulator[ii] | mesh.is_insulator[jj])
        mid = semi & (np.abs(xm - 67.0) < 10) & (ym > 0) & (ym < 10)          # the channel under the gate
        sd = semi & ((xm < 45) | (xm > 89))                                     # n+ S/D, outside the gate
        self.assertGreater(w[mid].min(), 0.99)
        self.assertLess(w[sd].max(), 1e-3)
        self.assertTrue(np.all(w[~semi] == 0.0))
        np.testing.assert_allclose(b["inv_p"] * rep["mu_bal_p"], w, rtol=1e-12, atol=1e-15)

    def test_off_identical_and_linear_current(self):
        """Ballistic off = no term (bit-identical currents); on, the linear
        current at a fixed state drops by the Matthiessen factor of the
        channel edges, and a solved bias point converges with a lower Id."""
        import main2d_mosfet_sweep as m
        off = _context_2d("off")
        on = _context_2d("on")
        self.assertIsNone(getattr(off[0], "edge_mu_ballistic", None))
        mesh, mat, dev, Cs, eq, vsat = on
        st = (eq["psi"], eq.get("n"), eq.get("p"))          # the surface terms are solution-dependent
        mu_off = low_field_mobility(off[0], mat, *st)[0]
        mu_on = low_field_mobility(mesh, mat, *st)[0]
        w = mesh.edge_mu_ballistic["inv_n"] * mesh.ballistic_report["mu_bal_n"]
        np.testing.assert_allclose(mu_on, bm.matthiessen(mu_off, mesh.ballistic_report["mu_bal_n"], w), rtol=1e-12)

        def solve(ctx, Vgs, Vds):
            mesh_, mat_, dev_, Cs_, eq_, vsat_ = ctx
            r = eq_
            for v in np.linspace(0, Vgs, 5)[1:]:
                r = m.solve_mosfet(mesh_, mat_, dev_, Cs_, v, 0.0, psi_init=r["psi"], phin_init=r["phin"],
                                   phip_init=r["phip"], cold_retry=False, velocity_saturation=vsat_)
            r = m.solve_mosfet(mesh_, mat_, dev_, Cs_, Vgs, Vds, psi_init=r["psi"], phin_init=r["phin"],
                               phip_init=r["phip"], cold_retry=False, velocity_saturation=vsat_)
            self.assertTrue(m._converged(r))
            return m._id_per_um(mesh_, mat_, r)

        I_off, I_on = solve(off, 0.8, 0.05), solve(on, 0.8, 0.05)
        # the channel mobility drops ~ (1/mu + 1/mu_bal)^-1 / mu (~0.8 here); the
        # current less, through the series S/D and the vsat-limited part
        self.assertLess(I_on, 0.97 * I_off)
        self.assertGreater(I_on, 0.6 * I_off)


if __name__ == "__main__":
    unittest.main()
