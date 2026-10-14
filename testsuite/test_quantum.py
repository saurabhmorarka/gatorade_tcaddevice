"""Schrodinger-Poisson channel charge (quantum/, plans/quantum_plan.md):
the line solver against closed forms, the subband masses, charge
conservation of the Lambda coupling, and the 1D MOS-cap / double-gate
physics (threshold shift, centroid, volume inversion). All fast."""
import unittest

import numpy as np

from core.mesh import build_mos_grid
from core.params import Material
from mos.mos_params import MOSDevice
from quantum import bands
from quantum.mos1d import QuantumConfig, double_gate_grid, solve_dg, solve_mos, node_volumes
from quantum.schrodinger1d import HB2M_EV_NM2, nc3d, solve_line, subband_density

MAT = Material()
AIRY = np.array([2.33811, 4.08795, 5.52056])


class TestLineSolver(unittest.TestCase):
    def test_airy_triangular_well(self):
        h = 1e-8
        s = h * np.arange(1, 4001)
        for F, mz in ((1e5, 0.916), (1e6, 0.19)):
            E, _ = solve_line(F * s, np.full(len(s), mz), h, 3)
            Ea = AIRY * (HB2M_EV_NM2 / mz) ** (1 / 3) * (F * 1e-7) ** (2 / 3)
            np.testing.assert_allclose(E, Ea, rtol=2e-3)

    def test_infinite_well(self):
        h, t = 1e-8, 5e-7
        s = h * np.arange(1, int(round(t / h)))
        E, psi = solve_line(np.zeros(len(s)), np.full(len(s), 0.916), h, 2)
        np.testing.assert_allclose(E, HB2M_EV_NM2 * np.pi ** 2 * np.array([1, 4]) / 0.916 / (t * 1e7) ** 2, rtol=2e-3)
        np.testing.assert_allclose((psi ** 2).sum(axis=1) * h, 1.0, rtol=1e-12)

    def test_classical_limit_is_nc(self):
        h = 0.2e-7
        s = h * np.arange(1, 1000)
        vals = bands.electron_valleys()
        n, _ = subband_density(np.zeros(len(s)), np.ones(len(s), bool), vals, 0.5, h, 1500, 300.0, 0.0)
        self.assertAlmostEqual(n[len(s) // 2] / nc3d(vals, 300.0), 1.0, delta=3e-3)


class TestBands(unittest.TestCase):
    def test_orientations(self):
        def table(nrm):
            return [(round(v.mz, 3), round(v.md, 3), v.g) for v in bands.electron_valleys(nrm)]
        self.assertEqual(table((0, 1, 0)), [(0.916, 0.19, 2), (0.19, 0.417, 4)])
        self.assertEqual(table((1, 1, 0)), [(0.315, 0.324, 4), (0.19, 0.417, 2)])
        self.assertEqual(table((1, 1, 1)), [(0.258, 0.358, 6)])


class TestMOSGrid(unittest.TestCase):
    def test_oxide_spans_tox(self):
        """build_mos_grid used to drop the -t_ox node (a 1.39 nm oxide for
        tox = 1.5 nm); Gauss's law must now close on the default grid."""
        dev = MOSDevice(Na=1e17, t_ox=1.5e-7)
        g = build_mos_grid(MAT, dev, -1e17)
        self.assertAlmostEqual(g["x"][0], -1.5e-7, delta=1e-15)
        r = solve_mos(g, MAT, dev, -1e17, 1.9)
        V = node_volumes(g["x"])
        semi = ~g["is_oxide"]
        tot = np.sum(((r["p"] - r["n"] + g["Cdop"]) * V)[semi][:-1]) * 1.602176634e-19
        self.assertAlmostEqual(r["Qs"] / tot, 1.0, delta=1e-3)


class TestMOSCap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = MOSDevice(Na=1e18, t_ox=1.5e-7)
        cls.g = build_mos_grid(MAT, cls.dev, -1e18)

    def _vg_at(self, q, N, VG):
        Ni, psi, lam = [], None, None
        for v in VG:
            r = solve_mos(self.g, MAT, self.dev, -1e18, v, q, psi_init=psi, lam_init=lam)
            psi, lam = r["psi"], r["lam"]
            Ni.append(r["Ninv"])
        return np.interp(np.log(N), np.log(Ni), VG)

    def test_threshold_shift(self):
        VG = np.arange(1.1, 1.45, 0.05)
        dvt = self._vg_at(QuantumConfig(), 1e11, VG) - self._vg_at(None, 1e11, VG)
        self.assertTrue(0.070 < dvt < 0.100, dvt)          # 83 mV (quantum.validate_1d)

    def test_charge_conserved_and_centroid(self):
        r = solve_mos(self.g, MAT, self.dev, -1e18, 2.0, QuantumConfig())
        info = r["line_info"]
        V = node_volumes(self.g["x"])
        q = info["q"]
        dd = (r["n"] * V)[info["idx"]]
        sig = q > 1e-3 * q.max()
        # the DD charge of every node equals the line's quantum charge, to
        # the loop tolerance (1 mV in Lambda: exp(1e-3/Vt) - 1 = 4%)
        np.testing.assert_allclose(dd[sig], q[sig], rtol=0.04)
        self.assertGreater(r["centroid"], 0.9e-7)            # 1.1 nm (classical 0.25)
        self.assertLessEqual(r["outer"], 5)

    def test_coarse_mesh(self):
        """The fab's 0.5 nm interface spacing gives the same quantum charge."""
        dev, x0 = self.dev, [-1.5e-7, -1e-7, -0.5e-7]
        xs = [0.0]
        while xs[-1] < 1e-4:
            xs.append(xs[-1] + min(8e-7, 0.5e-7 + 0.35 * xs[-1]))
        x = np.array(x0 + xs)
        ox = x < 0
        xm = 0.5 * (x[1:] + x[:-1])
        gc = dict(x=x, Cdop=np.where(ox, 0, -1e18), eps_edge=np.where(xm < 0, dev.eps_ox, MAT.eps),
                  ni_arr=np.where(ox, 0, MAT.ni), is_oxide=ox, oxide_index=3, gate_oxide_index=0)
        a = solve_mos(self.g, MAT, dev, -1e18, 2.0, QuantumConfig())["Ninv"]
        b = solve_mos(gc, MAT, dev, -1e18, 2.0, QuantumConfig())["Ninv"]
        self.assertAlmostEqual(b / a, 1.0, delta=0.01)


class TestDoubleGate(unittest.TestCase):
    def _profile(self, t):
        g = double_gate_grid(MAT, t, 1.5e-7)
        psi = lam = None
        for v in np.arange(0.0, 1.21, 0.2):
            r = solve_dg(g, MAT, v, QuantumConfig(), psi_init=psi, lam_init=lam)
            psi, lam = r["psi"], r["lam"]
        s = g["is_semi"]
        return g["x"][s], r["n"][s]

    def test_volume_inversion_thin_body(self):
        x, n = self._profile(3e-7)
        self.assertAlmostEqual(x[np.argmax(n)], 1.5e-7, delta=0.1e-7)     # one peak, at the center

    def test_two_peaks_thick_body(self):
        x, n = self._profile(20e-7)
        d = min(x[np.argmax(n)], 20e-7 - x[np.argmax(n)])
        self.assertTrue(0.5e-7 < d < 2e-7, d)                               # off the wall, not centered
        self.assertLess(np.interp(10e-7, x, n) / n.max(), 0.1)


if __name__ == "__main__":
    unittest.main()


class TestDeviceQuantum2D(unittest.TestCase):
    """quantum/device.py on 2D Cartesian meshes through the full drift-
    diffusion Newton (newton_solve_2d(quantum=...))."""

    def test_stack_matches_1d(self):
        """A laterally uniform gate/oxide/substrate stack: every column must
        give the 1D Schrodinger-Poisson inversion charge."""
        import warnings
        from main2d_mosfet_sweep import gate_bc
        from mesh3d.cartesian import build_cartesian_mesh
        from solver2d.newton_solver_qf_2d import newton_solve_2d
        from quantum.device import DeviceQuantum
        Na, dev = 1e18, MOSDevice(Na=1e18, t_ox=1.5e-7)
        ys = [0.0]
        while ys[-1] < 1e-4:
            ys.append(ys[-1] + min(8e-7, 0.5e-7 + 0.35 * ys[-1]))
        y = np.concatenate([[-1.5e-7, -1e-7, -0.5e-7], ys])
        x = np.linspace(0, 10e-7, 6)
        cm = np.where((0.5 * (y[1:] + y[:-1]))[None, :] < 0, 1, 0) * np.ones((len(x) - 1, 1), int)
        mats = [dict(name="si", kind="semi"), dict(name="ox", kind="insulator", eps_r=3.9)]
        m = build_cartesian_mesh([x, y], cm, mats, np.where(y[None, :] >= 0, -Na, 0.0) * np.ones((len(x), 1)), MAT,
                                 face_contacts={"gate": (1, "min"), "body": (1, "max")})
        qd = DeviceQuantum(m, MAT)
        self.assertEqual((len(qd.lines), qd.carrier), (6, "n"))

        def solve(VG, quantum=None, g=None):
            pb, nb, qb = gate_bc(dev, MAT, -Na, VG)
            g = g or {}
            return newton_solve_2d(m, MAT, {"gate": VG, "body": 0.0}, psi_init=g.get("psi"), phin_init=g.get("phin"),
                                   phip_init=g.get("phip"), psi_bc_override={"gate": pb},
                                   phin_bc_override={"gate": nb}, phip_bc_override={"gate": qb}, quantum=quantum)
        ox = y < 0
        ym = 0.5 * (y[1:] + y[:-1])
        g1 = dict(x=y, Cdop=np.where(ox, 0, -Na), eps_edge=np.where(ym < 0, dev.eps_ox, MAT.eps),
                  ni_arr=np.where(ox, 0, MAT.ni), is_oxide=ox, oxide_index=3, gate_oxide_index=0)
        col = np.flatnonzero(np.abs(m.points[:, 0] - 4e-7) < 1e-12)
        st = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for VG in (1.0, 2.0):
                st = solve(VG, None, st)
                rq = qd.solve(lambda sh, g: solve(VG, sh, g), st)
                self.assertTrue(rq["converged"])
                N2 = np.sum(rq["n"][col] * m.cv_area_semi[col] / 2e-7)
                N1 = solve_mos(g1, MAT, dev, -Na, VG, QuantumConfig(n_states=10))["Ninv"]
                self.assertAlmostEqual(N2 / N1, 1.0, delta=0.07 if VG < 1.5 else 0.01)

    def test_double_gate_volume_inversion(self):
        """Gates on both sides of a 5 nm body: the lines are found wall to
        wall (one per column) and the charge peaks at the body center."""
        import warnings
        from mesh3d.cartesian import build_cartesian_mesh
        from solver2d.newton_solver_qf_2d import newton_solve_2d
        from quantum.device import DeviceQuantum
        tb = 5e-7
        x = np.linspace(0, 20e-7, 21)
        y = np.concatenate([[-2e-7], np.linspace(-1.5e-7, 0, 4), np.linspace(0.25e-7, tb, 20),
                            np.linspace(tb + 0.5e-7, tb + 1.5e-7, 3), [tb + 2e-7]])
        xm, ym = 0.5 * (x[1:] + x[:-1])[:, None], 0.5 * (y[1:] + y[:-1])[None, :]
        cm = np.where((ym < -1.5e-7) | (ym > tb + 1.5e-7), 2, np.where((ym < 0) | (ym > tb), 1, 0)) * np.ones_like(xm, int)
        cm = np.where((cm == 2) & (xm < 2e-7), -1, cm)
        mats = [dict(name="si", kind="semi"), dict(name="ox", kind="insulator", eps_r=3.9),
                dict(name="g", kind="conductor", contact="gate")]
        X, Y = np.meshgrid(x, y, indexing="ij")
        cd = np.where((Y >= 0) & (Y <= tb), np.where(X <= 4e-7, 1e20, -1e15), 0.0)
        m = build_cartesian_mesh([x, y], cm, mats, cd, MAT, face_contacts={"source": (0, "min")})
        qd = DeviceQuantum(m, MAT)
        self.assertEqual(len(qd.lines), 19)
        self.assertTrue(all(L.wall_end for L in qd.lines))

        def solve(VG, quantum=None, g=None):
            g = g or {}
            return newton_solve_2d(m, MAT, {"gate": VG, "source": 0.0}, psi_init=g.get("psi"),
                                   phin_init=g.get("phin"), phip_init=g.get("phip"), psi_bc_override={"gate": VG},
                                   phin_bc_override={"gate": VG}, phip_bc_override={"gate": VG}, quantum=quantum)
        st = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for VG in (0.0, 0.3, 0.6):
                st = solve(VG, None, st)
            rq = qd.solve(lambda sh, g: solve(0.6, sh, g), st)
        col = np.flatnonzero((np.abs(m.points[:, 0] - 15e-7) < 1e-12) & ~m.is_insulator)
        yy = m.points[col, 1]
        self.assertTrue(rq["converged"])
        self.assertAlmostEqual(yy[np.argmax(rq["n"][col])], tb / 2, delta=0.3e-7)     # classical: at a wall
        self.assertGreater(st["n"][col][np.argmin(np.abs(yy))], st["n"][col][np.argmin(np.abs(yy - tb / 2))])
