"""Inversion-layer mobility stack (solver2d/surface_mobility.py,
docs/mobility.md, DEVELOPMENT_LOG 83): the model formulas against hand
numbers, the normal field (least-squares gradient exact for a linear
potential on 2D and 3D meshes; the D-continuity interface value against the
analytic MOS surface field), the orientation classifier on a fin, the
Matthiessen order with the stress factor, the interface charge, the lagged
Newton converging to a self-consistent state, and "off" leaving the solver
bit for bit unchanged.

    python3 -m unittest testsuite.test_surface_mobility
"""
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import _resolve_material_block                      # noqa: E402
from core.params import EPS0, Q                                       # noqa: E402
from solver2d import surface_mobility as sm                           # noqa: E402

MAT = _resolve_material_block({})


def fin_mesh(h=0.5):
    """A 3D rectilinear fin: Si 6 nm wide (z 2..8 nm), oxide on top (y 0..2)
    and on the sides, a gate metal layer above, the body contact at the
    bottom face (y = 14 nm). y points down (the fab frame: y = depth)."""
    from mesh3d.cartesian import CONDUCTOR, INSULATOR, SEMI, build_cartesian_mesh
    x = np.arange(0.0, 6.0 + 1e-9, 1.0)
    y = np.arange(-1.0, 14.0 + 1e-9, h)
    z = np.arange(0.0, 10.0 + 1e-9, h)
    xc, yc, zc = (0.5 * (a[:-1] + a[1:]) for a in (x, y, z))
    Y, Z = np.meshgrid(yc, zc, indexing="ij")
    lab2 = np.where(Y < 0.0, 2, np.where((Y > 2.0) & (Z > 2.0) & (Z < 8.0), 0, 1))
    lab2 = np.where((Y > 10.0), 0, lab2)                     # the substrate under the fin
    lab = np.broadcast_to(lab2, (len(xc),) + lab2.shape).copy()
    mats = [dict(name="si", kind=SEMI, material=MAT), dict(name="ox", kind=INSULATOR, eps_r=3.9),
            dict(name="gate", kind=CONDUCTOR, contact="gate")]
    net = np.full((len(x), len(y), len(z)), -1e16)
    return build_cartesian_mesh([a * 1e-7 for a in (x, y, z)], lab, mats, net, MAT,
                                face_contacts={"body": (1, "max")})


def wrapped_fin_mesh(h=0.5):
    """A 3D fin with a wrap-around gate over STI: Si fin z 3..7 nm from
    y 2 nm down into the substrate (y > 10), a 1 nm gate oxide on its top
    and sidewalls down to the STI top (y 6), the gate metal around it, STI
    oxide below y 6 beside the sub-fin. The gate electrode on the STI top
    sits 4 nm above the substrate surface (y 10)."""
    from mesh3d.cartesian import CONDUCTOR, INSULATOR, SEMI, build_cartesian_mesh
    x = np.arange(0.0, 4.0 + 1e-9, 1.0)
    y = np.arange(0.0, 14.0 + 1e-9, h)
    z = np.arange(0.0, 10.0 + 1e-9, h)
    xc, yc, zc = (0.5 * (a[:-1] + a[1:]) for a in (x, y, z))
    Y, Z = np.meshgrid(yc, zc, indexing="ij")
    si = ((Y > 2.0) & (Z > 3.0) & (Z < 7.0)) | (Y > 10.0)
    ox = (Y > 6.0) | ((Y > 1.0) & (Z > 2.0) & (Z < 8.0))
    lab2 = np.where(si, 0, np.where(ox, 1, 2))
    lab = np.broadcast_to(lab2, (len(xc),) + lab2.shape).copy()
    mats = [dict(name="si", kind=SEMI, material=MAT), dict(name="ox", kind=INSULATOR, eps_r=3.9),
            dict(name="gate", kind=CONDUCTOR, contact="gate")]
    net = np.full((len(x), len(y), len(z)), -1e16)
    return build_cartesian_mesh([a * 1e-7 for a in (x, y, z)], lab, mats, net, MAT,
                                face_contacts={"body": (1, "max")})


def mosfet_2d(mode="fast"):
    import main2d_mosfet_sweep as m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m._context(m.DEFAULT_PATH, None, None, mode, ballistic="off")      # (ballistic: test_ballistic_mobility.py)


class TestFormulas(unittest.TestCase):
    def test_channel_direction_sets_the_sidewall_family(self):
        """(001) wafer: a <110> channel puts the FinFET side walls (device z) on (110) and the
        fin top (device y) on (100); a <100> channel (45 degrees rotated) makes the side walls (100)."""
        from mechanics.elastic import channel_frame
        n = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        w110 = sm.orientation_weights(n, 0.0, channel_frame(110))
        w100 = sm.orientation_weights(n, 0.0, channel_frame(100))
        np.testing.assert_array_equal(w110, [[0, 1, 0], [1, 0, 0]])
        np.testing.assert_array_equal(w100, [[1, 0, 0], [1, 0, 0]])
        with self.assertRaises(ValueError):
            channel_frame(111)

    def test_lombardi_hand_numbers(self):
        """Plan 1.2: electrons N 1e17, F 1 MV/cm: mu_ac ~ 47.5 + 580*1e17^0.125/100
        = 821, mu_sr = 582; holes 0.5 MV/cm: mu_ac ~ 20 + 2947*1e17^0.0317/79.4."""
        F, N = np.array([1e6]), np.array([1e17])
        inv = sm.lombardi(F, N, 0.0, sm.LOMBARDI["n"])
        mu_ac = 4.75e7 / 1e6 + 580.0 * (1e17 + 1) ** 0.125 / 100.0
        self.assertAlmostEqual(mu_ac, 820.9, delta=1.0)
        np.testing.assert_allclose(inv, 1 / mu_ac + 1 / 582.0 + 1e18 / 5.82e30, rtol=1e-6)
        F = np.array([5e5])
        inv = sm.lombardi(F, N, 0.0, sm.LOMBARDI["p"])
        mu_ac = 9.925e6 / 5e5 + 2947.0 * (1e17 + 1) ** 0.0317 / np.cbrt(5e5)
        np.testing.assert_allclose(inv, 1 / mu_ac + 2.5e11 / 2.0546e14 + 1.25e17 / 2.0546e30, rtol=1e-6)

    def test_darwish_a_star(self):
        """The Darwish A* = A + alpha (n + p) / (N + N1)^nu lowers mu_sr at high density."""
        F, N = np.array([1e6]), np.array([1e15])
        lo = sm.lombardi(F, N, np.array([0.0]), sm.DARWISH["n"])
        hi = sm.lombardi(F, N, np.array([1e20]), sm.DARWISH["n"])
        self.assertGreater(hi[0], lo[0])
        A = 2.58 + 6.85e-21 * 1e20 / (1e15 + 1) ** 0.0767
        mu_ac = 3.61e7 / 1e6 + 1.70e4 * (1e15 + 1) ** 0.0233 / 100.0
        np.testing.assert_allclose(hi, 1 / mu_ac + 1e6 ** A / 3.58e18, rtol=1e-6)

    def test_klaassen_limits(self):
        """Undoped: the lattice mobility; majority carriers at 1e15-1e18 within
        15% of Masetti (both fit the same bulk data); carrier screening only
        raises the mobility."""
        from solver2d.newton_solver_qf_2d import mobility_masetti
        z = np.zeros(1)
        self.assertAlmostEqual(float(sm.klaassen(z, z, z + 1e10, z + 1e10, "n")[0]), 1417.0, delta=2.0)
        self.assertAlmostEqual(float(sm.klaassen(z, z, z + 1e10, z + 1e10, "p")[0]), 470.5, delta=1.0)
        for N in (1e15, 1e16, 1e17, 1e18):
            Nd = np.array([N])
            mn = sm.klaassen(Nd, z, Nd, z + 1e20 / N, "n")[0]
            mp = sm.klaassen(z, Nd, z + 1e20 / N, Nd, "p")[0]
            self.assertAlmostEqual(mn / mobility_masetti(Nd, MAT.masetti_n)[0], 1.0, delta=0.15)
            self.assertAlmostEqual(mp / mobility_masetti(Nd, MAT.masetti_p)[0], 1.0, delta=0.15)
            screened = sm.klaassen(z, Nd, Nd * 100, z, "n")[0]     # electrons in p-type, inversion density
            unscreened = sm.klaassen(z, Nd, z + 1e5, z, "n")[0]
            self.assertGreater(screened, unscreened)

    def test_coulomb_terms(self):
        F = np.array([5e5])
        a = sm.interface_coulomb(F, 1e11, np.array([1e18]))
        b = sm.interface_coulomb(F, 1e12, np.array([1e18]))
        c = sm.interface_coulomb(F, 1e12, np.array([1e19]))
        self.assertGreater(b[0], a[0])          # more charge, more scattering
        self.assertLess(c[0], b[0])             # screened by the inversion carriers
        self.assertLess(sm.interface_coulomb(np.array([1e3]), 1e12, np.array([1e18]))[0], 1e-3 * b[0])   # flat band
        # Sentaurus Eq. 311 by hand, Table 63 (mu1 40, c_trans 1e18): nu1 = 1 in the screening term,
        # nu2 = 0.5 in the prefactor (the first version had them swapped)
        x, c0, F0 = 5.0, 2e18, 4e5
        mu = 40.0 / x ** 0.5 * (1.0 + (c0 / (1e18 * x)) ** 1.5)
        f = 1.0 - np.exp(-(F0 / 2e5) ** 2)
        got = sm.interface_coulomb(np.array([F0]), x * 1e11, np.array([c0]), prm=sm.COULOMB_IC_SENTAURUS)
        np.testing.assert_allclose(got, f / mu, rtol=1e-12)
        # screening: mu_C rises monotonically with the carrier density, for both parameter sets
        cc = np.logspace(15, 20, 26)
        for prm in (sm.COULOMB_IC, sm.COULOMB_IC_SENTAURUS):
            inv = sm.interface_coulomb(np.full(26, 1e6), 3e11, cc, prm=prm)
            self.assertTrue(np.all(np.diff(inv) < 0))
        r = sm.remote_highk(np.array([1e6]), np.array([1e15]), np.array([1e19]))
        self.assertTrue(1 / 2000 < r[0] < 1 / 100)

    def test_orientation_classifier(self):
        """Device frame (001)/<110>: y (the fin top, planar surface) is (100),
        z and x (the fin sidewalls, sheet edges) are (110), the fin-corner
        diagonal is nearest (111)."""
        n = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]], float)
        self.assertEqual(list(np.argmax(sm.family_cosines(n), axis=1)), [0, 1, 1, 2])
        w = sm.orientation_weights(n, 0.02)
        np.testing.assert_allclose(w.sum(axis=1), 1.0)
        self.assertGreater(w[0, 0], 0.99)
        self.assertGreater(w[1, 1], 0.99)


class TestGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fin = fin_mesh()
        cls.fin_sm = sm.SurfaceMobility(cls.fin, MAT, sm.MobilityModel(mode="fast"))

    def test_fin_orientation_and_distance(self):
        g = self.fin_sm.geo
        P = self.fin.points * 1e7
        top = g["is_interface"] & (np.abs(P[:, 1] - 2.0) < 1e-6) & (P[:, 2] > 2.6) & (P[:, 2] < 7.4)
        side = g["is_interface"] & (np.abs(P[:, 2] - 2.0) < 1e-6) & (P[:, 1] > 2.6) & (P[:, 1] < 9.4)
        self.assertTrue(top.any() and side.any())
        fam = np.full(len(P), -1)
        fam[self.fin_sm.act] = self.fin_sm.family
        self.assertTrue(np.all(fam[top] == 0))                  # (100) top
        self.assertTrue(np.all(fam[side] == 1))                 # (110) sidewall
        np.testing.assert_allclose(np.abs(g["normal"][top][:, 1]), 1.0)
        np.testing.assert_allclose(np.abs(g["normal"][side][:, 2]), 1.0)
        # a node 1.5 nm inside the sidewall, mid height: 1.5 nm from it
        k = np.argmin(np.linalg.norm(P - [3.0, 6.0, 3.5], axis=1))
        self.assertAlmostEqual(g["dist"][k] * 1e7, 1.5, places=6)
        self.assertAlmostEqual(self.fin_sm.D[k], np.exp(-1.5e-7 / 1e-6), places=9)

    def test_interface_charge_gate_only(self):
        """interface_charge_where: gate (the default) charges the fin top and
        the sidewalls under the gate oxide, not the sub-fin / STI sidewalls
        nor the substrate under the STI (the gate electrode 4 nm above it);
        all charges every interface node. The Coulomb term follows the
        charge (an active node takes its nearest interface node's)."""
        mesh = wrapped_fin_mesh()
        P = mesh.points * 1e7
        gate = sm.SurfaceMobility(mesh, MAT, sm.MobilityModel(mode="fast"))
        g = gate.geo
        ifn = g["is_interface"]
        top = ifn & (np.abs(P[:, 1] - 2.0) < 1e-6) & (P[:, 2] > 3.4) & (P[:, 2] < 6.6)
        side = ifn & (np.abs(P[:, 2] - 3.0) < 1e-6) & (P[:, 1] > 2.4) & (P[:, 1] < 5.6)
        subfin = ifn & (np.abs(P[:, 2] - 3.0) < 1e-6) & (P[:, 1] > 6.9) & (P[:, 1] < 9.6)
        sub_top = ifn & (np.abs(P[:, 1] - 10.0) < 1e-6) & ((P[:, 2] < 2.6) | (P[:, 2] > 7.4))
        for m in (top, side, subfin, sub_top):
            self.assertTrue(m.any())
        self.assertTrue(gate.charged[top].all() and gate.charged[side].all())
        self.assertFalse(gate.charged[subfin].any() or gate.charged[sub_top].any())
        self.assertAlmostEqual(np.percentile(gate.gate_hit[np.isfinite(gate.gate_hit)], 5) * 1e7, 1.0, delta=0.3)
        allm = sm.SurfaceMobility(mesh, MAT, sm.MobilityModel.from_options(
            dict(mode="fast", interface_charge_where="all")))
        np.testing.assert_array_equal(allm.charged, ifn)
        try:
            sm.attach_mobility(mesh, MAT, dict(mode="fast"))
            fc = mesh.fixed_charge
            np.testing.assert_allclose(fc, 3e11 * g["area"] * gate.charged)
            self.assertTrue(np.all(fc[subfin | sub_top] == 0) and np.all(fc[top | side] > 0))
        finally:
            sm.attach_mobility(mesh, MAT, "off")
        # Coulomb: on near the gated sidewall, off near the sub-fin
        a = gate.act
        near = g["nearest"][a]
        self.assertTrue(np.all(gate.ic_mask[gate.charged[near]] == 1.0))
        self.assertTrue(np.all(gate.ic_mask[subfin[near]] == 0.0))
        with self.assertRaises(ValueError):
            sm.MobilityModel.from_options(dict(mode="fast", interface_charge_where="sti"))

    def test_lsq_gradient_exact_for_linear_psi(self):
        """F = |grad psi . n| exactly for a linear psi, at every active node
        whose value comes from the least-squares gradient (2D triangle mesh
        and 3D rectilinear mesh)."""
        mesh2 = mosfet_2d("off")[0]
        for mesh in (self.fin, mesh2):
            s = sm.SurfaceMobility(mesh, MAT, sm.MobilityModel(mode="fast", interface_field="lsq"))
            P = mesh.points
            gvec = np.array([3e4, -2e5, 7e4][:P.shape[1]])
            psi = P @ gvec
            F = s.normal_field(psi)
            a = s.act
            want = np.abs(s.geo["normal"][a] @ gvec)
            np.testing.assert_allclose(F[a], np.sqrt(want ** 2 + 1e6), rtol=1e-6)

    def test_dcontinuity_equals_analytic_surface_field(self):
        """On the 1D MOS capacitor the interface value (the oxide-side flux
        / eps_si) is the solution's total semiconductor charge / eps_si
        (Gauss, exact at any bias), and in depletion / weak inversion the
        analytic surface field of the solved surface potential (0.5%; in
        strong classical inversion the 1-Angstrom grid itself is off the
        analytic profile, but Gauss still holds)."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "scripts", "mobility"))
        from universal import mesh_1d
        from core.mesh import build_mos_grid
        from mos.mos_params import MOSDevice
        from quantum.mos1d import solve_mos
        Na = 1e17
        dev = MOSDevice(Na=Na, t_ox=2e-7)
        g = build_mos_grid(MAT, dev, -Na)
        # the 1D solve carries no sheet charge: no interface-charge field offset either
        s = sm.SurfaceMobility(mesh_1d(g, MAT), MAT, sm.MobilityModel(mode="fast", interface_charge=0.0))
        oi = g["oxide_index"]
        self.assertTrue(s.geo["dcont"][oi])
        Vt = MAT.Vt
        for VG in (0.4, 1.0, 2.0):
            st = solve_mos(g, MAT, dev, -Na, VG)
            psi = st["psi"]
            F = s.normal_field(psi)[oi]
            self.assertAlmostEqual(F * MAT.eps / abs(st["Qs"]), 1.0, delta=1e-4)   # Qs: mos_solver tolerance
            if VG < 1.5:
                ps = psi[oi] - psi[-1]
                n0p0 = (MAT.ni / Na) ** 2
                Fs = np.sqrt(2 * Q * Na / MAT.eps) * np.sqrt(
                    Vt * np.exp(-ps / Vt) + ps - Vt + n0p0 * (Vt * np.exp(ps / Vt) - ps - Vt))
                self.assertAlmostEqual(F / Fs, 1.0, delta=1e-2)

    def test_interface_area(self):
        """The interface area per node sums to the interface: the fin top
        and both sidewalls (6 nm x 6 nm + 2 x 8 nm x 6 nm) plus the substrate
        top beside the fin (2 x 2 nm x 6 nm). Flat faces are exact; the
        corner nodes (normal on the bisector) are approximate: 5%."""
        g = self.fin_sm.geo
        area = g["area"].sum() * 1e14
        self.assertAlmostEqual(area / (6 * 6 + 2 * 8 * 6 + 2 * 2 * 6), 1.0, delta=0.05)
        P = self.fin.points * 1e7
        top = g["is_interface"] & (np.abs(P[:, 1] - 2.0) < 1e-6) & (P[:, 2] > 2.1) & (P[:, 2] < 7.9)
        self.assertAlmostEqual(g["area"][top].sum() * 1e14, 6 * 5.5, delta=1e-6)    # 11 nodes z 2.5..7.5, 0.5 nm each


class TestSolver(unittest.TestCase):
    def test_off_is_bit_identical(self):
        """attach_mobility(off) leaves no model: the edge currents are the
        same bits as a mesh that never had one."""
        from solver2d.newton_solver_qf_2d import edge_currents
        mesh, mat, _, _, eq, _ = mosfet_2d("off")
        st = [eq[k] for k in ("psi", "n", "p", "phin", "phip")]
        a = edge_currents(mesh, mat, *st, True, jacobian=True)
        sm.attach_mobility(mesh, mat, "off")
        b = edge_currents(mesh, mat, *st, True, jacobian=True)
        for x, y in zip(a[:2], b[:2]):
            np.testing.assert_array_equal(x, y)
        for x, y in zip(a[2], b[2]):
            np.testing.assert_array_equal(x, y)
        self.assertIsNone(getattr(mesh, "fixed_charge", None))

    def test_matthiessen_order_with_stress(self):
        """low_field_mobility = (edge mean of the nodal Matthiessen sum) x the
        static stress factor: the factor scales the surface-limited mobility,
        not the bulk one."""
        from solver2d.newton_solver_qf_2d import low_field_mobility, set_edge_mobility_factor
        mesh, mat, _, _, eq, _ = mosfet_2d("fast")
        s = mesh.surface_mobility
        psi, n, p = eq["psi"], eq["n"], eq["p"]
        psi = psi + 0.0
        st = s.state(mat, psi, n, p)
        mu_n, mu_p, _ = s.nodal(mat, psi, n, p)
        ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
        np.testing.assert_allclose(st.mu_n, 0.5 * (mu_n[ii] + mu_n[jj]))
        a = s.act
        mb_n = s.mu_b[0]
        F = s.normal_field(psi)
        # the default stack: Lombardi + the interface-charge Coulomb term (Qit 3e11)
        self.assertEqual(s.ic, 3e11)
        add = s.orient["n"] * sm.lombardi(F[a], s.Ntot[a], n[a] + p[a], s.sets["n"]) \
            + sm.interface_coulomb(F[a], s.ic, n[a], s.T, s.ic_prm)
        want = 1.0 / (1.0 / mb_n[a] + s.D[a] * add)
        np.testing.assert_allclose(mu_n[a], want, rtol=1e-12)
        f = np.linspace(0.5, 2.0, len(mesh.edges))
        try:
            set_edge_mobility_factor(mesh, "stress", f, 1.0 / f)
            got_n, got_p = low_field_mobility(mesh, mat, psi, n, p)
        finally:
            set_edge_mobility_factor(mesh, "stress", None)
        np.testing.assert_allclose(got_n, st.mu_n * f, rtol=1e-12)
        np.testing.assert_allclose(got_p, st.mu_p / f, rtol=1e-12)

    def test_surface_degradation_and_self_consistency(self):
        """2D NMOS, Vds 50 mV, Vgs 1.2 V: the lagged Newton converges; the
        mobility at the converged state is self-consistent (re-evaluated, it
        moves < the tolerance), the drain current is below the bulk-mobility
        current, and the interface edges carry less than 60% of mu_bulk."""
        from main2d_mosfet_sweep import continuation, solve_mosfet
        from solver2d.current import contact_current
        out = {}
        for mode in ("off", "fast"):
            mesh, mat, dev, Cs, eq, vsat = mosfet_2d(mode)

            def at(p_, g, mesh=mesh, mat=mat, dev=dev, Cs=Cs, vsat=vsat):
                return solve_mosfet(mesh, mat, dev, Cs, p_, 0.05, psi_init=g["psi"], phin_init=g["phin"],
                                    phip_init=g["phip"], cold_retry=False, velocity_saturation=vsat)
            ramp, _, _ = continuation(lambda p_, g: solve_mosfet(mesh, mat, dev, Cs, 0.0, p_, psi_init=g["psi"],
                                                                  phin_init=g["phin"], phip_init=g["phip"],
                                                                  cold_retry=False, velocity_saturation=vsat),
                                      0.0, eq, [0.05])
            states, _, _ = continuation(at, 0.0, ramp[0.05], [1.2])
            r = states[1.2]
            self.assertTrue(r["converged"])
            out[mode] = contact_current(mesh, mat, r, "drain")
            if mode == "fast":
                s = mesh.surface_mobility
                ms = s.state(mat, r["psi"], r["n"], r["p"])
                self.assertLessEqual(r["mu_change"], s.tol)
                ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
                on = s.geo["is_interface"][ii] & s.geo["is_interface"][jj] & s.weight
                mb = 0.5 * (s.mu_b[0][ii] + s.mu_b[0][jj])
                self.assertLess(np.median(ms.mu_n[on] / mb[on]), 0.6)
        self.assertLess(out["fast"], 0.85 * out["off"])

    def test_interface_charge_electrostatics(self):
        """A positive interface charge on a p-substrate NMOS raises the
        surface potential at equilibrium (Gauss), and the fixed charge per
        node is the sheet density times the node's interface area."""
        import main2d_mosfet_sweep as m
        from main2d_mosfet_sweep import solve_mosfet
        mesh, mat, dev, Cs, eq0, vsat = mosfet_2d("off")
        rep = sm.attach_mobility(mesh, mat, dict(mode="fast", interface_charge=5e11))
        try:
            self.assertEqual(rep["interface_charge"], 5e11)
            fc = mesh.fixed_charge
            np.testing.assert_allclose(fc, 5e11 * mesh.surface_mobility.geo["area"])
            eq = solve_mosfet(mesh, mat, dev, Cs, 0.0, 0.0, velocity_saturation=vsat)
            self.assertTrue(eq["converged"])
            k = np.flatnonzero(fc > 0)
            self.assertGreater(np.mean(eq["psi"][k] - eq0["psi"][k]), 0.01)
        finally:
            sm.attach_mobility(mesh, mat, "off")
            m._CTX.clear()

    def test_interface_coulomb_screened_and_monotonic(self):
        """Regression (DEVELOPMENT_LOG 83): 2D NMOS, Qit 1e12, Vds 50 mV,
        Vgs 0-1.5 V. The first interface-Coulomb version gave mu_eff 200 ->
        380 -> 50 and a falling Id (negative gm). Now: Id rises at every
        step, the Coulomb-limited mobility of the mid-channel column rises
        with N_inv (screening), and the Coulomb loss above N_inv 2e12 is
        shrinks with N_inv against the same run without Qit at the same N_inv
        (about 20% left at 2e12 for this large Qit; 3e11 leaves 3.5%)."""
        import main2d_mosfet_sweep as m
        from main2d_mosfet_sweep import continuation, solve_mosfet
        from solver2d.current import contact_current
        mesh, mat, dev, Cs, eq0, vsat = mosfet_2d("off")
        gate = next(c for c in mesh.domain.contacts if c.name == "gate")
        x0, x1 = gate.x_range_cm
        P = mesh.points
        col = ((P[:, 0] > x0 + 0.25 * (x1 - x0)) & (P[:, 0] < x1 - 0.25 * (x1 - x0)) & (P[:, 1] >= 0.0)
               & (P[:, 1] < 30e-7) & ~mesh.is_insulator)
        VG = [round(v, 2) for v in np.arange(0.0, 1.51, 0.1)]
        res = {}
        try:
            for qit in (0.0, 1e12):
                sm.attach_mobility(mesh, mat, dict(mode="fast", interface_charge=qit))
                S = mesh.surface_mobility

                def solve(vg, vd, g):
                    return solve_mosfet(mesh, mat, dev, Cs, vg, vd, psi_init=g["psi"], phin_init=g["phin"],
                                        phip_init=g["phip"], cold_retry=False, velocity_saturation=vsat)
                st0, _, _ = continuation(lambda p_, g: solve(0.0, p_, g), 0.0, eq0, [0.0])
                ramp, _, _ = continuation(lambda p_, g: solve(0.0, p_, g), 0.0, st0[0.0], [0.05])
                st, _, _ = continuation(lambda p_, g: solve(p_, 0.05, g), 0.0, ramp[0.05], VG)
                rows = []
                for v in VG:
                    r = st[v]
                    self.assertTrue(r["converged"])
                    w = (r["n"] * mesh.cv_area_semi)[col]
                    mu = S.nodal(mat, r["psi"], r["n"], r["p"])[0]
                    a = S.act
                    inv_c = np.zeros(len(P))
                    if S.ic is not None:
                        Fa = S.normal_field(r["psi"])[a]
                        inv_c[a] = S.Da * sm.interface_coulomb(Fa, S.ic, r["n"][a], S.T, S.ic_prm)
                    rows.append((abs(contact_current(mesh, mat, r, "drain")), w.sum() / (x1 - x0) * 2,
                                 (mu[col] * w).sum() / w.sum(), w.sum() / max((inv_c[col] * w).sum(), 1e-300)))
                res[qit] = np.array(rows)
        finally:
            sm.attach_mobility(mesh, mat, "off")
            m._CTX.clear()
        I, Ni, mu, muC = res[1e12].T
        self.assertTrue(np.all(np.diff(I) > 0), I)
        on = Ni > 4e11                     # above threshold (below it f(F) switches the term on)
        self.assertTrue(np.all(np.diff(muC[on]) > 0), muC[on])
        # mu_eff / mu_eff(no Qit) at the same N_inv rises towards 1 (screening merges the curves)
        ok = (Ni > 8e11) & (Ni <= res[0.0][:, 1].max())
        ratio = (mu / np.interp(np.log(Ni), np.log(res[0.0][:, 1]), res[0.0][:, 2]))[ok]
        self.assertGreaterEqual(ok.sum(), 3)
        self.assertTrue(np.all(np.diff(ratio) > 0), ratio)
        self.assertGreater(ratio[-1], 0.78)


if __name__ == "__main__":
    unittest.main()
