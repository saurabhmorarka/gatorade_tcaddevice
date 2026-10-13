"""Schottky metal/semiconductor contacts (plans/schottky_plan.md).

The S/D contact of every modern transistor is a silicide or Ti/TiN metal on
degenerately doped epi: a Schottky barrier that conducts by tunneling. This
case checks the model against references, not only against itself:

  TestWKB, TestSupplyAndBarrier - the kernel (contacts/schottky.py): the WKB
      integral is exact for a triangular barrier, its analytic derivatives
      match finite differences, the supply function and barrier models.
  TestSchottky1D - the 1D solver (contacts/newton_schottky_1d.py):
      * thermionic-emission limit = the analytic Richardson diode;
      * rho_c(N) against the independent depletion-model (Yu) calculation;
      * mesh independence of rho_c;
      * current conservation (metal end = far ohmic end) through bias
        sweeps of the two reference S/D contacts: NMOS Si:P 7e20 and PMOS
        strained Si0.5Ge0.5:B 5e20;
      * golden rho_c of those two contacts (regression).
  TestSchottkySubgrid - the sub-grid contact used on 2D/3D meshes
      (contacts/subgrid.py): a quasi-1D slab on the fab's rectilinear
      mesher (metal | n+ or p+ | ohmic) at 1 and 2 nm spacing must give the
      resolved 1D model's current (0.05 nm mesh) within 2%, identical on
      both meshes, with the exact interface area and current conserved.
  TestSchottky3D - the tiny planar NMOS (configs/fab_planar_tiny.yaml,
      octree) with the template-default Schottky S/D contacts, against the
      same device with ohmic contacts: interface area = the metal/silicon
      voxel faces, every point converges, Kirchhoff (source + drain + body
      = 0), the Schottky drain current below the ohmic one, golden currents.
      (test_3d_examples keeps the ohmic reference device.)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contacts import schottky as sk


class TestWKB(unittest.TestCase):
    def test_triangular_barrier_exact(self):
        # U = Phi - F s (linear): Int sqrt(U - E) = (2/3) (Phi - E)^1.5 / F,
        # exact on any grid, the turning point at s = (Phi - E)/F.
        phi, F = 0.6, 2e6                          # eV, eV/cm (2 MV/cm)
        for npts in (3, 7, 41):
            s = np.linspace(0.0, 1e-6, npts)
            U = phi - F * s
            E = np.array([0.0, 0.2, 0.55, 0.7])
            I, _, jt, t, ok = sk.wkb_integral(s, U, E)
            np.testing.assert_array_equal(ok, [True, True, True, False])
            exact = (2.0 / 3.0) * (phi - E[:3]) ** 1.5 / F
            np.testing.assert_allclose(I[:3], exact, rtol=1e-12)
            st = s[jt[:3]] + t[:3] * (s[jt[:3] + 1] - s[jt[:3]])
            np.testing.assert_allclose(st, (phi - E[:3]) / F, rtol=1e-12)

    def test_no_turning_point(self):
        s = np.linspace(0.0, 1e-7, 5)
        U = np.full(5, 0.5)
        I, dI, _, _, ok = sk.wkb_integral(s, U, np.array([0.1]))
        self.assertFalse(ok[0])
        self.assertEqual(I[0], 0.0)
        self.assertFalse(dI.any())

    def test_derivative_matches_finite_difference(self):
        rng = np.random.default_rng(1)
        s = np.cumsum(np.r_[0.0, rng.uniform(0.5e-8, 2e-8, 11)])
        U = 0.7 - 0.9 * (1 - np.exp(-s / 4e-8)) + 0.002 * rng.standard_normal(12)
        E = np.linspace(-0.15, 0.65, 23)
        I, dI, jt, _, ok = sk.wkb_integral(s, U, E)
        h = 1e-7
        for m in range(len(U)):
            Up = U.copy(); Up[m] += h
            Um = U.copy(); Um[m] -= h
            Ip, _, jtp, _, okp = sk.wkb_integral(s, Up, E)
            Im, _, jtm, _, okm = sk.wkb_integral(s, Um, E)
            same = ok & okp & okm & (jtp == jt) & (jtm == jt)
            np.testing.assert_allclose(dI[same, m], ((Ip - Im) / (2 * h))[same], rtol=1e-5, atol=1e-14)


class TestSupplyAndBarrier(unittest.TestCase):
    def test_supply_limits(self):
        Vt = 0.02585
        N, _ = sk.supply(np.array([0.0]), -0.05, -0.05, Vt)
        self.assertAlmostEqual(N[0], 0.0)
        # Far below both Fermi levels: N -> (EF_s - EF_m)/Vt.
        N, _ = sk.supply(np.array([-2.0]), 0.1, 0.0, Vt)
        self.assertAlmostEqual(N[0], 0.1 / Vt, places=6)
        # dN/dEF_s by finite difference.
        E = np.linspace(-0.3, 0.3, 13)
        _, dN = sk.supply(E, 0.02, 0.0, Vt)
        Np, _ = sk.supply(E, 0.02 + 1e-7, 0.0, Vt)
        Nm, _ = sk.supply(E, 0.02 - 1e-7, 0.0, Vt)
        np.testing.assert_allclose(dN, (Np - Nm) / 2e-7, rtol=1e-6)

    def test_barrier_models(self):
        chi, Eg = 4.05, 1.12
        c = sk.SchottkyContact(metal="Ti")
        self.assertAlmostEqual(c.barriers(chi, Eg)[0], 4.33 - 4.05)
        c = sk.SchottkyContact(metal="Ti", barrier_model="pinned", pinning_S=0.0)
        self.assertAlmostEqual(c.barriers(chi, Eg)[0], Eg / 2)
        c = sk.SchottkyContact(metal="Ti", barrier_model="pinned", pinning_S=1.0)
        self.assertAlmostEqual(c.barriers(chi, Eg)[0], 4.33 - 4.05)
        c = sk.SchottkyContact(metal="Ti", barrier_n_eV=0.6)
        self.assertEqual(c.barriers(chi, Eg), (0.6, Eg - 0.6))
        c = sk.SchottkyContact(barrier_p_eV=0.3)
        self.assertAlmostEqual(c.barriers(chi, 0.8)[0], 0.5)

    def test_boundary_state_equilibrium_consistency(self):
        Vt, ni, Nc, Nv = 0.02585, 1e10, 2.8e19, 1.04e19
        c = sk.SchottkyContact(barrier_n_eV=0.6)
        for maj in ("n", "p"):
            st = sk.boundary_state(c, 4.05, 1.12, Nc, Nv, ni, Vt, maj, V=0.3)
            self.assertAlmostEqual(st["n_B"] * st["p_B"] / ni ** 2, 1.0)
            # psi_s reproduces the boundary densities with phin = phip = V.
            n_s = ni * np.exp((st["psi_s"] - 0.3) / Vt)
            self.assertAlmostEqual(n_s / st["n_B"], 1.0)
        st = sk.boundary_state(c, 4.05, 1.12, Nc, Nv, ni, Vt, "n")
        self.assertAlmostEqual(Vt * np.log(Nc / st["n_B"]), 0.6)

    def test_te_velocity_si(self):
        # Si electrons: A** = 112, Nc = 2.8e19 -> ~2.25e6 cm/s (Sentaurus default 2.57e6).
        v = sk.te_velocity(112.0, 300.0, 2.8e19)
        self.assertTrue(2.0e6 < v < 2.6e6)


class TestSchottky1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.params import Material
        from core.materials import AlloyMaterial, resolve_material
        from core import material_db as db
        from contacts import newton_schottky_1d as ns
        cls.ns = ns
        cls.si_default = Material()
        cls.si = resolve_material(db.get("Silicon"))
        cls.sige = resolve_material(db.derive_alloy(
            "Si0.5Ge0.5_strained", AlloyMaterial("SixGe1-x", "Silicon", "Germanium", bowing_eV=0.36),
            0.5, strained_on="Silicon"))

    def test_thermionic_emission_limit(self):
        # n-Si 1e16, Phi_B = 0.6 eV, no tunneling. With a very high mobility
        # the diffusion velocity is infinite (Crowell-Sze), so J must be the
        # Richardson diode A** T^2 exp(-Phi_B/Vt) (exp(V/Vt) - 1).
        from core.params import Material
        ns = self.ns
        mat = Material(mu_n=1.35e6, mu_p=4.8e5)
        x = ns.contact_grid(2e-4, h0=0.1e-7, h_max=20e-7)
        Cdop = np.full(len(x), 1e16)
        c = sk.SchottkyContact(barrier_n_eV=0.6, tunneling=False)
        rs = ns.sweep(x, Cdop, mat, {"left": c, "right": None}, [0.1, 0.2])
        J0 = c.richardson_n * 300.0 ** 2 * np.exp(-0.6 / mat.Vt)
        for V, r in zip([0.1, 0.2], rs):
            self.assertAlmostEqual(r["J_left"] / (J0 * np.expm1(V / mat.Vt)), 1.0, delta=2e-3)

    def test_rho_c_against_depletion_model(self):
        # Same Tsu-Esaki physics through an analytic parabolic barrier. The
        # self-consistent barrier has a free-carrier tail, so the solver is
        # higher, approaching the analytic value as the doping rises.
        ns, mat = self.ns, self.si_default
        c = sk.SchottkyContact(barrier_n_eV=0.6)
        ratios = []
        for N in (1e19, 1e20, 7e20, 2e21):
            rs = ns.contact_resistivity(mat, N, c)
            ry = sk.rho_c_depletion_model(N, 0.6, mat.Vt, mat.Nc, mat.eps, c.richardson_n, 300.0, c.m_tunnel_n)
            ratios.append(rs / ry)
        for r in ratios:
            self.assertTrue(1.0 < r < 2.0, ratios)
        self.assertTrue(all(np.diff(ratios) < 0), ratios)

    def test_mesh_independence(self):
        ns, mat = self.ns, self.si_default
        c = sk.SchottkyContact(barrier_n_eV=0.6)
        fine = ns.contact_resistivity(mat, 7e20, c, h0=0.05e-7)
        coarse = ns.contact_resistivity(mat, 7e20, c, h0=0.2e-7)
        self.assertLess(abs(coarse / fine - 1.0), 0.03)

    def test_current_conservation_nmos_pmos(self):
        # Bias sweeps of the two reference S/D contacts on a 30 nm slab (the
        # tunneling path reaches the far contact: its node must stay out).
        ns = self.ns
        x = ns.contact_grid(30e-7)
        Vs = [0.1, 0.3, -0.1, -0.3]
        for mat, N, c in ((self.si, 7e20, sk.SchottkyContact(barrier_n_eV=0.55)),
                          (self.sige, -5e20, sk.SchottkyContact(barrier_p_eV=0.2))):
            rs = ns.sweep(x, np.full(len(x), N), mat, {"left": c, "right": None}, Vs)
            for V, r in zip(Vs, rs):
                self.assertIsNotNone(r)
                self.assertTrue(r["converged"])
                self.assertLess(abs(r["J_left"] - r["J_right"]) / abs(r["J_left"]), 1e-4, (N, V))
                self.assertGreater(r["J_left"] * V, 0.0)      # current flows from the higher potential

    def test_reference_contacts_golden(self):
        # TiN-on-S/D reference numbers (Schottky-Mott barriers, Study A).
        ns = self.ns
        rn = ns.contact_resistivity(self.si, 7e20, sk.SchottkyContact(barrier_n_eV=0.55))
        rp = ns.contact_resistivity(self.sige, -5e20, sk.SchottkyContact(barrier_p_eV=0.2))
        self.assertAlmostEqual(rn / 5.18e-9, 1.0, delta=3e-3)
        self.assertAlmostEqual(rp / 6.62e-9, 1.0, delta=3e-3)

    def test_nisi_pmos_leaning(self):
        # Study A, NiSi: near-midgap pinning. The Eg - Phi_Bn transfer must
        # give its measured p-Si hole barrier (~0.45 eV) from the measured
        # n-Si one (0.65-0.70), and carried to Si0.5Ge0.5 it must favor PMOS:
        # worse than TiN on Si:P, better than TiN on SiGe:B.
        from contacts import study_ti_barriers as st
        rows = {r["metal"]: r for r in st.measured_rows(self.si, self.sige)}
        self.assertLess(abs(rows["NiSi"]["phi_bp_si"] - st.MEASURED_P_SI["NiSi"]), 0.05)
        ns = self.ns
        rho = {m: (ns.contact_resistivity(self.si, 7e20, sk.SchottkyContact(barrier_n_eV=r["phi_bn_si"])),
                   ns.contact_resistivity(self.sige, -5e20, sk.SchottkyContact(barrier_p_eV=r["phi_bp_sige"])))
               for m, r in rows.items() if m in ("NiSi", "TiN")}
        self.assertGreater(rho["NiSi"][0], 1.5 * rho["TiN"][0])
        self.assertLess(rho["NiSi"][1], 0.5 * rho["TiN"][1])
        self.assertAlmostEqual(rho["NiSi"][0] / 7.97e-9, 1.0, delta=5e-3)
        self.assertAlmostEqual(rho["NiSi"][1] / 3.64e-9, 1.0, delta=5e-3)


class TestSchottkySubgrid(unittest.TestCase):
    @staticmethod
    def slab(mat, h, N, L=30e-7, W=4e-7):
        from mesh3d.cartesian import build_cartesian_mesh
        xs = np.linspace(0.0, W, 3)
        ys = np.unique(np.concatenate([[-2e-7], np.arange(0.0, L + 1e-12, h), [L]]))
        lab = np.zeros((len(xs) - 1, len(ys) - 1), int)
        lab[:, 0] = 1
        mats = [dict(name="si", kind="semi"), dict(name="metal", kind="conductor", contact="source")]
        mesh = build_cartesian_mesh([xs, ys], lab, mats, np.full((len(xs), len(ys)), N), mat,
                                    face_contacts={"body": (1, "max")})
        return mesh, xs, ys, lab

    def test_slab_matches_resolved_1d(self):
        from core.params import Material
        from contacts import newton_schottky_1d as ns
        from contacts import subgrid as sg
        from solver2d.newton_solver_qf_2d import newton_solve_2d
        from solver2d.current import contact_current
        # constant mobility in both (the 2D default is Masetti)
        mat = Material(masetti_n=(0.0, 0.0, 0.0, 0.0, 1e40, 1e40, 1.0, 1.0, 1350.0),
                       masetti_p=(0.0, 0.0, 0.0, 0.0, 1e40, 1e40, 1.0, 1.0, 480.0))
        Vs = [0.05, 0.1]
        for N, c in ((7e20, sk.SchottkyContact(barrier_n_eV=0.55)), (-5e20, sk.SchottkyContact(barrier_p_eV=0.2))):
            x1 = ns.contact_grid(30e-7)
            Vs_dir = [v * np.sign(N) for v in Vs]
            ref = [r["J_left"] for r in ns.sweep(x1, np.full(len(x1), N), mat, {"left": c, "right": None}, Vs_dir,
                                                 dV_max=0.05)]
            Js = []
            for h in (1e-7, 2e-7):
                mesh, xs, ys, lab = self.slab(mat, h, N)
                area = sg.voxel_interface_areas(lab, ["semi", "conductor"], [None, "source"], [xs, ys], mesh,
                                                ["source"])
                self.assertAlmostEqual(area.sum() / (xs[-1] - xs[0]), 1.0, places=9)
                sub = sg.SubgridContacts.setup(mesh, mat, {"source": c}, area, cache=False)
                r, row = None, []
                for V, Jr in zip(Vs_dir, ref):
                    kw = {} if r is None else dict(psi_init=r["psi"], phin_init=r["phin"], phip_init=r["phip"])
                    r = newton_solve_2d(mesh, mat, {"source": V, "body": 0.0}, schottky=sub, **kw)
                    self.assertTrue(r["converged"], (N, h, V))
                    W = xs[-1] - xs[0]
                    J = contact_current(mesh, mat, r, "source") / W
                    Jb = -contact_current(mesh, mat, r, "body") / W
                    self.assertLess(abs(J / Jr - 1.0), 0.02, (N, h, V, J, Jr))
                    self.assertLess(abs(Jb / J - 1.0), 1e-3, (N, h, V))
                    row.append(J)
                Js.append(row)
            np.testing.assert_allclose(Js[0], Js[1], rtol=1e-3)


class TestSchottky3D(unittest.TestCase):
    # Golden drain currents (A, full device) with Schottky contacts - captured
    # 2026-10-12 with the recessed, implanted contacts, after checking them
    # against the ohmic device (lin 0.43x, sat 0.75x: DEVELOPMENT_LOG 63).
    GOLDEN = {"lin": 6.055e-06, "sat": 2.656e-05}
    RTOL = 5e-3             # linear-solver backend drift (venv SuperLU vs conda PETSc)

    @classmethod
    def setUpClass(cls):
        from fab import flow
        from fab.simulate import build_device, schottky_contacts
        cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
                           "fab_planar_tiny.yaml")
        d = build_device(flow.load(cfg), mesher="octree", log=lambda *a: None)
        d["schottky"] = schottky_contacts(d, log=lambda *a: None)
        cls.d = d
        cls.res = {kind: cls.solve(d, d["schottky"] if kind == "schottky" else None)
                   for kind in ("schottky", "ohmic")}

    @staticmethod
    def solve(d, schottky):
        from main2d_mosfet_sweep import continuation, solve_mosfet
        from mos.mos_params import MOSDevice
        mesh, mat, Cs = d["mesh"], d["mat"], d["Cs"]
        dev = MOSDevice(gate_workfunction_eV=d["wf"])

        def at(Vgs, Vds):
            return lambda p, g: solve_mosfet(mesh, mat, dev, Cs, *((p, Vds) if Vgs is None else (Vgs, p)),
                                             psi_init=g["psi"], phin_init=g["phin"], phip_init=g["phip"],
                                             cold_retry=False, velocity_saturation=True, schottky=schottky)
        eq = solve_mosfet(mesh, mat, dev, Cs, 0.0, 0.0, velocity_saturation=True, schottky=schottky)
        lin, _, _ = continuation(at(None, 0.05), 0.0, eq, [0.6])
        sat, _, _ = continuation(at(0.6, None), 0.05, lin.get(0.6, eq), [0.6])
        return dict(eq=eq, lin=lin.get(0.6), sat=sat.get(0.6))

    def test_interface_area(self):
        d = self.d
        lab, origin, h = d["voxels"]
        from fab.simulate import device_materials
        mats = device_materials(d["st"])
        semi = np.array([m["kind"] == "semi" for m in mats] + [False])
        sk_ = d["schottky"]
        for name in ("source", "drain"):
            metal = np.array([m.get("contact") == name for m in mats] + [False])
            L = np.where(lab >= 0, lab, len(mats))
            A = 0.0
            for a in range(3):
                lo = [slice(None)] * 3
                hi = [slice(None)] * 3
                lo[a], hi[a] = slice(0, -1), slice(1, None)
                n = ((semi[L[tuple(lo)]] & metal[L[tuple(hi)]]) | (metal[L[tuple(lo)]] & semi[L[tuple(hi)]])).sum()
                A += n * np.prod([h[b] for b in range(3) if b != a])
            got = sk_.area[sk_.specs[name]["nodes"]].sum() * 1e14
            self.assertGreater(A, 0)
            self.assertAlmostEqual(got / A, 1.0, places=9)

    def test_converged_and_kirchhoff(self):
        from solver2d.current import contact_current
        d = self.d
        for kind, r in self.res.items():
            for pt in ("eq", "lin", "sat"):
                self.assertIsNotNone(r[pt], (kind, pt))
                self.assertTrue(r[pt]["converged"], (kind, pt))
            for pt in ("lin", "sat"):
                I = {c: contact_current(d["mesh"], d["mat"], r[pt], c) for c in ("source", "drain", "body")}
                self.assertLess(abs(sum(I.values())) / abs(I["drain"]), 2e-3, (kind, pt, I))

    def test_schottky_below_ohmic_and_golden(self):
        from fab.simulate import drain_current
        d = self.d
        for pt in ("lin", "sat"):
            Is = drain_current(d, self.res["schottky"][pt])
            Io = drain_current(d, self.res["ohmic"][pt])
            self.assertLess(Is, Io, pt)
            self.assertAlmostEqual(Is / self.GOLDEN[pt], 1.0, delta=self.RTOL)

    def test_linear_regime_resistance(self):
        """Vds = 0.05 V: each contact of the tiny device (~140 nm^2 per half,
        recessed, ~3e20 after the contact implant) is ~5e3 Ohm at zero bias,
        comparable to the channel. The total resistance must lie between the
        ohmic device's plus the contacts' zero-bias resistance (an upper
        bound: J(u) is superlinear past ~Vt) and the ohmic one plus a third
        of it."""
        from fab.simulate import drain_current
        d, sk_ = self.d, self.d["schottky"]
        R_c = 0.0
        for name in ("source", "drain"):
            s_ = sk_.specs[name]
            i = s_["nodes"]
            J, _ = s_["table"](np.abs(d["mesh"].Cdop[i]), np.full(len(i), 1e-4))
            R_c += 1.0 / (2.0 * np.sum(sk_.area[i] * J / 1e-4))          # full device = 2 halves
        R_s = 0.05 / drain_current(d, self.res["schottky"]["lin"])
        R_o = 0.05 / drain_current(d, self.res["ohmic"]["lin"])
        self.assertLess(R_s, R_o + R_c)
        self.assertGreater(R_s, R_o + R_c / 3.0)


if __name__ == "__main__":
    unittest.main()
