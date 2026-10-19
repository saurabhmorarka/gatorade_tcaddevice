"""Stress solver (mechanics/): elastic tensors, closed-form checks of the
2D plane-strain solver, and the process-history engine (birth, death,
anneal, per-stressor superposition)."""
import unittest

import numpy as np

import os

from mechanics import elastic as el
from mechanics.fe2d import Grid2D, StressHistory2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _film_on_substrate(nx=20, h=1.0, t_film=4, t_sub=30, film_mat=None, sub_mat=None):
    g = Grid2D(nx, t_film + t_sub + 6, h, y0=-(t_film + 6) * h)
    mats = {1: sub_mat or el.material("Si"), 2: film_mat or el.material("SiO2")}
    sh = StressHistory2D(g, mats)
    lab = np.zeros((g.ny, g.nx), np.int32)
    lab[t_film + 6:, :] = 1
    sh.step("substrate", lab)
    lab2 = lab.copy()
    lab2[6:t_film + 6, :] = 2
    return g, sh, lab, lab2


class TestElastic(unittest.TestCase):
    def test_rotated_silicon(self):
        C = el.material("Si").C
        c11, c12, c44 = 165.7, 63.9, 79.6
        self.assertAlmostEqual(C[0, 0], (c11 + c12) / 2 + c44, places=9)    # [110]
        self.assertAlmostEqual(C[1, 1], c11, places=9)                         # [001]
        self.assertAlmostEqual(C[0, 2], (c11 + c12) / 2 - c44, places=9)
        self.assertAlmostEqual(C[4, 4], (c11 - c12) / 2, places=9)
        # a rotation by 90 deg about [001] leaves a cubic crystal unchanged
        C100 = el.rotate_C(el.cubic_C(c11, c12, c44), el.frame(el.FRAME_001_100))
        np.testing.assert_allclose(C100, el.cubic_C(c11, c12, c44)[[0, 2, 1, 3, 5, 4]][:, [0, 2, 1, 3, 5, 4]],
                                   atol=1e-9)

    def test_isotropic_is_rotation_invariant(self):
        C = el.isotropic_C(70.0, 0.17)
        np.testing.assert_allclose(el.rotate_C(C, el.frame(el.FRAME_001_110)), C, atol=1e-9)

    def test_sige_mismatch(self):
        self.assertAlmostEqual(el.sige_mismatch(0.5), 0.0197, places=3)
        self.assertEqual(el.sige_mismatch(0.0), 0.0)

    def test_eigenstrain_from_stress(self):
        m = el.material("Si3N4")
        sig0 = np.array([1.5, 1.5, 1.5, 0, 0, 0])
        np.testing.assert_allclose(m.C @ (-m.eigenstrain_from_stress(sig0)), sig0, atol=1e-12)


class TestPlaneStrain(unittest.TestCase):
    def test_film_biaxial_closed_form(self):
        """Isotropic film, eigenstrain e, side walls held: eps_xx = eps_zz = 0,
        sigma_yy = 0 => sigma_xx = sigma_zz = -E e / (1 - nu), substrate 0."""
        E, nu, e = 70.0, 0.17, 0.01
        g, sh, lab, lab2 = _film_on_substrate()
        sh.step("film", lab2, stressors={"film": (lab2 == 2, [e, e, e, 0, 0, 0])})
        s = sh.total().reshape(g.ny, g.nx, 6)
        film = s[6:10]
        np.testing.assert_allclose(film[..., 0], -E * e / (1 - nu), rtol=1e-6)
        np.testing.assert_allclose(film[..., 2], -E * e / (1 - nu), rtol=1e-6)
        self.assertLess(np.abs(film[..., 1]).max(), 1e-6)
        self.assertLess(np.abs(s[10:]).max(), 1e-6)

    def test_sige_pseudomorphic_on_001(self):
        """SiGe on Si(001): in-plane biaxial stress -(C11 + C12 - 2 C12^2/C11) f,
        the same along <110> and <100> (the (001) plane is isotropic in-plane
        for biaxial strain)."""
        x = 0.3
        f = el.sige_mismatch(x)
        c11, c12 = [(1 - x) * a + x * b for a, b in ((165.7, 128.5), (63.9, 48.3))]
        expect = -(c11 + c12 - 2 * c12 ** 2 / c11) * f
        for R in (el.FRAME_001_110, el.FRAME_001_100):
            g, sh, lab, lab2 = _film_on_substrate(film_mat=el.material("SiGe", R, x_ge=x),
                                                  sub_mat=el.material("Si", R))
            sh.step("epi", lab2, stressors={"SiGe": (lab2 == 2, [f, f, f, 0, 0, 0])})
            s = sh.total().reshape(g.ny, g.nx, 6)[6:10]
            np.testing.assert_allclose(s[..., 0], expect, rtol=1e-6)
            np.testing.assert_allclose(s[..., 2], expect, rtol=1e-6)

    def test_mesh_convergence_patterned_film(self):
        """A film over half the width (edge relaxation): the stress under the
        film edge converges as the pixel shrinks."""
        vals = []
        for h in (2.0, 1.0, 0.5):
            n = int(round(40 / h))
            g = Grid2D(n, n, h, y0=-10.0)
            sh = StressHistory2D(g, {1: el.material("Si"), 2: el.material("Si3N4")})
            X, Y = g.coords()
            lab = np.where(Y > 0, 1, 0).astype(np.int32)
            sh.step("sub", lab)
            lab2 = lab.copy()
            lab2[(Y < 0) & (Y > -6) & (X < 20)] = 2
            sh.step("film", lab2, stressors={"liner": (lab2 == 2, el.material("Si3N4").eigenstrain_from_stress(
                [1.0, 1.0, 1.0, 0, 0, 0]))})
            probe = (X > 14) & (X < 20) & (Y > 0) & (Y < 6)
            vals.append(sh.probe(probe)[0][0])
        self.assertLess(abs(vals[2] - vals[1]), 0.5 * abs(vals[1] - vals[0]) + 1e-4)


class TestHistory(unittest.TestCase):
    def test_deposit_then_remove_returns(self):
        g, sh, lab, lab2 = _film_on_substrate(nx=16)
        # a patterned film so the substrate really is stressed
        lab2[:, 8:] = lab[:, 8:]
        sh.step("film", lab2, stressors={"film": (lab2 == 2, [0.01, 0.01, 0.01, 0, 0, 0])})
        self.assertGreater(np.abs(sh.total()[lab.ravel() == 1]).max(), 1e-3)
        sh.step("strip", lab)
        # 0 up to the void pixels' 1e-7 stiffness (they carried a sliver of the load)
        self.assertLess(np.abs(sh.total()).max(), 1e-5)

    def test_superposition(self):
        """Two stressors switched on together = each alone, summed."""
        def run(which):
            g, sh, lab, lab2 = _film_on_substrate(nx=16)
            lab2[:, 8:] = lab[:, 8:]
            st = {}
            if "a" in which:
                st["a"] = (lab2 == 2, [0.01, 0.01, 0.01, 0, 0, 0])
            if "b" in which:
                st["b"] = (lab2 == 1, [0, 0.002, 0, 0, 0, 0])
            sh.step("go", lab2, stressors=st)
            return sh
        both = run("ab")
        np.testing.assert_allclose(both.total(), run("a").total() + run("b").total(), atol=1e-10)

    def test_stress_memorization(self):
        """Cap a block with a stressed film, anneal the block (r = 1), strip
        the cap: the block keeps a stress of the opposite sign to what it
        had under the cap (the SMT mechanism, elastically)."""
        g, sh, lab, _ = _film_on_substrate(nx=24, t_film=4, t_sub=20)
        X, Y = g.coords()
        top = (Y > 0) & (Y < 6) & (X > 6) & (X < 18) & (lab == 1)
        cap = lab.copy()
        cap[(Y < 0) & (Y > -4) & (X > 4) & (X < 20)] = 3     # patterned: a uniform film stresses nothing
        sh.mats[3] = el.material("Si3N4")
        sh.step("cap", cap, stressors={"cap": (cap == 3, el.material("Si3N4").eigenstrain_from_stress(
            [1.5, 1.5, 1.5, 0, 0, 0]))})
        under_cap = sh.probe(top)[0][0]
        sh.step("anneal", cap, relax=(top, 1.0))
        # (not 0 after the anneal: the relaxed block re-equilibrates with its
        # unrelaxed surroundings, which still hold the cap's load)
        self.assertLess(abs(sh.probe(top)[0][0]), abs(under_cap))
        sh.step("strip", lab)
        kept = sh.probe(top)[0][0]
        self.assertGreater(abs(kept), 1e-3)
        self.assertLess(np.sign(kept) * np.sign(under_cap), 0)


class TestDislocation(unittest.TestCase):
    def test_edge_dislocation_matches_volterra_field(self):
        """Volterra cut vs the analytic infinite-medium edge dislocation
        (Hirth & Lothe), isotropic, 10-20 nm from the core. sigma_yy and
        sigma_xy only: the side walls are held, which adds a near-uniform
        sigma_xx (inserting b into a constrained box), ~E b / L."""
        from mechanics.dislocation import edge_dislocation, edge_dislocation_infinite
        E, nu, b, L = 100.0, 0.25, 0.4, 200.0
        n = int(L)
        g = Grid2D(n, n, 1.0, -L / 2, -L / 2)
        sh = StressHistory2D(g, {1: el.ElasticMaterial("iso", el.isotropic_C(E, nu))})
        lab = np.ones((n, n), np.int32)
        sh.step("bulk", lab)
        mask, eps = edge_dislocation(g, (0.0, 0.0), (0.0, -1.0), (b, 0.0), labels=lab)
        sh.step("disl", lab, stressors={"d": (mask, eps)})
        s = sh.total().reshape(n, n, 6)
        X, Y = g.coords()
        for x, y in [(15.5, 0.5), (0.5, 15.5), (12.5, -7.5), (-11.5, -7.5), (8.5, 3.5)]:
            j, i = int(y + L / 2), int(x + L / 2)
            ax, ay, axy = edge_dislocation_infinite(X[j, i], -Y[j, i], b, E, nu)   # textbook y is up
            for fe, an in ((s[j, i, 1], ay), (s[j, i, 5], -axy)):
                if abs(an) > 0.05:
                    self.assertLess(abs(fe - an), 0.2 * abs(an))

    def test_smt_dislocation_tension(self):
        """Missing half planes at the S/D edges put the NMOS channel in
        tension (the SMT dislocation boost)."""
        from mechanics.planar_flow import DEFAULTS, run_flow
        with_d = run_flow("nmos_smt", par=dict(DEFAULTS, smt_disl=1.0)).rows[-1][2]
        self.assertGreater(with_d["dislocations"][0], 200.0)    # MPa


class TestPiezo(unittest.TestCase):
    def test_110_coefficients_and_signs(self):
        from mechanics.piezo import mobility_change, pi_110_on_001
        n, p = pi_110_on_001("n"), pi_110_on_001("p")
        self.assertAlmostEqual(n["l"], -0.312, places=3)          # Smith, <110> longitudinal
        self.assertAlmostEqual(p["l"], 0.718, places=3)
        # textbook signs: NMOS gains from longitudinal tension, PMOS from compression
        self.assertGreater(mobility_change([0.1, 0, 0, 0, 0, 0], "n"), 0)
        self.assertGreater(mobility_change([-0.1, 0, 0, 0, 0, 0], "p"), 0)
        self.assertLess(mobility_change([0.1, 0, 0, 0, 0, 0], "p"), 0)


class TestHistory3D(unittest.TestCase):
    """mechanics/fe3d.py (direct solves here: the .venv has no PETSc)."""

    def _film(self, film_mat, f, R=el.FRAME_001_110):
        from mechanics.fe3d import Grid3D, StressHistory3D
        g = Grid3D((4, 24, 3), 1.0, (0, -6, 0))
        sh = StressHistory3D(g, {1: el.material("Si", R), 2: film_mat})
        X, Y, Z = g.centres()
        lab = np.where(Y > 0, 1, 0)
        sh.step("sub", lab)
        lab2 = np.where((Y < 0) & (Y > -4), 2, lab)
        sh.step("film", lab2, stressors={"f": (lab2 == 2, [f, f, f, 0, 0, 0])})
        return sh.element_stress().reshape(g.shape + (6,)), sh, lab

    def test_film_closed_form(self):
        E, nu, e = 70.0, 0.17, 0.01
        s, sh, lab = self._film(el.material("SiO2"), e)
        film = s[:, 2:6]
        np.testing.assert_allclose(film[..., 0], -E * e / (1 - nu), rtol=1e-9)
        np.testing.assert_allclose(film[..., 2], -E * e / (1 - nu), rtol=1e-9)
        self.assertLess(np.abs(s[:, 6:]).max(), 1e-9)
        sh.step("strip", lab)                       # no void stiffness in 3D: back to exactly 0
        self.assertLess(np.abs(sh.element_stress()).max(), 1e-12)

    def test_sige_biaxial(self):
        x = 0.3
        f = el.sige_mismatch(x)
        c11, c12 = [(1 - x) * a + x * b for a, b in ((165.7, 128.5), (63.9, 48.3))]
        s, _, _ = self._film(el.material("SiGe", x_ge=x), f)
        np.testing.assert_allclose(s[:, 2:6, :, 0], -(c11 + c12 - 2 * c12 ** 2 / c11) * f, rtol=1e-9)

    def _patterned(self, flush):
        from mechanics.fe3d import Grid3D, StressHistory3D
        g = Grid3D((16, 20, 4), 1.0, (0, -6, 0))
        sh = StressHistory3D(g, {1: el.material("Si"), 2: el.material("SiGe", x_ge=0.5), 3: el.material("Poly")})
        X, Y, Z = g.centres()
        lab = np.where(Y > 0, 1, 0)
        sh.step("sub", lab, flush=flush)
        lab = lab.copy()
        lab[(Y < 0) & (Y > -5) & (np.abs(X - 8) < 3)] = 3                    # a "gate"
        sh.step("gate", lab, flush=flush)
        lab[(Y > 0) & (Y < 5) & (np.abs(X - 8) > 4)] = 0
        sh.step("recess", lab, flush=flush)
        f = el.sige_mismatch(0.5)
        lab[(Y > -2) & (Y < 5) & (np.abs(X - 8) > 4)] = 2
        sh.step("epi", lab, stressors={"sige": (lab == 2, [f, f, f, 0, 0, 0])}, flush=flush)
        lab[lab == 3] = 0
        sh.step("gate pull", lab, flush=flush)
        lab[(Y < -3)] = 0
        sh.step("top cut", lab, flush=flush)
        sh.flush()
        return sh, g, X, Y

    def test_lazy_equals_stepwise(self):
        """Solving only where needed (flush=False) gives the same final
        stress as solving after every step."""
        a, g, X, Y = self._patterned(True)
        b, *_ = self._patterned(False)
        np.testing.assert_allclose(a.element_stress(), b.element_stress(), atol=1e-10)
        self.assertLess(b.n_solves, a.n_solves)
        ch = ((np.abs(X - 8) < 2) & (Y > 0) & (Y < 2))
        self.assertLess(a.probe(ch)[0][0], -0.5)                            # SiGe S/D compress the channel

    def test_z_uniform_slab_matches_2d(self):
        """With rollers on both z walls and a z-uniform structure the 3D
        solution is plane strain: it must equal the 2D solver's."""
        from mechanics.fe3d import Grid3D, StressHistory3D
        mats = {1: el.material("Si"), 2: el.material("SiGe", x_ge=0.5)}
        f = el.sige_mismatch(0.5)
        g2 = Grid2D(16, 20, 1.0, y0=-4.0)
        X2, Y2 = g2.coords()
        g3 = Grid3D((16, 20, 2), 1.0, (0, -4, 0))
        X3, Y3, Z3 = g3.centres()
        out = []
        for X, Y, sh in ((X2, Y2, StressHistory2D(g2, mats)), (X3, Y3, StressHistory3D(g3, mats))):
            lab = np.where(Y > 0, 1, 0)
            sh.step("sub", lab)
            lab = lab.copy()
            lab[(Y > -2) & (Y < 6) & (np.abs(X - 8) > 4)] = 2
            sh.step("epi", lab, stressors={"s": (lab == 2, [f, f, f, 0, 0, 0])})
            m = (np.abs(X - 8) < 2) & (Y > 0) & (Y < 3)
            out.append(sh.probe(m)[0])
        # 2e-6: the 2D solver's void pixels carry 1e-7 x Si stiffness; 3D has no void unknowns
        np.testing.assert_allclose(out[1], out[0], rtol=1e-5, atol=1e-9)


if __name__ == "__main__":
    unittest.main()


class TestFabStressSnapshots(unittest.TestCase):
    def test_bilayer_seed_folds_into_the_epi_birth(self):
        """The bilayer epi's seed is carved out of the just-grown epi: the
        stress history must see one birth (seed + fill), not a second
        stress-free birth of the seed inside an already strained epi. The
        seed and fill get their own Ge fraction from the material name."""
        from mechanics import fab_stress as fs
        snaps, regions, grid, st = fs.snapshots(os.path.join(ROOT, "configs", "fab_finfet_pmos.yaml"))
        labels = [l for l, _ in snaps]
        self.assertTrue(any("S/D epitaxy" in l and "epi seed" in l for l in labels))
        self.assertFalse(any(l.startswith("S/D epi seed") for l in labels))
        x = {n: fs._ge_fraction(n, m, fs.DEFAULTS) for n, m in regions.values() if m.startswith("SiGe")}
        self.assertEqual(x, {"sd_epi": 0.5, "sd_epi_seed": 0.2})
