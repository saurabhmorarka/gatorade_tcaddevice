"""Stress solver (mechanics/): elastic tensors, closed-form checks of the
2D plane-strain solver, and the process-history engine (birth, death,
anneal, per-stressor superposition)."""
import unittest

import numpy as np

from mechanics import elastic as el
from mechanics.fe2d import Grid2D, StressHistory2D


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


if __name__ == "__main__":
    unittest.main()
