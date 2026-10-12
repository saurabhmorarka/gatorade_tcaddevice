"""Compiled kernels (native/, core/native.py) agree with their numpy
references. Skipped when the extension is not built.

Run with:  python3 testsuite/test_native.py [-v]
"""
import os
import sys
import unittest

import numpy as np

TCAD1D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TCAD1D_ROOT)

from core.native import native  # noqa: E402


@unittest.skipIf(native is None, "tcad_native not built (see README, 'Compiled kernels')")
class TestTracer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from btbt import main_btbt2d_sweep as B
        from main2d_mosfet_sweep import continuation
        cwd = os.getcwd()
        os.chdir(TCAD1D_ROOT)
        try:
            ctx = B._context(B.DEFAULT_PATH)
        finally:
            os.chdir(cwd)
        # GIDL bias (Vg = -1 V, Vd = 1 V): complete Kane paths near the gate edge
        ramp, _, _ = continuation(lambda p, i: B._newton(ctx, None, 0.0, p, 0.0, i), 0.0, ctx["eq"], [1.0])
        dn, _, _ = continuation(lambda p, i: B._newton(ctx, None, p, 1.0, 0.0, i), 0.0, ramp[1.0], [-1.0])
        cls.ctx, cls.r = ctx, dn[-1.0]

    def test_matches_numpy(self):
        from btbt import paths2d
        g, mat, psi = self.ctx["geom"], self.ctx["mat"], self.r["psi"]
        g_node = g.node_grad(psi)
        st = np.flatnonzero(g.is_semi_node & ~g.is_contact & (np.hypot(*g_node.T) > paths2d.F_START_KANE))
        lev_c = psi + g.dEi - g.ec_off
        tracer = paths2d.FieldLineTracer(g, mat.Eg_eV)
        cases = (("kane", +1, lev_c, g.Eg[st], g.Eg[st] / paths2d.F_EFF_MIN),
                 ("tat_n", +1, lev_c, g.ec_off[st], paths2d.L_MAX_TAT),
                 ("tat_p", -1, lev_c + g.Eg, (g.Eg - g.ec_off)[st], paths2d.L_MAX_TAT))
        saved = paths2d.native
        try:
            for name, sign, lev, target, L_max in cases:
                paths2d.native = None
                ref = tracer.trace(psi, g_node, st, target, sign, L_max, level=lev)
                paths2d.native = native
                got = tracer.trace(psi, g_node, st, target, sign, L_max, level=lev)
                with self.subTest(name):
                    fin = np.isfinite(ref["l"])
                    self.assertGreater(fin.sum(), 50, "enough complete paths to test")
                    np.testing.assert_array_equal(np.isfinite(got["l"]), fin)
                    np.testing.assert_array_equal(got["end_tri"], ref["end_tri"])
                    np.testing.assert_allclose(got["l"][fin], ref["l"][fin], rtol=1e-10)
                    np.testing.assert_allclose(got["end"][fin], ref["end"][fin], rtol=1e-10, atol=1e-16)
                    for k in ("s", "dpsi", "eg_int"):
                        np.testing.assert_allclose(got[k], ref[k], rtol=1e-10, atol=1e-20)
        finally:
            paths2d.native = saved

    def test_rejects_bad_input(self):
        g = self.ctx["geom"]
        c = np.ascontiguousarray
        z = np.zeros(1)
        with self.assertRaises(ValueError):         # nodal arrays of the wrong length
            native.trace_field_lines(c(g.points), c(g.triangles, dtype=np.int64), g.neighbors, *g.vertex_tris,
                                     c(g._Minv), c(g._v2), c(g.tri_semi), np.zeros((3, 2)), z, z,
                                     np.array([0]), np.array([-2]), z, z, 1.0, 1.12, 0.05, 5e-9, 2e-7)



@unittest.skipIf(native is None, "tcad_native not built (see README, 'Compiled kernels')")
class TestLevelSet(unittest.TestCase):
    """fab/levelset.py redistance(): the Rust closest-point kernels against
    the numpy path, on a distorted (non-distance) box field with sharp edges."""

    def test_redistance_matches_numpy(self):
        from fab import levelset
        h = np.array([1.0, 0.5, 1.0])
        X, Y, Z = np.meshgrid(*[np.arange(n) * hh for n, hh in zip((50, 90, 40), h)], indexing="ij")
        q = np.abs(np.stack([X - 24.3, Y - 22.2, Z - 19.1], -1)) - np.array([12.4, 9.15, 10.7])
        box = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
        f = box * (1.0 + 0.6 * np.sin(X / 7.0) * np.cos(Z / 5.0)) ** 3
        d_rust = levelset.redistance(f, h, band=8)
        saved, levelset.native = levelset.native, None
        try:
            d_numpy = levelset.redistance(f, h, band=8)
        finally:
            levelset.native = saved
        np.testing.assert_allclose(d_rust, d_numpy, rtol=0, atol=1e-9)

    def test_rejects_fortran_order(self):
        phi = np.zeros((3, 3, 3))
        with self.assertRaises(ValueError):
            native.ls_project(phi, np.ones(3), np.asfortranarray(np.ones((4, 3))), 1)


if __name__ == "__main__":
    unittest.main()
