"""Virtual fab (fab/): level-set operations checked against analytic
geometry on small grids, the expression evaluator, and the planar template
build.

Run with:  python3 testsuite/test_fab.py [-v]
"""
import os
import sys
import unittest

import numpy as np

TCAD1D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TCAD1D_ROOT)

from fab import levelset                                     # noqa: E402
from fab.expr import ExprError, evaluate, resolve_params     # noqa: E402
from fab.structure import Grid, Structure                   # noqa: E402


def line_crossing(f, coords):
    """Sub-voxel coordinate where 1D samples f first change sign (- to + or + to -)."""
    s = np.signbit(f)
    i = int(np.flatnonzero(s[:-1] != s[1:])[0])
    return coords[i] + (coords[i + 1] - coords[i]) * f[i] / (f[i] - f[i + 1])


def _non_wedge(warnings):
    """Structure-check warnings other than one-voxel-thick cells of the epi
    (and its seed layer), the ILD or the contacts: a shaped epi has real acute corners (the ILD
    under a diamond's lower facet at the STI, a facet against the fin side,
    the epi's curved top under the flat contact floor) whose tips staircase
    to one voxel."""
    return [w for w in warnings if not (("epi'" in w or "epi_seed'" in w or "'ild'" in w or "_contact'" in w)
                                        and "one voxel thick" in w)]


def top_height(st, x, z, regions=None):
    """Height (up, nm) of the top of the solid (or of `regions`) at (x, z)."""
    g = st.grid
    phi = st.solid() if regions is None else st.phi_union(st.regions_of(names=regions))
    i = int(np.argmin(np.abs(g.axes[0] - x)))
    k = int(np.argmin(np.abs(g.axes[2] - z)))
    t = levelset.column_top(phi[i:i + 1, :, k:k + 1], g.axes[1])[0, 0]
    return -t


class TestRedistance(unittest.TestCase):
    def test_sphere_and_plane(self):
        h = np.array([0.5, 0.25, 0.5])
        ax = [np.arange(n) * hh for n, hh in zip((60, 120, 60), h)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        sph = np.sqrt((X - 15.1) ** 2 + (Y - 14.9) ** 2 + (Z - 14.7) ** 2) - 9.37
        d = levelset.redistance(sph * (1 + 0.5 * np.sin(X / 3.0)) ** 2, h, band=6)
        band = np.abs(sph) < 6
        self.assertLess(np.abs(d - sph)[band].max(), 0.06)
        self.assertLess(np.abs(d - sph)[band].mean(), 0.01)
        pl = (0.3 * X + 0.5 * Y - 0.81 * Z - 2.0) / np.sqrt(0.3 ** 2 + 0.5 ** 2 + 0.81 ** 2)
        d = levelset.redistance(pl * 3.7, h, band=6)
        # away from the walls (there the plane's nearest point is outside the grid)
        inner = np.ones(X.shape, bool)
        for A, L in ((X, 29.5), (Y, 29.75), (Z, 29.5)):
            inner &= (A > 6) & (A < L - 6)
        self.assertLess(np.abs(d - pl)[(np.abs(pl) < 6) & inner].max(), 1e-3)

    def test_polygon_sdf(self):
        x, z = np.meshgrid(np.linspace(-5, 5, 41), np.linspace(-5, 5, 41), indexing="ij")
        d = levelset.polygon_sdf(x, z, levelset.rect_poly(-2, 2, -1, 1))
        exact = np.maximum(np.abs(x) - 2, np.abs(z) - 1)
        out = (np.abs(x) > 2) & (np.abs(z) > 1)
        exact = np.where(out, np.hypot(np.abs(x) - 2, np.abs(z) - 1), exact)
        np.testing.assert_allclose(d, exact, atol=1e-12)


class TestOps(unittest.TestCase):
    def setUp(self):
        # 40 x 40 nm footprint, heights -10 (down) .. 30 (up)
        self.st = Structure(Grid(lo=(0, -30, 0), hi=(40, 10, 40), h=(0.5, 0.25, 0.5)))
        self.st.substrate("Si", "substrate")

    def block(self, height=10.0):
        """A 20 x 20 nm poly block of `height` on the silicon (x, z in [10, 30])."""
        self.st.deposit("Poly", "block", "fill", to_height=height, mask={"rect": [10, 30, 10, 30]})

    def test_substrate_and_fill(self):
        st = self.st
        self.assertAlmostEqual(top_height(st, 5, 5), 0.0, places=6)
        self.block()
        self.assertAlmostEqual(top_height(st, 20, 20), 10.0, places=6)
        self.assertAlmostEqual(top_height(st, 5, 5), 0.0, places=6)
        vol = st.volume("block")
        self.assertAlmostEqual(vol / (20 * 20 * 10), 1.0, delta=0.03)

    def test_conformal_corner_radius(self):
        st, t = self.st, 4.0
        self.block()
        st.deposit("SiN", "liner", "conformal", thickness=t)
        self.assertAlmostEqual(top_height(st, 3, 3), t, delta=0.02)                 # on flat silicon
        self.assertAlmostEqual(top_height(st, 20, 20), 10 + t, delta=0.02)          # on the block
        for dx in (0.5, 1.5, 2.5, 3.5):                                             # rounded top edge
            self.assertAlmostEqual(top_height(st, 30 + dx, 20), 10 + np.sqrt(t * t - dx * dx), delta=0.06,
                                   msg=f"dx={dx}")
        # sidewall thickness at mid height: the liner ends t beyond the block side
        g = st.grid
        j = int(np.argmin(np.abs(g.axes[1] + 5.0)))
        k = int(np.argmin(np.abs(g.axes[2] - 20)))
        phi = levelset.redistance(st.solid(), g.h, band=8)[:, j, k]
        xs = g.axes[0]
        right = xs > 30.0
        self.assertAlmostEqual(line_crossing(phi[right], xs[right]), 30 + t, delta=0.03)

    def test_anisotropic_etch_stops_on_oxide(self):
        st = self.st
        st.deposit("SiO2", "gox", "fill", to_height=2.0)
        st.deposit("Poly", "gate", "fill", to_height=20.0)
        ox0 = st.volume("gox")
        st.pattern("gate", {"rect": [15, 25, -1, 41]})
        st.etch("anisotropic", materials=["Poly"], mask="!gate", stop_on=["SiO2"])
        self.assertAlmostEqual(top_height(st, 20, 20), 20.0, places=4)              # gate kept
        self.assertAlmostEqual(top_height(st, 5, 20), 2.0, places=4)                # down to the oxide
        self.assertEqual(st.volume("gox"), ox0)                                     # oxide untouched
        self.assertAlmostEqual(st.volume("gate") / (10 * 40 * 18), 1.0, delta=0.06)

    def test_spacer(self):
        """Conformal SiN + anisotropic etch leaves a spacer exactly t wide,
        also when the block edge is between nodes (x = 30.3)."""
        st, t = self.st, 5.0
        st.deposit("Poly", "block", "fill", to_height=20.0, mask={"rect": [10, 30.3, 10, 30]})
        st.deposit("SiN", "spacer", "conformal", thickness=t)
        st.etch("anisotropic", materials=["SiN"], depth=1.1 * t)
        self.assertLess(top_height(st, 3, 3), 1e-6)                                 # flat SiN cleared
        self.assertAlmostEqual(top_height(st, 20, 20), 20.0, places=4)              # block top cleared
        g = st.grid
        k = int(np.argmin(np.abs(g.axes[2] - 20)))
        xs = g.axes[0]
        for hgt in (0.5, 5.0, 12.0):
            j = int(np.argmin(np.abs(g.axes[1] + hgt)))
            phi = st.phi_union(st.regions_of(names=["spacer"]))[:, j, k]
            m = xs > 31
            self.assertAlmostEqual(line_crossing(phi[m], xs[m]), 30.3 + t, delta=0.05, msg=f"height {hgt}")

    def test_selective_growth_on_silicon(self):
        st = self.st
        self.block()
        st.deposit("SiO2", "ox", "conformal", thickness=1.5, on=["Si"])
        self.assertAlmostEqual(top_height(st, 3, 3), 1.5, delta=0.02)               # grows on Si
        self.assertAlmostEqual(top_height(st, 20, 20), 10.0, places=4)              # not on the poly

    def test_deposit_must_not_reach_the_ceiling(self):
        with self.assertRaises(ValueError):
            self.st.deposit("SiO2", "ox", "fill", to_height=40.0)            # domain top is 30

    def test_planarize_and_isotropic(self):
        st = self.st
        self.block(height=12.0)
        st.planarize(8.0)
        self.assertAlmostEqual(top_height(st, 20, 20), 8.0, places=4)
        st.etch("isotropic", materials=["Poly"], thickness=2.0)
        self.assertAlmostEqual(top_height(st, 20, 20), 6.0, delta=0.03)
        g = st.grid
        j = int(np.argmin(np.abs(g.axes[1] + 3.0)))
        k = int(np.argmin(np.abs(g.axes[2] - 20)))
        phi = st.phi_union(st.regions_of(names=["block"]))[:, j, k]
        xs = g.axes[0]
        self.assertAlmostEqual(line_crossing(phi[xs > 20], xs[xs > 20]), 28.0, delta=0.05)
        self.assertAlmostEqual(top_height(st, 3, 3), 0.0, places=4)                 # Si not etched

    def test_implant(self):
        st = self.st
        self.block(height=20.0)
        st.implant("As", rp=4.0, sigma=2.0, peak=1e20, lateral_sigma=2.0, blocked_by=["Poly"])
        g = st.grid
        k = int(np.argmin(np.abs(g.axes[2] - 20)))
        i = int(np.argmin(np.abs(g.axes[0] - 3)))
        col = st.donors[i, :, k]
        jpk = int(np.argmax(col))
        self.assertAlmostEqual(g.axes[1][jpk], 4.0, delta=g.h[1])                   # peak at Rp below Si
        self.assertAlmostEqual(col.max() / 1e20, 1.0, delta=0.02)
        j = int(np.argmin(np.abs(g.axes[1] - 4.0)))
        row = st.donors[:, j, k] / 1e20
        xs = g.axes[0]
        self.assertAlmostEqual(np.interp(10.0, xs, row), 0.5, delta=0.02)          # erfc: 1/2 at the edge
        self.assertLess(row[np.argmin(np.abs(xs - 20))], 1e-6)                     # none mid-block
        nd, na, net = st.net_doping()
        self.assertEqual(nd[:, g.axes[1] < 0, :][..., :].max(), 0.0)               # none outside silicon

    def test_epi_doping(self):
        """Peak inside the epi; outside, exp(-dx^2/2sx^2) exp(-dy^2/2sy^2) of the
        distance to the epi (a box here, so exactly separable)."""
        st = self.st
        st.deposit("Si", "sd_epi", "fill", to_height=6, mask={"rect": [0, 10, 0, 40]})
        st.epi_doping("P", 1e20, sx=2.0, sy=4.0, regions=["sd_epi"])
        g = st.grid
        k = int(np.argmin(np.abs(g.axes[2] - 20)))

        def N(x, h):
            return st.donors[int(np.argmin(np.abs(g.axes[0] - x))), int(np.argmin(np.abs(g.axes[1] + h))), k]
        self.assertAlmostEqual(N(5, 3) / 1e20, 1.0)                           # inside
        self.assertAlmostEqual(N(13, 3) / 1e20, np.exp(-0.5 * (3 / 2) ** 2), places=6)     # dx = 3
        self.assertAlmostEqual(N(5, -4) / 1e20, np.exp(-0.5 * (4 / 4) ** 2), places=6)     # dy = 4 below the substrate top
        self.assertAlmostEqual(N(12, -2) / 1e20, np.exp(-0.5 * ((2 / 2) ** 2 + (2 / 4) ** 2)), places=6)

    def test_epi_doping_stops_at_dielectric(self):
        """The tail does not cross a dielectric, however thin (a buried /
        bottom-dielectric layer between the epi and the substrate)."""
        st = self.st
        st.deposit("SiN", "bdi", "fill", to_height=3)
        st.deposit("Si", "sd_epi", "fill", to_height=8, mask={"rect": [0, 10, 0, 40]})
        st.epi_doping("P", 1e20, sx=2.0, sy=4.0, regions=["sd_epi"])
        g = st.grid
        i, k = int(np.argmin(np.abs(g.axes[0] - 5))), int(np.argmin(np.abs(g.axes[2] - 20)))
        nd, na, net = st.net_doping()
        j_epi = int(np.argmin(np.abs(g.axes[1] + 6)))
        self.assertAlmostEqual(nd[i, j_epi, k] / 1e20, 1.0)
        self.assertEqual(nd[:, g.axes[1] > 0.25, :].max(), 0.0)                  # substrate: none

    def test_conformal_cap(self):
        """A conformal deposit with to_height stops there (capped epitaxy)."""
        self.block(10.0)
        self.st.deposit("Si", "epi", "conformal", thickness=4, to_height=6)
        self.assertLessEqual(top_height(self.st, 5, 20, regions=["epi"]), 6.0 + 1e-6)

    def test_dose(self):
        st = Structure(Grid(lo=(0, -5, 0), hi=(4, 60, 4), h=(1, 0.25, 1)))
        st.substrate("Si", "substrate")
        st.implant("B", rp=20.0, sigma=6.0, dose=1e13)
        col = st.acceptors[2, :, 2]
        dose = np.trapezoid(col, st.grid.axes[1] * 1e-7)
        self.assertAlmostEqual(dose / 1e13, 1.0, delta=1e-3)


class TestEpitaxy(unittest.TestCase):
    """Shaped S/D epitaxy (Structure.epitaxy): one 5 nm fin, 28 nm above a
    2 nm oxide, in a 40 nm wide cell with mirror side walls; the epi capped
    at 35 nm."""

    def build(self, shape, lateral, Lz=40.0, fin_top=30.0):
        st = Structure(Grid(lo=(0, -40, 0), hi=(3, 10, Lz), h=(0.5, 0.5, 0.5)))
        st.substrate("Si", "substrate")
        c = Lz / 2
        st.deposit("Si", "fin", "fill", to_height=fin_top, mask={"rect": [-1, 4, c - 2.5, c + 2.5]})
        st.deposit("SiO2", "sti", "fill", to_height=2.0)
        st.epitaxy("SiGe", "epi", on=["Si"], to_height=35.0, shape=shape, lateral=lateral)
        return st

    @staticmethod
    def width(st, height):
        """Epi extent across z (nm, the sub-voxel surface crossings) at
        `height` (on a node row), mid x."""
        g = st.grid
        j = int(np.argmin(np.abs(g.axes[1] + height)))
        f = st.regions[st.region_index("epi")].phi[2, j, :]
        if not (f < 0).any():
            return 0.0
        z = g.axes[2]
        if (f < 0).all():
            return z[-1] - z[0]
        return line_crossing(f[::-1], z[::-1]) - line_crossing(f, z)

    def test_facet_diamond(self):
        st = self.build("facet", 6.0)
        # lower {111} facets from the fin's base corners: 1/tan(54.74 deg)
        # = 0.707 nm wider per nm of height, on both sides
        for h in (4.0, 8.0):                    # (lower, the wedge is thinner than a voxel)
            self.assertAlmostEqual(self.width(st, h), 5.0 + 2 * (h - 2.0) * 0.7071, delta=0.01, msg=f"h={h}")
        self.assertAlmostEqual(self.width(st, 18.0), 5.0 + 2 * 6.0, delta=0.01)       # the vertical sides
        self.assertAlmostEqual(self.width(st, 34.5), 5.0 + 2 * 0.5 * 0.7071, delta=0.01)   # (100) top: fin wide
        self.assertAlmostEqual(self.width(st, 33.0), 5.0 + 2 * 2.0 * 0.7071, delta=0.01)
        self.assertEqual(self.width(st, 36.0), 0.0)
        self.assertEqual(_non_wedge(st.check()), [])

    def test_ellipse_blob(self):
        st = self.build("ellipse", 5.0)
        widths = [self.width(st, h) for h in np.arange(2.5, 35.0, 0.5)]
        self.assertAlmostEqual(max(widths), 5.0 + 2 * 5.0, delta=0.02)                # lateral beyond the fin
        w = self.width(st, 2.5)                                                       # the foot: half way out
        self.assertTrue(10.0 < w < 10.5, w)
        self.assertGreater(33.0, 2 * max(widths))                                     # taller than wide
        self.assertEqual(self.width(st, 36.0), 0.0)

    def test_merge_follows_pitch(self):
        # 24 nm pitch: 6 nm beyond the fin leaves 7 nm to the mirrored
        # neighbour, 11 nm overlaps it by 3 nm
        self.assertAlmostEqual(self.width(self.build("facet", 6.0, Lz=24.0), 18.0), 17.0, delta=0.01)
        st = self.build("facet", 11.0, Lz=24.0)
        lab = st.node_labels()
        j = int(np.argmin(np.abs(st.grid.axes[1] + 18.0)))
        self.assertTrue(np.isin(lab[2, j, :], [st.region_index("epi"), st.region_index("fin")]).all())
        self.assertEqual(lab[2, j, 0], st.region_index("epi"))                       # merged with the mirror
        # the void under the merge: no epi at the wall just above the oxide
        j = int(np.argmin(np.abs(st.grid.axes[1] + 4.0)))
        self.assertNotEqual(lab[2, j, 0], st.region_index("epi"))

    def test_contact_plug(self):
        """A bullet into the epi: straight sides `shell` inside its edges at
        the trench floor, a round bottom `depth` below the floor."""
        st = self.build("ellipse", 5.0, fin_top=8.0)                        # a recessed fin
        w0 = self.width(st, 30.0)                                           # the epi at the floor
        st.contact_plug(["epi"], {"rect": [-1, 4, -1, 41]}, floor_height=30.0, depth=12.0, shell=2.0)
        g = st.grid
        f = st.regions[st.region_index("epi")].phi[2]
        z = g.axes[2]
        c = 20.0
        k = int(np.argmin(np.abs(z - c)))
        col = f[:, k]
        # down the plug's axis: no epi from the floor to the tip, epi below it
        self.assertAlmostEqual(-line_crossing(col, g.axes[1]), 30.0 - 12.0, delta=0.02)
        r = 0.5 * round((w0 - 4.0) / g.h[2]) * g.h[2]                       # (the epi edge in nodes)
        j = int(np.argmin(np.abs(g.axes[1] + 25.0)))                       # straight part of the plug
        inner = line_crossing(f[j, k:], z[k:])
        self.assertAlmostEqual(inner - c, r, delta=0.3)
        self.assertGreater(self.width(st, 25.0), 2 * r)                    # the epi shell is still there

    def test_epi_seed(self):
        """The bilayer epi's seed: the first 5 nm grown on the silicon, cut
        out of the epi without changing its outer shape (the union field is
        the old epi field, node for node), on a flat face exactly 5 nm."""
        st = self.build("facet", 5.0, fin_top=8.0)                          # a recessed fin
        old = st.regions[st.region_index("epi")].phi.copy()
        lab0 = st.cell_labels()
        n_epi = (lab0 == st.region_index("epi")).sum()
        st.epi_seed("epi", "seed", "SiGe0.2", ["Si"], 5.0)
        seed, epi = st.regions[st.region_index("seed")], st.regions[st.region_index("epi")]
        self.assertEqual((seed.material, seed.parent), ("SiGe0.2", "epi"))
        np.testing.assert_array_equal(np.minimum(seed.phi, epi.phi), old)
        lab = st.cell_labels()
        ks, ke = st.region_index("seed"), st.region_index("epi")
        self.assertEqual((lab == ks).sum() + (lab == ke).sum(), n_epi)
        g = st.grid
        k = int(np.argmin(np.abs(g.axes[2] - 20.0)))
        col = lab[2, :, k]                                                  # up the fin's axis
        ys = -(g.axes[1][:-1] + 0.5 * g.h[1])                               # cell centre heights
        self.assertTrue((col[(ys > 8.0) & (ys < 13.0)] == ks).all())           # (to half a voxel)
        self.assertTrue((col[(ys > 13.5) & (ys < 35.0)] == ke).all())
        # no new one-voxel layers: the split's thin cells are the old epi's
        # (the diamond's acute tip against the fin side)
        from fab.structure import _thin_cells
        thin_after = _thin_cells(lab == ks) | _thin_cells(lab == ke)
        self.assertFalse((thin_after & ~_thin_cells(lab0 == st.region_index("epi"))).any())

    def test_sliver_merge_rejected(self):
        with self.assertRaisesRegex(ValueError, "mirror image"):
            self.build("facet", 9.7, Lz=24.0)                                         # 0.4 nm overlap


class TestDespeckle(unittest.TestCase):
    def test_flake_removed_film_kept(self):
        from fab.structure import _despeckle
        lab = np.full((5, 7, 5), -1, dtype=np.int16)
        lab[:, :3, :] = 0                      # silicon
        lab[:, 3, :] = 1                       # one-voxel film between silicon and gate: kept
        lab[:, 4:, :] = 2
        lab[2, 5, 2] = -1                      # a lone gas voxel inside the gate: removed
        out, n = _despeckle(lab)
        self.assertEqual(n, 1)
        self.assertTrue((out[:, 3, :] == 1).all())
        self.assertEqual(out[2, 5, 2], 2)


class TestExpr(unittest.TestCase):
    def test_evaluate(self):
        p = {"Lg": "30nm", "sp": 8, "W": "0.06um"}
        self.assertEqual(evaluate("Lg/2 + sp", p), 23.0)
        self.assertEqual(evaluate("W", p), 60.0)
        self.assertEqual(evaluate("max(W, 2*Lg) - 5A", p), 59.5)
        self.assertEqual(evaluate(3, p), 3.0)
        for bad in ("__import__('os')", "Lg.real", "open('x')", "unknown + 1", "[1, 2]"):
            with self.assertRaises(ExprError, msg=bad):
                evaluate(bad, p)

    def test_resolve(self):
        r = resolve_params({"a": "2nm", "b": "a*3", "mat": "SiN", "opts": {"x": 1}})
        self.assertEqual((r["a"], r["b"], r["mat"], r["opts"]), (2.0, 6.0, "SiN", {"x": 1}))


class TestPlanar(unittest.TestCase):
    """The planar template: the built structure matches its own parameters
    at the centre cross-section (z = Lz/2), and the flow round-trips through
    the editable layer-2 YAML."""

    @classmethod
    def setUpClass(cls):
        from fab import flow
        cls.cfg = flow.load(os.path.join(TCAD1D_ROOT, "configs", "fab_planar.yaml"))
        cls.st, cls.times = flow.build(cls.cfg, log=lambda *a: None)
        cls.p = flow.resolve_params(flow.expand(cls.cfg)["params"])

    def crossing(self, region, axis, at, along_range):
        """Sub-voxel zero crossing of a region's phi along `axis` through the
        point `at` (dict axis -> value), searched within along_range."""
        g = self.st.grid
        phi = self.st.phi_union(self.st.regions_of(names=[region]))
        idx = [int(np.argmin(np.abs(g.axes[a] - at[a]))) if a != axis else slice(None) for a in range(3)]
        f, c = phi[tuple(idx)], g.axes[axis]
        m = (c >= along_range[0]) & (c <= along_range[1])
        return line_crossing(f[m], c[m])

    def test_gate_stack(self):
        p, zc = self.p, self.p["Lz"] / 2
        xc = (p["xg0"] + p["xg1"]) / 2
        y_mid = -(p["tox"] + p["gate_height"] / 2)
        self.assertAlmostEqual(self.crossing("gate", 0, {1: y_mid, 2: zc}, (0, xc)), p["xg0"], delta=0.05)
        self.assertAlmostEqual(self.crossing("gate", 0, {1: y_mid, 2: zc}, (xc, p["Lx"])), p["xg1"], delta=0.05)
        self.assertAlmostEqual(top_height(self.st, xc, zc, ["gate"]), p["tox"] + p["gate_height"], delta=0.02)
        self.assertAlmostEqual(top_height(self.st, xc, zc, ["gate_oxide"]), p["tox"], delta=0.02)
        self.assertAlmostEqual(top_height(self.st, xc, zc, ["substrate"]), 0.0, delta=0.02)

    def test_spacer(self):
        p, zc = self.p, self.p["Lz"] / 2
        y = -(p["tox"] + 1.0)                     # just above the oxide
        outer = self.crossing("spacer", 0, {1: y, 2: zc}, (p["xg1"] + 0.5, p["Lx"]))
        self.assertAlmostEqual(outer - p["xg1"], p["spacer"], delta=0.05)

    def test_sti(self):
        p = self.p
        z_sti = p["sti_width"] / 2
        x = p["xg0"] / 2
        self.assertAlmostEqual(top_height(self.st, x, z_sti, ["sti"]), 0.0, delta=0.05)  # flush: the strip leaves the STI alone
        bottom = self.crossing("sti", 1, {0: x, 2: z_sti}, (5, p["sti_depth"] + 10))   # through the STI below its top
        self.assertAlmostEqual(bottom, p["sti_depth"], delta=0.05)
        y = 10.0                                    # active-area edges in z, inside the silicon
        self.assertAlmostEqual(self.crossing("sti", 2, {0: x, 1: y}, (0, p["Lz"] / 2)), p["zw0"], delta=0.05)
        self.assertAlmostEqual(self.crossing("sti", 2, {0: x, 1: y}, (p["Lz"] / 2, p["Lz"])), p["zw1"], delta=0.05)

    def test_junction_depth(self):
        """Deep S/D junction far from the gate = where the analytic Gaussians
        balance (the doping is analytic, so this checks the implant depth
        reference: the silicon surface after the dielectric strip)."""
        p, g = self.p, self.st.grid
        x, zc = p["xg0"] / 2, p["Lz"] / 2
        _, _, net = self.st.net_doping()
        i, k = int(np.argmin(np.abs(g.axes[0] - x))), int(np.argmin(np.abs(g.axes[2] - zc)))
        y = g.axes[1]
        m = (y > 1) & (y < p["sti_depth"])
        got = line_crossing(net[i, m, k], y[m])

        def gauss(d, peak, rp, sg):
            return peak * np.exp(-0.5 * ((d - rp) / sg) ** 2)
        yy = np.linspace(1, p["sti_depth"], 200001)
        n = gauss(yy, p["sd_peak"], p["sd_rp"], p["sd_sigma"]) + gauss(yy, p["ext_peak"], p["ext_rp"], p["ext_sigma"])
        a = gauss(yy, p["well_peak"], p["well_rp"], p["well_sigma"]) + p["sub_conc"]
        want = yy[np.argmax(n - a < 0)]
        self.assertAlmostEqual(got, want, delta=0.1)
        self.assertGreater(want, 20)                # a ~30 nm deep junction for a 30 nm gate
        self.assertLess(want, 40)

    def test_partition(self):
        """Every voxel is in exactly one region or gas; volumes add up."""
        lab = self.st.cell_labels()
        names = [r.name for r in self.st.regions]
        self.assertEqual(names, ["substrate", "sti", "gate_oxide", "gate", "spacer", "ild", "source_contact",
                                 "drain_contact"])
        self.assertTrue(((lab >= -1) & (lab < len(names))).all())
        g, p = self.st.grid, self.p
        heights = -(g.axes[1][:-1] + g.h[1] / 2)
        below = heights < p["ild_top"]                      # ILD fills everything: no gas below its top
        self.assertEqual(int((lab[:, below, :] == -1).sum()), 0)
        self.assertTrue((lab[:, ~below, :] == -1).all())

    def test_contacts_taller_than_gate(self):
        """Contacts rise to the ILD top, above the gate; ILD (not gas)
        between spacer and contact."""
        lab, g, p = self.st.cell_labels(), self.st.grid, self.p
        k = {r.name: i for i, r in enumerate(self.st.regions)}
        heights = -(g.axes[1][:-1] + g.h[1] / 2)
        top = lambda name: heights[(lab == k[name]).any(axis=(0, 2))].max()
        self.assertAlmostEqual(top("source_contact") + g.h[1] / 2, p["ild_top"], delta=1e-9)
        self.assertGreater(top("source_contact"), top("gate") + 5)
        x_gap = int(p["xg0"] - p["spacer"] - 2)             # between spacer and source contact
        col = lab[x_gap, (heights > 0) & (heights < p["ild_top"]), int(g.shape[2] // 2)]
        self.assertTrue((col == k["ild"]).all())

    def test_structure_check(self):
        """No voids, fragments or one-voxel layers; STI flush with the
        silicon (no recess step) and the gate bottom flat at tox."""
        issues = [w for w in self.st.check() if "flake" not in w]
        self.assertEqual(issues, [])
        lab = self.st.cell_labels()
        g = self.st.grid
        k = {r.name: i for i, r in enumerate(self.st.regions)}
        h = -(g.axes[1][:-1] + g.h[1] / 2)                   # cell-centre heights
        sti_col = lab[5, :, 2]                              # under the source, in the STI
        self.assertAlmostEqual(h[np.argmax(sti_col == k["sti"])], 0.0, delta=g.h[1])
        gx = int(g.shape[0] // 2)
        bottom = [h[np.argmax(lab[gx, :, z] == k["gate"])] for z in range(lab.shape[2])
                  if (lab[gx, :, z] == k["gate"]).any()]
        self.assertLess(max(bottom) - min(bottom), 1e-9)    # one level over channel and STI alike

    def test_layer2_roundtrip(self):
        """The expanded (layer 2/3) YAML rebuilds the same structure."""
        import yaml
        from fab import flow
        text = yaml.safe_dump(flow.expand(self.cfg), sort_keys=False)
        st2, _ = flow.build(yaml.safe_load(text), log=lambda *a: None)
        self.assertEqual([r.name for r in st2.regions], [r.name for r in self.st.regions])
        for a, b in zip(st2.regions, self.st.regions):
            np.testing.assert_array_equal(a.phi, b.phi)
        np.testing.assert_array_equal(st2.donors, self.st.donors)

    def test_snapshots_in_viewer(self):
        """Snapshots load in tcadview's model: steps, regions, material
        field, smooth surfaces, doping NaN outside silicon, probes."""
        import tempfile
        try:
            from tcadview import model
        except ImportError:
            self.skipTest("viewer dependencies not installed")
        from fab import flow
        from fab.export import SnapshotWriter
        with tempfile.TemporaryDirectory() as t:
            w = SnapshotWriter(t)
            flow.build(self.cfg, on_step=w, log=lambda *a: None)
            sess = model.Session(w.finish())
            self.assertEqual(len(sess), len(flow.expand(self.cfg)["process"]))
            self.assertTrue(sess.states[7].name.endswith("gate dielectric"))
            st0, st = sess.state(0), sess.state(len(sess) - 1)
            self.assertTrue(st.is_process)
            self.assertEqual(st0.region_names, ["substrate"])
            self.assertEqual(st.fields()[0], model.MATERIAL_FIELD)
            self.assertIn("DopingConcentration", st.fields())
            for k in range(len(st.region_names)):
                self.assertIsNotNone(st.surface(k), st.region_names[k])
            p = self.p
            xc, zc = (p["xg0"] + p["xg1"]) / 2e3, p["Lz"] / 2e3
            r = st.probe([xc, -p["tox"] / 2e3, zc])               # inside the gate oxide
            self.assertEqual(r["region"], "gate_oxide")
            self.assertTrue(np.isnan(r["fields"]["DopingConcentration"]))
            r = st.probe([xc, 0.01, zc])                           # channel, 10 nm deep
            self.assertEqual(r["region"], "substrate")
            self.assertLess(r["fields"]["DopingConcentration"], -1e17)

    def test_pmos(self):
        from fab import flow
        cfg = {"template": "planar", "params": {"type": "pmos", "voxel": [2.0, 1.0, 2.0]}}
        st, _ = flow.build(cfg, log=lambda *a: None)
        nd, na, net = st.net_doping()
        self.assertGreater(na.max(), 5e19)                         # p+ source/drain
        self.assertGreater(nd.max(), 1e18)                         # n well



class TestFinFET(unittest.TestCase):
    """The FinFET template: fin width and height, the gate wrapping the fin,
    the recessed and doped S/D epi, a clean structure check."""

    @classmethod
    def setUpClass(cls):
        from fab import flow
        cls.cfg = flow.load(os.path.join(TCAD1D_ROOT, "configs", "fab_finfet.yaml"))
        cls.st, _ = flow.build(cls.cfg, log=lambda *a: None)
        cls.p = flow.resolve_params(flow.expand(cls.cfg)["params"])

    def test_structure_check(self):
        # Gate-last (RMG): where the HK lining the gate trench wall meets the
        # HK wrapping the fin's top edge, the two rounded conformal coats leave
        # a few one-voxel metal cells (6 at the default size); the check
        # relabels them to their surroundings, so the labels a device mesh is
        # built from are clean, as in the planar flow.
        issues = [w for w in _non_wedge(self.st.check()) if "flake" not in w]
        self.assertEqual(issues, [])
        self.assertLessEqual(sum(int(w.split()[0]) for w in self.st.check() if "flake" in w), 12)

    def test_fin_and_gate_wrap(self):
        st, p = self.st, self.p
        g = st.grid
        lab = st.cell_labels()
        names = [r.name for r in st.regions]
        i = int(np.argmin(np.abs(g.axes[0] - p["cpp"] / 2)))
        j = int(np.argmin(np.abs(g.axes[1] + (-p["H_fin"] / 2))))       # mid fin height
        row = [names[l] if l >= 0 else "gas" for l in lab[i, j, :]]
        si = [k for k, r in enumerate(row) if r == "substrate"]
        self.assertAlmostEqual(len(si) * g.h[2], p["W_fin"], delta=1e-9)            # exact, in cells
        self.assertEqual(si[-1] - si[0] + 1, len(si))
        self.assertEqual(row[si[0] - 1], "gate_oxide")
        self.assertEqual(row[si[-1] + 1], "gate_oxide")
        self.assertIn("gate", row[:si[0] - 1])
        self.assertIn("gate", row[si[-1] + 2:])
        self.assertAlmostEqual(top_height(st, p["cpp"] / 2, p["fin_pitch"] / 2, regions=["substrate"]), 0.0,
                               delta=g.h[1])

    def test_epi(self):
        st, p = self.st, self.p
        nd, na, net = st.net_doping()
        g = st.grid
        k = int(np.argmin(np.abs(g.axes[2] - p["fin_pitch"] / 2)))
        j = int(np.argmin(np.abs(g.axes[1] + (-20))))                      # (below the contact plug)
        self.assertGreater(net[int(np.argmin(np.abs(g.axes[0] - 5))), j, k], 5e20)          # in the epi
        self.assertLess(net[int(np.argmin(np.abs(g.axes[0] - p["cpp"] / 2))), j, k], 0)     # channel p
        # the epi stops above the punch-through stopper: none below the STI top
        epi = st.phi_union(st.regions_of(names=["sd_epi"])) < -1e-3
        self.assertFalse(epi[:, g.axes[1] > p["H_fin"] - 1e-6, :].any())

    def test_epi_shapes(self):
        """NMOS: an elliptical blob 5 nm beyond the fin, apart from the
        neighbouring fins' epi; PMOS: a {111} diamond that merges with them
        at the 24 nm fin pitch."""
        from fab import flow
        p = self.p
        k = self.st.region_index("sd_epi")
        lab = self.st.node_labels()
        g = self.st.grid
        i = int(np.argmin(np.abs(g.axes[0] - 10.0)))                     # S/D, outside the contact
        rows = [(lab[i, j, :] == k).sum() * g.h[2] for j in range(g.shape[1])]
        self.assertAlmostEqual(max(rows), p["W_fin"] + 2 * 5.0, delta=1.0)
        cfg = flow.load(os.path.join(TCAD1D_ROOT, "configs", "fab_finfet_pmos.yaml"))
        st, _ = flow.build(cfg, log=lambda *a: None)
        # flakes: the RMG trench corners, see TestFinFET.test_structure_check
        self.assertEqual([w for w in _non_wedge(st.check()) if "flake" not in w], [])
        lab = st.node_labels()
        epi = [st.region_index(n) for n in ("sd_epi", "sd_epi_seed")]        # fill + bilayer seed
        merged = np.isin(lab[i], epi).all(axis=1)                         # rows across the whole cell
        self.assertTrue(merged.any())


class TestNanosheet(unittest.TestCase):
    """The nanosheet template: three released sheets wrapped by the gate
    dielectric and metal, inner spacers between them under the spacers."""

    @classmethod
    def setUpClass(cls):
        from fab import flow
        cls.cfg = flow.load(os.path.join(TCAD1D_ROOT, "configs", "fab_nanosheet.yaml"))
        cls.st, _ = flow.build(cls.cfg, log=lambda *a: None)
        cls.p = flow.resolve_params(flow.expand(cls.cfg)["params"])

    def test_sheets(self):
        st, p = self.st, self.p
        g = st.grid
        lab = st.node_labels()
        names = [r.name for r in st.regions]
        i = int(np.argmin(np.abs(g.axes[0] - p["cpp"] / 2)))
        k = int(np.argmin(np.abs(g.axes[2] - p["stack_pitch"] / 2)))
        col = [names[l] if l >= 0 else "gas" for l in lab[i, :, k]]
        h = -g.axes[1]
        sheet = np.array([c == "sheets" for c in col])
        runs = np.flatnonzero(np.diff(sheet.astype(int)))                  # sheet edges along the column
        self.assertEqual(len(runs), 2 * p["n_sheets"])
        tops = h[runs[0::2] + 1]
        self.assertTrue(np.allclose(np.diff(sorted(tops)), p["t_sheet"] + p["t_sige"], atol=g.h[1]))
        self.assertNotIn("sige", names)                                    # fully released
        between = [c for c, hh in zip(col, h) if -p["t_sheet"] - p["t_sige"] + 1 < hh < -p["t_sheet"] - 1]
        self.assertIn("gate", between)                                     # metal between the sheets
        self.assertIn("gate_oxide", between)

    def test_inner_spacers(self):
        st, p = self.st, self.p
        g = st.grid
        lab = st.node_labels()
        names = [r.name for r in st.regions]
        x = p["xg0"] - p["inner_spacer"] / 2                               # under the outer spacer
        i = int(np.argmin(np.abs(g.axes[0] - x)))
        j = int(np.argmin(np.abs(g.axes[1] - (p["t_sheet"] + p["t_sige"] / 2))))   # mid first gap
        k = int(np.argmin(np.abs(g.axes[2] - p["stack_pitch"] / 2)))
        self.assertEqual(names[lab[i, j, k]], "inner_spacer")



    def test_epi_unmerged(self):
        """The 35 nm NMOS blob and the 39 nm PMOS diamond leave the 46 nm
        stack pitch unmerged."""
        from fab import flow
        p = self.p
        for dev, width in (("nanosheet", p["W_sheet"] + 10), ("nanosheet_pmos", p["W_sheet"] + 14)):
            st = self.st if dev == "nanosheet" else flow.build(
                flow.load(os.path.join(TCAD1D_ROOT, "configs", f"fab_{dev}.yaml")), log=lambda *a: None)[0]
            g = st.grid
            lab = st.node_labels()
            i = int(np.argmin(np.abs(g.axes[0] - 10.0)))
            epi = lab[i] == st.region_index("sd_epi")
            self.assertFalse(epi[:, 0].any() or epi[:, -1].any(), dev)              # off the side walls
            z = g.axes[2][epi.any(axis=0)]
            self.assertAlmostEqual(z.max() - z.min(), width, delta=1.0, msg=dev)


class TestNanosheetBDI(unittest.TestCase):
    """isolation: bdi - a SiN layer under the whole stack (gate and S/D), so
    the S/D epi meets only the sheet ends and none of its doping reaches the
    substrate."""

    @classmethod
    def setUpClass(cls):
        from fab import flow
        cfg = flow.load(os.path.join(TCAD1D_ROOT, "configs", "fab_nanosheet_bdi.yaml"))
        cls.st, _ = flow.build(cfg, log=lambda *a: None)
        cls.p = flow.resolve_params(flow.expand(cfg)["params"])

    def test_bdi_under_stack(self):
        st, p = self.st, self.p
        g = st.grid
        lab = st.node_labels()
        names = [r.name for r in st.regions]
        self.assertNotIn("bdi_sacrificial", names)
        j = int(np.argmin(np.abs(g.axes[1] - (p["H_stack"] + p["t_bdi"] / 2))))
        k = int(np.argmin(np.abs(g.axes[2] - p["stack_pitch"] / 2)))
        for x in (p["cpp"] / 2, 3.0, p["cpp"] - 3.0):                         # gate centre, source, drain
            i = int(np.argmin(np.abs(g.axes[0] - x)))
            self.assertEqual(names[lab[i, j, k]], "bdi")

    def test_no_epi_doping_in_substrate(self):
        st = self.st
        lab = st.cell_labels()
        names = [r.name for r in st.regions]
        nd, na, net = st.net_doping()
        sub = st.node_labels() == names.index("substrate")
        self.assertEqual(nd[sub].max(), 0.0)

if __name__ == "__main__":
    unittest.main()
