"""tcadview (the Qt viewer): the data model on real exported states, and
the window driven headless (off-screen VTK, no display needed). Skipped when
the viewer dependencies (PySide6, pyvista, pyvistaqt) are not installed.

Run with:  python3 testsuite/test_tcadview.py [-v]
"""
import os
import sys
import tempfile
import unittest

import numpy as np

TCAD1D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TCAD1D_ROOT)

try:
    import pyvista  # noqa: F401
    import pyvistaqt  # noqa: F401
    from PySide6 import QtCore, QtWidgets
    HAVE_VIEWER = True
except ImportError:
    HAVE_VIEWER = False


def _write_sweep(tmp, extrude=False):
    """A 3-state Vg 'sweep' of the 2D MOSFET (equilibrium with shifted psi,
    so states differ) - enough to test the viewer without a full solve."""
    from solver2d.vtk_export import write_sweep
    cwd = os.getcwd()
    os.chdir(TCAD1D_ROOT)
    try:
        import main2d_mosfet_sweep as M
        mesh, mat, _, _, eq, _ = M._context(M.DEFAULT_PATH)
    finally:
        os.chdir(cwd)
    nz = 1
    if extrude:
        from mesh3d.extrude import extrude_mesh
        nz = 4
        mesh = extrude_mesh(mesh, mat, 0.1e-4, nz)
    base = {k: np.tile(v, nz) for k, v in eq.items() if k in ("psi", "n", "p", "phin", "phip")}
    states = []
    for vg in (0.0, 0.5, 1.0):
        s = dict(base, psi=base["psi"] + vg)
        states.append(dict(state=s, bias=vg, label=f"Vgs={vg:+.3f}V", contact_bias={"gate": vg}))
    return write_sweep(tmp, "sweep", mesh, mat, states), mesh, base


@unittest.skipIf(not HAVE_VIEWER, "viewer dependencies not installed (pip install -r requirements-viewer.txt)")
class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pvd, cls.mesh, cls.base = _write_sweep(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_session(self):
        from tcadview.model import Session
        s = Session(self.pvd)
        self.assertEqual([i.bias for i in s.states], [0.0, 0.5, 1.0])
        st = s.state(2)
        self.assertEqual(st.dim, 2)
        self.assertEqual(st.region_names, ["substrate", "source", "drain", "oxide"])
        self.assertEqual([st.region_material(k) for k in range(4)], ["Silicon"] * 3 + ["SiO2"])
        self.assertEqual(st.contact_bias["gate"], 1.0)
        self.assertEqual(st.units["eDensity"], "cm^-3")
        self.assertEqual(sum(st.region(k).n_cells for k in range(4)), st.grid.n_cells)

    def test_scales(self):
        from tcadview.model import scale_values
        a = np.array([-1e18, 0.0, 1e16, np.nan])
        np.testing.assert_allclose(scale_values(a, "log"), [18, np.nan, 16, np.nan])
        np.testing.assert_allclose(scale_values(a, "signed log")[[0, 1, 2]], [-18, 0, 16], atol=1e-6)
        np.testing.assert_array_equal(scale_values(a, "linear"), a)

    def test_range_ignores_oxide_nan(self):
        from tcadview.model import Session
        st = Session(self.pvd).state(0)
        lo, hi = st.range("eDensity", "log")
        n = self.base["n"][~self.mesh.is_insulator]
        self.assertAlmostEqual(lo, np.log10(n.min()), places=6)
        self.assertAlmostEqual(hi, np.log10(n.max()), places=6)
        # oxide cells: interior nodes are NaN, only the shared Si/SiO2
        # interface nodes carry a density, and the range comes from those
        ox = st.values("eDensity", st.region(3))
        self.assertTrue(np.isnan(ox).any() and np.isfinite(ox).any())
        lo3, hi3 = st.range("eDensity", "log", regions=[3])
        fin = np.log10(ox[np.isfinite(ox)])
        self.assertEqual((lo3, hi3), (fin.min(), fin.max()))

    def test_probe_matches_nodes(self):
        from tcadview.model import Session
        st = Session(self.pvd).state(1)
        P = self.mesh.points * 1e4
        for i in np.linspace(0, len(P) - 1, 25).astype(int):
            r = st.probe([P[i, 0], P[i, 1], 0.0])
            self.assertIsNotNone(r)
            self.assertAlmostEqual(r["fields"]["ElectrostaticPotential"], self.base["psi"][i] + 0.5, places=9)
        self.assertIsNone(st.probe([100.0, 100.0, 0.0]))

    def test_cut_line(self):
        from tcadview.model import Session
        st = Session(self.pvd).state(0)
        d, v = st.cut_line([0.55, -0.005, 0], [0.55, 0.2, 0], "eDensity", n=50)
        self.assertAlmostEqual(d[-1], 0.205)
        self.assertTrue(np.isnan(v[0]), "starts in the oxide (no carriers)")
        self.assertTrue(np.all(np.isfinite(v[d > 0.006])) and np.all(v[d > 0.006] > 0))
        d, v = st.cut_line([2.0, 0.1, 0], [3.0, 0.1, 0], "ElectrostaticPotential", n=10)
        self.assertTrue(np.all(np.isnan(v)), "outside the mesh")


@unittest.skipIf(not HAVE_VIEWER, "viewer dependencies not installed")
class TestWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pvd2, _, _ = _write_sweep(os.path.join(cls.tmp.name, "a"))
        cls.pvd3, _, _ = _write_sweep(os.path.join(cls.tmp.name, "b"), extrude=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def window(self, path):
        from tcadview.app import MainWindow
        w = MainWindow(path, off_screen=True)
        self.addCleanup(w.close)
        return w

    def test_2d_controls(self):
        w = self.window(self.pvd2)
        self.assertEqual(w.region_tree.topLevelItemCount(), 4)
        self.assertTrue(w.bias_box.isVisibleTo(w))
        w.field_combo.setCurrentText("eDensity")
        self.assertEqual(w.scale, "log")
        w.bias_slider.setValue(2)
        self.assertIn("Vgs=+1.000V", w.bias_label.text())
        w.field_combo.setCurrentText("ElectrostaticPotential")
        w.region_tree.topLevelItem(3).setCheckState(0, QtCore.Qt.Unchecked)
        self.assertEqual(w.visible_regions(), [0, 1, 2])
        w.cut_p0.setText("0.55, 0.0")
        w.cut_p1.setText("0.55, 0.3")
        w._on_cut_plot()
        self.assertIsNotNone(w.cut)
        self.assertGreater(len(w.cutplot.ax.lines), 0)
        res = w.probe([0.55, 0.01, 0.0])
        self.assertEqual(res["region"], "substrate")
        with tempfile.TemporaryDirectory() as t:
            png = os.path.join(t, "s.png")
            w.plotter.screenshot(png)
            self.assertGreater(os.path.getsize(png), 10000)

    def test_bias_step_shifts_range(self):
        w = self.window(self.pvd2)
        w.field_combo.setCurrentText("ElectrostaticPotential")
        lo0, hi0 = float(w.rmin.text()), float(w.rmax.text())
        w.bias_slider.setValue(2)               # psi + 1 V everywhere
        self.assertAlmostEqual(float(w.rmin.text()), lo0 + 1.0, places=4)
        self.assertAlmostEqual(float(w.rmax.text()), hi0 + 1.0, places=4)
        w.auto_range.setChecked(False)
        w.rmin.setText("0")
        w.rmax.setText("2")
        w._on_manual_range()
        self.assertEqual(w.clim(), (0.0, 2.0))

    def test_3d_slices(self):
        w = self.window(self.pvd3)
        self.assertTrue(w.slice_box.isVisibleTo(w))
        n_surface = len(w.plotter.renderer.actors)
        w.slice_on["z"].setChecked(True)
        w.slice_on["x"].setChecked(True)
        geoms = w._display_geometry(0)
        self.assertEqual(len(geoms), 2)
        self.assertTrue(all(g.n_points > 0 for g in geoms))
        self.assertGreater(len(w.plotter.renderer.actors), n_surface)
        z = w._slice_origin("z")
        self.assertTrue(np.allclose(geoms[0].points[:, 2], z) or np.allclose(geoms[1].points[:, 2], z))


if __name__ == "__main__":
    unittest.main()
