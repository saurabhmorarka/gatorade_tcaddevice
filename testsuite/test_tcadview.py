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
    import pyvista
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
class TestSection(unittest.TestCase):
    """Section views of an extruded 3D state: the flattened plane, and that
    probes/cuts made in it equal the 3D samples at the same points."""

    @classmethod
    def setUpClass(cls):
        from tcadview.model import Session
        cls.tmp = tempfile.TemporaryDirectory()
        pvd, _, _ = _write_sweep(cls.tmp.name, extrude=True)
        cls.st = Session(pvd).state(1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_mapping(self):
        from tcadview.model import Section
        for ax, (u, v) in (("x", (2, 1)), ("y", (0, 2)), ("z", (0, 1))):
            sec = Section(self.st, ax, 0.037)
            p = sec.to3d([0.011, 0.022])
            self.assertEqual(p["xyz".index(ax)], 0.037)
            self.assertEqual((p[u], p[v]), (0.011, 0.022))
            np.testing.assert_array_equal(sec.to2d(p), [0.011, 0.022])
            self.assertTrue(sec.contains(p))
            self.assertFalse(sec.contains(p + 1e-3))

    def test_flat_regions_carry_fields(self):
        from tcadview.model import Section
        sec = Section(self.st, "z", 0.05)
        n = 0
        for k in range(len(self.st.region_names)):
            g = sec.region(k)
            self.assertIsNotNone(g)
            self.assertTrue(np.all(g.points[:, 2] == 0.0))
            self.assertIn("ElectrostaticPotential", g.point_data.keys())
            n += g.n_cells
        self.assertGreater(n, 100)
        self.assertIsNone(Section(self.st, "z", 5.0).region(0), "plane misses the device")

    def test_section_probe_is_3d_probe(self):
        from tcadview.model import Section
        for ax, uv in (("z", (0.55, 0.05)), ("x", (0.03, 0.05)), ("y", (0.55, 0.06))):
            sec = Section(self.st, ax, 0.05 if ax != "x" else 0.55)
            p = sec.to3d(uv)
            a = self.st.probe(p)
            self.assertIsNotNone(a, ax)
            # the displayed (flattened) value agrees with the 3D sample too
            g = pyvista.PolyData(np.array([[uv[0], uv[1], 0.0]])).sample(
                sec.region(self.st.region_names.index(a["region"])), tolerance=1e-9)
            self.assertAlmostEqual(float(g.point_data["ElectrostaticPotential"][0]),
                                   a["fields"]["ElectrostaticPotential"], places=6)

    def test_axis_cuts(self):
        from tcadview.model import Section, cut_axis
        sec = Section(self.st, "z", 0.05)
        p0, p1 = sec.axis_cut([0.55, 0.05], "v")
        b = self.st.grid.bounds
        np.testing.assert_allclose(p0, [0.55, b[2], 0.05])
        np.testing.assert_allclose(p1, [0.55, b[3], 0.05])
        self.assertEqual(cut_axis(p0, p1), 1)
        p0, p1 = sec.axis_cut([0.55, 0.05], "h")
        self.assertEqual(cut_axis(p0, p1), 0)
        self.assertIsNone(cut_axis([0, 0, 0], [1, 1, 0]))
        # the vertical cut equals the same cut entered by hand in 3D
        d, v = self.st.cut_line(p0, p1, "eDensity", n=60)
        d2, v2 = self.st.cut_line([b[0], 0.05, 0.05], [b[1], 0.05, 0.05], "eDensity", n=60)
        np.testing.assert_array_equal(v, v2)
        self.assertTrue(np.isfinite(v).sum() > 30, "inside the silicon")


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

    def test_section_view_probe_and_cuts(self):
        w = self.window(self.pvd3)
        self.assertFalse(w.sec_panel.isVisibleTo(w))
        w.section_combo.setCurrentText("z")
        self.assertTrue(w.sec_panel.isVisibleTo(w))
        z = w._slice_origin("z")
        self.assertEqual((w.section.axis, w.section.pos), ("z", z))
        self.assertGreater(len(w.sec_plotter.renderer.actors), 0)
        # right-click inside the section probes the interior (not a surface)
        w._on_section_right_click((0.55, 0.05, 0.0))
        np.testing.assert_allclose(w.last_probe, [0.55, 0.05, z])
        self.assertEqual(w.probe_table.item(1, 1).text(), "substrate")
        # vertical cut through it, plotted against depth y
        w.field_combo.setCurrentText("ElectrostaticPotential")
        w.vcut_btn.click()
        p0, p1 = w.cut
        self.assertTrue(w.section.contains(p0) and w.section.contains(p1))
        self.assertEqual(w.cutplot.ax.get_xlabel(), "y (um)")
        x = w.cutplot.ax.lines[0].get_xdata()
        self.assertAlmostEqual(x[0], w.state.grid.bounds[2])
        # hold: stepping the bias overlays curves
        w.overlay.setChecked(True)
        w.bias_slider.setValue(2)
        self.assertEqual(len(w.cutplot.ax.lines), 2)
        y0, y2 = (ln.get_ydata() for ln in w.cutplot.ax.lines)
        np.testing.assert_allclose(y2[np.isfinite(y2)], y0[np.isfinite(y0)] + 1.0, atol=1e-9)   # Vg 0 -> 1 V
        self.assertIs(w.section.state, w.state, "section follows the bias")
        # moving the slider moves the section; changing field drops held curves
        w.slice_pos["z"].setValue(100)
        self.assertAlmostEqual(w.section.pos, w._slice_origin("z"))
        w.field_combo.setCurrentText("eDensity")
        self.assertEqual(len(w.cutplot.ax.lines), 1)
        # Pick in the section sets an in-plane cut
        w.pick_btn.setChecked(True)
        w._on_section_right_click((0.3, 0.02, 0.0))
        w._on_section_right_click((0.8, 0.02, 0.0))
        self.assertFalse(w.pick_btn.isChecked())
        self.assertEqual(w.cutplot.ax.get_xlabel(), "x (um)")
        # a Pick click is also the point H/V cuts run through
        w.vcut_btn.click()
        self.assertAlmostEqual(w.cut[0][0], 0.8)
        self.assertEqual(w.cutplot.ax.get_xlabel(), "y (um)")
        # Clear resets the points, the line and the curves
        w._on_cut_clear()
        self.assertEqual((w.cut_p0.text(), w.cut_p1.text(), w.cut), ("", "", None))
        self.assertEqual(len(w.cutplot.ax.lines), 0)
        w.section_combo.setCurrentText("off")
        self.assertIsNone(w.section)
        self.assertFalse(w.sec_panel.isVisibleTo(w))

    def test_section_views_render(self):
        w = self.window(self.pvd3)
        for ax in "xyz":
            w.section_combo.setCurrentText(ax)
            with tempfile.TemporaryDirectory() as t:
                png = os.path.join(t, "s.png")
                w.sec_plotter.screenshot(png)
                self.assertGreater(os.path.getsize(png), 5000, ax)

    def test_2d_axis_cut_and_single_state_note(self):
        w = self.window(self.pvd2)
        w.hcut_btn.click()                          # no point yet: through the middle
        b = w.state.grid.bounds
        np.testing.assert_allclose(w.cut[0][1], (b[2] + b[3]) / 2)
        w.probe([0.55, 0.05, 0.0])
        w.hcut_btn.click()
        self.assertEqual(w.cutplot.ax.get_xlabel(), "x (um)")
        w2 = self.window(self._first_vtu(self.pvd2))
        self.assertFalse(w2.bias_box.isEnabled())
        self.assertIn("--vtk all", w2.bias_label.text())

    @staticmethod
    def _first_vtu(pvd):
        d = os.path.dirname(pvd)
        return os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".vtu"))[0])

    def test_section_splitter_resizes(self):
        """Opening a section splits the view area evenly, and the divider
        moves both ways (the section title used to pin a ~600 px minimum)."""
        w = self.window(self.pvd3)
        w.resize(1400, 900)
        w.show()
        self.app.processEvents()
        w.section_combo.setCurrentText("z")
        self.app.processEvents()
        a, b = w.splitter.sizes()
        self.assertLessEqual(abs(a - b), 2)
        self.assertLess(w.sec_panel.minimumSizeHint().width(), 200)
        w.splitter.moveSplitter(int(0.75 * (a + b)), 1)
        self.app.processEvents()
        self.assertGreater(w.splitter.sizes()[0], 0.7 * (a + b))


if __name__ == "__main__":
    unittest.main()
