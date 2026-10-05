"""tcadview main window: a 3D view of one state of a sweep, with a region
tree, field/scale/range controls, slice planes, a probe and a cut-line
panel, and a bias stepper. All data handling is in tcadview/model.py.

Mouse: left-drag rotates (3D) or pans (2D), wheel zooms, RIGHT-click
probes a point (or sets cut-line end points in "Pick" mode).
"""
import os

os.environ.setdefault("QT_API", "pyside6")

import numpy as np                                                   # noqa: E402
import pyvista as pv                                                 # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets                         # noqa: E402
from pyvistaqt import QtInteractor                                   # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg      # noqa: E402
from matplotlib.figure import Figure                                 # noqa: E402

from tcadview import model                                           # noqa: E402

CMAPS = ("turbo", "viridis", "coolwarm", "plasma", "jet", "gray")
SLIDER_STEPS = 1000
NAN_COLOR = "lightgray"


def _fmt(v):
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:.4g}" if 1e-3 <= abs(v) < 1e5 or v == 0 else f"{v:.4e}"


def _parse_point(text, dim):
    vals = [float(t) for t in text.replace(",", " ").split()]
    if len(vals) == 2:
        vals.append(0.0)
    if len(vals) != 3:
        raise ValueError("enter x, y (and z) in um")
    return np.array(vals)


class CutPlot(FigureCanvasQTAgg):
    """1D panel: a field along the cut line."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 2.2), layout="constrained")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.clear()

    def clear(self, msg="Cut line: set two points (or Pick them with right-clicks) and press Plot"):
        self.ax.clear()
        self.ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax.transAxes, color="gray")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw_idle()

    def plot(self, d, v, name, unit, scale, title):
        self.ax.clear()
        y = np.abs(v) if scale == "log" else v
        self.ax.plot(d, y, lw=1.5)
        if scale == "log":
            self.ax.set_yscale("log")
        elif scale == "signed log":
            fin = np.abs(v[np.isfinite(v)])
            self.ax.set_yscale("symlog", linthresh=max(1.0, float(fin.min()) if len(fin) else 1.0))
        self.ax.set_xlabel("distance along cut (um)")
        self.ax.set_ylabel(f"{name}" + (f" ({unit})" if unit else ""))
        self.ax.set_title(title, fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path=None, off_screen=False):
        super().__init__()
        self.setWindowTitle("tcadview")
        self.resize(1400, 900)
        self.session = None
        self.idx = 0
        self.region_visible = {}
        self.region_opacity = {}
        self.cut = None                   # (p0, p1) in um
        self._pick_cut = []               # pending picked cut points
        self._picking = False
        self._camera_set = False
        self._probe_actor = None

        if off_screen:
            # headless (tests, --screenshot): a plain off-screen VTK plotter
            # with the same API; the Qt widgets still exist and drive it
            self.plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
            self.setCentralWidget(QtWidgets.QWidget())
        else:
            self.plotter = QtInteractor(self)
            self.setCentralWidget(self.plotter.interactor)
            self.plotter.track_click_position(self._on_right_click, side="right")
        self.plotter.set_background("white")

        self._build_menu()
        self._build_controls()
        self._build_bottom()
        self.statusBar().showMessage("Open a .pvd or .vtu file (File > Open)")
        if path:
            self.open(path)

    # --- layout --------------------------------------------------------------
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Open...", self._open_dialog)
        a.setShortcut(QtGui.QKeySequence.Open)
        a = m.addAction("Save &screenshot...", self._screenshot)
        a.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        m.addSeparator()
        a = m.addAction("&Quit", self.close)
        a.setShortcut(QtGui.QKeySequence.Quit)
        v = self.menuBar().addMenu("&View")
        v.addAction("Reset camera", self._reset_camera)

    def _build_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        # bias stepper
        self.bias_box = QtWidgets.QGroupBox("Bias")
        bl = QtWidgets.QGridLayout(self.bias_box)
        self.bias_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bias_slider.valueChanged.connect(self.set_state)
        self.prev_btn = QtWidgets.QToolButton(text="<")
        self.next_btn = QtWidgets.QToolButton(text=">")
        self.prev_btn.clicked.connect(lambda: self.bias_slider.setValue(self.bias_slider.value() - 1))
        self.next_btn.clicked.connect(lambda: self.bias_slider.setValue(self.bias_slider.value() + 1))
        self.bias_label = QtWidgets.QLabel("-")
        bl.addWidget(self.prev_btn, 0, 0)
        bl.addWidget(self.bias_slider, 0, 1)
        bl.addWidget(self.next_btn, 0, 2)
        bl.addWidget(self.bias_label, 1, 0, 1, 3)
        lay.addWidget(self.bias_box)

        # field
        fbox = QtWidgets.QGroupBox("Field")
        fl = QtWidgets.QFormLayout(fbox)
        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.currentTextChanged.connect(self._on_field)
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(model.SCALES)
        self.scale_combo.currentTextChanged.connect(self._on_scale)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(CMAPS)
        self.cmap_combo.currentTextChanged.connect(lambda _: self.render())
        self.auto_range = QtWidgets.QCheckBox("auto (visible regions)")
        self.auto_range.setChecked(True)
        self.auto_range.toggled.connect(self._on_auto)
        self.rmin = QtWidgets.QLineEdit()
        self.rmax = QtWidgets.QLineEdit()
        for e in (self.rmin, self.rmax):
            e.setValidator(QtGui.QDoubleValidator())
            e.editingFinished.connect(self._on_manual_range)
        rr = QtWidgets.QHBoxLayout()
        rr.addWidget(self.rmin)
        rr.addWidget(QtWidgets.QLabel("to"))
        rr.addWidget(self.rmax)
        self.edges = QtWidgets.QCheckBox("mesh edges")
        self.edges.toggled.connect(lambda _: self.render())
        fl.addRow("Field", self.field_combo)
        fl.addRow("Scale", self.scale_combo)
        fl.addRow("Colormap", self.cmap_combo)
        fl.addRow("Range", self.auto_range)
        fl.addRow("", rr)
        fl.addRow("", self.edges)
        lay.addWidget(fbox)

        # regions
        rbox = QtWidgets.QGroupBox("Regions")
        rl = QtWidgets.QVBoxLayout(rbox)
        self.region_tree = QtWidgets.QTreeWidget()
        self.region_tree.setHeaderLabels(["Region", "Material"])
        self.region_tree.setRootIsDecorated(False)
        self.region_tree.itemChanged.connect(self._on_region_check)
        self.region_tree.itemSelectionChanged.connect(self._on_region_select)
        self.region_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.region_tree.setMinimumHeight(140)
        rl.addWidget(self.region_tree)
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(QtWidgets.QLabel("Opacity (selected)"))
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.valueChanged.connect(self._on_opacity)
        orow.addWidget(self.opacity)
        rl.addLayout(orow)
        lay.addWidget(rbox)

        # slices (3D)
        self.slice_box = QtWidgets.QGroupBox("Slices")
        sl = QtWidgets.QGridLayout(self.slice_box)
        self.slice_on, self.slice_pos, self.slice_lbl = {}, {}, {}
        for r, ax in enumerate("xyz"):
            cb = QtWidgets.QCheckBox(ax)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(0, SLIDER_STEPS)
            s.setValue(SLIDER_STEPS // 2)
            lbl = QtWidgets.QLabel("-")
            lbl.setMinimumWidth(70)
            cb.toggled.connect(lambda _: self.render())
            s.valueChanged.connect(lambda _, a=ax: self._on_slice(a))
            sl.addWidget(cb, r, 0)
            sl.addWidget(s, r, 1)
            sl.addWidget(lbl, r, 2)
            self.slice_on[ax], self.slice_pos[ax], self.slice_lbl[ax] = cb, s, lbl
        lay.addWidget(self.slice_box)

        # cut line
        cbox = QtWidgets.QGroupBox("Cut line (um)")
        cl = QtWidgets.QFormLayout(cbox)
        self.cut_p0 = QtWidgets.QLineEdit()
        self.cut_p1 = QtWidgets.QLineEdit()
        self.cut_p0.setPlaceholderText("x, y[, z]")
        self.cut_p1.setPlaceholderText("x, y[, z]")
        brow = QtWidgets.QHBoxLayout()
        self.pick_btn = QtWidgets.QPushButton("Pick")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip("Right-click two points in the view")
        self.pick_btn.toggled.connect(self._on_pick_toggle)
        plot_btn = QtWidgets.QPushButton("Plot")
        plot_btn.clicked.connect(self._on_cut_plot)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self._on_cut_clear)
        for b in (self.pick_btn, plot_btn, clear_btn):
            brow.addWidget(b)
        cl.addRow("From", self.cut_p0)
        cl.addRow("To", self.cut_p1)
        cl.addRow(brow)
        lay.addWidget(cbox)

        lay.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def _build_bottom(self):
        self.cutplot = CutPlot(self)
        self.cutplot.setMinimumHeight(160)
        d = QtWidgets.QDockWidget("Cut line", self)
        d.setWidget(self.cutplot)
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, d)
        self._cut_dock = d

        self.probe_table = QtWidgets.QTableWidget(0, 2)
        self.probe_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.probe_table.horizontalHeader().setStretchLastSection(True)
        self.probe_table.verticalHeader().setVisible(False)
        self.probe_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        p = QtWidgets.QDockWidget("Probe (right-click)", self)
        p.setWidget(self.probe_table)
        p.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, p)
        self._probe_dock = p
        self._docks_sized = False

    # --- data ----------------------------------------------------------------
    def _open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open result", "", "VTK results (*.pvd *.vtu)")
        if path:
            self.open(path)

    def open(self, path):
        self.session = model.Session(path)
        self.setWindowTitle(f"tcadview - {os.path.basename(path)}")
        n = len(self.session)
        self.bias_box.setVisible(n > 1)
        self.bias_slider.blockSignals(True)
        self.bias_slider.setRange(0, n - 1)
        self.bias_slider.setValue(0)
        self.bias_slider.blockSignals(False)
        self.idx = 0
        st = self.state
        self.region_visible = {k: True for k in range(len(st.region_names))}
        self.region_opacity = {k: 1.0 for k in range(len(st.region_names))}
        self.region_tree.blockSignals(True)
        self.region_tree.clear()
        for k, nm in enumerate(st.region_names):
            it = QtWidgets.QTreeWidgetItem([nm, st.region_material(k)])
            it.setData(0, QtCore.Qt.UserRole, k)
            it.setCheckState(0, QtCore.Qt.Checked)
            self.region_tree.addTopLevelItem(it)
        self.region_tree.resizeColumnToContents(0)
        self.region_tree.blockSignals(False)
        fields = st.fields()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(fields)
        default = "ElectrostaticPotential" if "ElectrostaticPotential" in fields else \
            ("DopingConcentration" if "DopingConcentration" in fields else fields[0])
        self.field_combo.setCurrentText(default)
        self.field_combo.blockSignals(False)
        self.scale_combo.blockSignals(True)
        self.scale_combo.setCurrentText(model.default_scale(default))
        self.scale_combo.blockSignals(False)
        self.slice_box.setVisible(st.dim == 3)
        self._camera_set = False
        self._update_bias_label()
        self._update_slice_labels()
        self.render()

    @property
    def state(self):
        return self.session.state(self.idx)

    @property
    def field(self):
        return self.field_combo.currentText()

    @property
    def scale(self):
        return self.scale_combo.currentText()

    def visible_regions(self):
        return [k for k, v in self.region_visible.items() if v]

    def set_state(self, i):
        if self.session is None:
            return
        self.idx = int(np.clip(i, 0, len(self.session) - 1))
        self._update_bias_label()
        self.render()
        if self.cut is not None:
            self._plot_cut()

    def _update_bias_label(self):
        info = self.session.states[self.idx]
        self.bias_label.setText(f"{info.name}   [{self.idx + 1}/{len(self.session)}]")

    # --- rendering -------------------------------------------------------------
    def clim(self):
        if self.auto_range.isChecked():
            lo, hi = self.state.range(self.field, self.scale, self.visible_regions())
            self.rmin.setText(f"{lo:.6g}")
            self.rmax.setText(f"{hi:.6g}")
            return lo, hi
        try:
            lo, hi = float(self.rmin.text()), float(self.rmax.text())
        except ValueError:
            return self.state.range(self.field, self.scale, self.visible_regions())
        return (lo, hi) if hi > lo else (hi, lo + 1e-12)

    def _slice_origin(self, ax):
        b = self.state.grid.bounds
        i = "xyz".index(ax)
        lo, hi = b[2 * i], b[2 * i + 1]
        return lo + (hi - lo) * self.slice_pos[ax].value() / SLIDER_STEPS

    def _display_geometry(self, k):
        """What to draw for region k: the cells (2D), the outer surface
        (3D), or the enabled slices through it (3D)."""
        st = self.state
        g = st.region(k)
        if st.dim == 2 or g.n_cells == 0:
            return [g]
        axes = [a for a in "xyz" if self.slice_on[a].isChecked()]
        if not axes:
            return [g.extract_surface(algorithm="dataset_surface")]
        out = []
        for a in axes:
            o = [0.0, 0.0, 0.0]
            o["xyz".index(a)] = self._slice_origin(a)
            s = g.slice(normal=a, origin=o)
            if s.n_points:
                out.append(s)
        return out

    def render(self):
        if self.session is None:
            return
        st, name, scale = self.state, self.field, self.scale
        if not name:
            return
        lo, hi = self.clim()
        unit = st.units.get(name, "")
        title = model.scale_label(name, scale, unit)
        cam = self.plotter.camera_position if self._camera_set else None
        for t in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(t, render=False)
        self.plotter.clear_actors()
        bar = False
        sliced = st.dim == 3 and any(self.slice_on[a].isChecked() for a in "xyz")
        for k in self.visible_regions():
            for geom in self._display_geometry(k):
                m = geom.copy(deep=False)
                assoc = st.association(name)
                if assoc == "point":
                    m.point_data["_display"] = model.scale_values(st.values(name, geom), scale)
                else:
                    m.cell_data["_display"] = model.scale_values(st.values(name, geom), scale)
                self.plotter.add_mesh(
                    m, scalars="_display", preference=assoc, cmap=self.cmap_combo.currentText(), clim=(lo, hi),
                    nan_color=NAN_COLOR, opacity=self.region_opacity.get(k, 1.0), show_edges=self.edges.isChecked(),
                    edge_color="black", line_width=0.5, show_scalar_bar=not bar,
                    scalar_bar_args=dict(title=title, vertical=False, position_x=0.2, position_y=0.03,
                                         width=0.6, height=0.09, fmt="%.3g", color="black",
                                         title_font_size=14, label_font_size=12, n_labels=6),
                    reset_camera=False)
                bar = True
            if sliced:
                self.plotter.add_mesh(st.region(k).extract_feature_edges(), color="gray", opacity=0.4,
                                      reset_camera=False)
        if self.cut is not None:
            self.plotter.add_mesh(pv.Line(*self.cut), color="magenta", line_width=3, reset_camera=False)
        self.plotter.add_text(st.label or self.session.states[self.idx].name, position="upper_left",
                              font_size=10, color="black", name="label")
        if cam is None:
            self._reset_camera()
        else:
            self.plotter.camera_position = cam
        self.plotter.render()

    def _reset_camera(self):
        """Device the right way up: y is depth (down), so look from -z with
        view-up -y; 3D gets an oblique view from above."""
        if self.session is None:
            return
        if self.state.dim == 2:
            self.plotter.view_vector((0, 0, -1), viewup=(0, -1, 0))
        else:
            self.plotter.view_vector((1.0, -0.8, -1.4), viewup=(0, -1, 0))
        self.plotter.reset_camera()
        self._camera_set = True

    # --- control callbacks ---------------------------------------------------------
    def _on_field(self, name):
        if not name:
            return
        self.scale_combo.blockSignals(True)
        self.scale_combo.setCurrentText(model.default_scale(name))
        self.scale_combo.blockSignals(False)
        self.auto_range.setChecked(True)
        self.render()
        if self.cut is not None:
            self._plot_cut()

    def _on_scale(self, _):
        self.auto_range.setChecked(True)
        self.render()
        if self.cut is not None:
            self._plot_cut()

    def _on_auto(self, on):
        self.rmin.setEnabled(not on)
        self.rmax.setEnabled(not on)
        self.render()

    def _on_manual_range(self):
        if not self.auto_range.isChecked():
            self.render()

    def _on_region_check(self, item, col):
        k = item.data(0, QtCore.Qt.UserRole)
        self.region_visible[k] = item.checkState(0) == QtCore.Qt.Checked
        self.render()

    def _selected_regions(self):
        return [it.data(0, QtCore.Qt.UserRole) for it in self.region_tree.selectedItems()]

    def _on_region_select(self):
        sel = self._selected_regions()
        if sel:
            self.opacity.blockSignals(True)
            self.opacity.setValue(int(round(100 * self.region_opacity[sel[0]])))
            self.opacity.blockSignals(False)

    def _on_opacity(self, v):
        for k in self._selected_regions():
            self.region_opacity[k] = v / 100.0
        self.render()

    def _update_slice_labels(self):
        for a in "xyz":
            self.slice_lbl[a].setText(f"{self._slice_origin(a):.4g} um")

    def _on_slice(self, ax):
        self.slice_lbl[ax].setText(f"{self._slice_origin(ax):.4g} um")
        if self.slice_on[ax].isChecked():
            self.render()

    # --- probe and cut line ---------------------------------------------------------
    def _on_right_click(self, pos):
        if self.session is None:
            return
        p = np.array(pos, dtype=float)
        if self.state.dim == 2:
            p[2] = 0.0
        if self._picking:
            self._pick_cut.append(p)
            (self.cut_p0 if len(self._pick_cut) == 1 else self.cut_p1).setText(
                ", ".join(f"{v:.5g}" for v in (p[:2] if self.state.dim == 2 else p)))
            if len(self._pick_cut) == 2:
                self.pick_btn.setChecked(False)
                self._on_cut_plot()
            return
        self.probe(p)

    def probe(self, p):
        p = np.asarray(p, dtype=float)
        res = self.state.probe(p)
        coord = ", ".join(f"{v:.5g}" for v in (p[:2] if self.state.dim == 2 else p))
        if res is None:
            self.statusBar().showMessage(f"({coord}) um: outside the mesh")
            return None
        rows = [("position (um)", coord), ("region", res["region"] or "-")]
        rows += [(n + (f" ({self.state.units[n]})" if n in self.state.units else ""), _fmt(v))
                 for n, v in res["fields"].items()]
        self.probe_table.setRowCount(len(rows))
        for r, (a, b) in enumerate(rows):
            self.probe_table.setItem(r, 0, QtWidgets.QTableWidgetItem(a))
            self.probe_table.setItem(r, 1, QtWidgets.QTableWidgetItem(b))
        self.probe_table.resizeColumnToContents(0)
        v = res["fields"].get(self.field)
        self.statusBar().showMessage(f"({coord}) um  {res['region']}:  {self.field} = {_fmt(v)}")
        self.plotter.add_mesh(pv.PolyData(p[None, :]), color="black", point_size=10, render_points_as_spheres=True,
                              name="probe", reset_camera=False)
        self.plotter.render()
        return res

    def _on_pick_toggle(self, on):
        self._picking = on
        self._pick_cut = []
        if on:
            self.statusBar().showMessage("Cut line: right-click the start point, then the end point")

    def _on_cut_plot(self):
        if self.session is None:
            return
        try:
            p0 = _parse_point(self.cut_p0.text(), self.state.dim)
            p1 = _parse_point(self.cut_p1.text(), self.state.dim)
        except ValueError as e:
            self.statusBar().showMessage(f"Cut line: {e}")
            return
        if np.allclose(p0, p1):
            self.statusBar().showMessage("Cut line: the two points coincide")
            return
        self.cut = (p0, p1)
        self.render()
        self._plot_cut()

    def _plot_cut(self):
        st = self.state
        d, v = st.cut_line(*self.cut, self.field)
        self.cutplot.plot(d, v, self.field, st.units.get(self.field, ""), self.scale,
                          st.label or self.session.states[self.idx].name)

    def _on_cut_clear(self):
        self.cut = None
        self.cutplot.clear()
        self.render()

    def _screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", "tcadview.png", "PNG (*.png)")
        if path:
            self.plotter.screenshot(path)

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._docks_sized:             # dock sizes only apply once shown
            self._docks_sized = True
            self.resizeDocks([self._cut_dock], [230], QtCore.Qt.Vertical)
            self.resizeDocks([self._probe_dock], [380], QtCore.Qt.Horizontal)

    def closeEvent(self, ev):
        self.plotter.close()
        super().closeEvent(ev)
