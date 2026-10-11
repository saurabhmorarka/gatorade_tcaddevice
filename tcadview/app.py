"""tcadview main window: a 3D view of one state of a sweep, with a region
tree, field/scale/range controls, slice planes, a probe and a cut-line
panel, and a bias stepper. For 3D results a "Section view" shows one slice
face-on in a second, 2D view on the right; probes and cut lines made there
lie inside the device. All data handling is in tcadview/model.py.

Mouse: left-drag rotates (3D) or pans (2D and section views), wheel zooms,
RIGHT-click probes a point (or sets cut-line end points in "Pick" mode).
H cut / V cut run a cut line through the last probed point.
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
    """1D panel: a field along the cut line. With hold on, earlier curves
    stay (to overlay bias points or cut positions) until the field or scale
    changes or Clear is pressed."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 2.2), layout="constrained")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.held = []                    # [(x, v, label)]
        self.current = None
        self.clear()

    def clear(self, msg="Cut line: set two points, Pick them with right-clicks, or use an H/V cut "
                        "through the last probe"):
        self.held, self.current = [], None
        self.ax.clear()
        self.ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax.transAxes, color="gray")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw_idle()

    def plot(self, x, v, name, unit, scale, label, xlabel="distance along cut (um)", hold=False):
        if hold and self.current is not None and self.current[2] != label:
            self.held = [c for c in self.held if c[2] != self.current[2]] + [self.current]
        elif not hold:
            self.held = []
        self.current = (x, v, label)
        self.ax.clear()
        curves = self.held + [self.current]
        for cx, cv, cl in curves:
            self.ax.plot(cx, np.abs(cv) if scale == "log" else cv, lw=1.5, label=cl)
        if scale == "log":
            self.ax.set_yscale("log")
        elif scale == "signed log":
            fin = np.abs(np.concatenate([c[1] for c in curves]))
            fin = fin[np.isfinite(fin) & (fin > 0)]
            # linear band 8 decades below the peak: readable decades for doping-sized values
            self.ax.set_yscale("symlog", linthresh=max(1.0, 1e-8 * float(fin.max())) if len(fin) else 1.0)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(f"{name}" + (f" ({unit})" if unit else ""))
        if len(curves) > 1:
            self.ax.legend(fontsize=7, loc="best")
        else:
            self.ax.set_title(label, fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path=None, off_screen=False):
        super().__init__()
        self.setWindowTitle("tcadview")
        self.resize(1400, 900)
        self.session = None
        self.idx = 0
        self.region_visible = {}          # by region name: process steps add regions
        self.region_opacity = {}
        self.cut = None                   # (p0, p1) in um
        self._pick_cut = []               # pending picked cut points
        self._picking = False
        self._camera_set = False
        self._sec_camera_set = False
        self._probe_actor = None
        self.last_probe = None            # 3D point of the last probe (um)
        self.section = None               # model.Section shown in the right view

        # 3D view on the left, 2D section view on the right (hidden until a
        # section is chosen), in a splitter the user can drag
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.sec_panel = QtWidgets.QWidget()
        sp = QtWidgets.QVBoxLayout(self.sec_panel)
        sp.setContentsMargins(0, 0, 0, 0)
        self.sec_title = QtWidgets.QLabel("-")
        self.sec_title.setStyleSheet("padding: 3px; font-weight: bold")
        # wrap, and never let the title set the panel's minimum width (it
        # used to pin the section panel at ~600 px and squeeze the 3D view)
        self.sec_title.setWordWrap(True)
        self.sec_title.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sp.addWidget(self.sec_title)
        if off_screen:
            # headless (tests, --screenshot): plain off-screen VTK plotters
            # with the same API; the Qt widgets still exist and drive them
            self.plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
            self.sec_plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
            self.splitter.addWidget(QtWidgets.QWidget())
        else:
            self.plotter = QtInteractor(self)
            self.sec_plotter = QtInteractor(self.sec_panel)
            self.splitter.addWidget(self.plotter.interactor)
            sp.addWidget(self.sec_plotter.interactor, 1)
            self.plotter.track_click_position(self._on_right_click, side="right")
            self.sec_plotter.track_click_position(self._on_section_right_click, side="right")
            self.sec_plotter.enable_2d_style()         # 2D: left-drag pans, no rotation
        self.splitter.addWidget(self.sec_panel)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.widget(0).setMinimumWidth(150)
        self.sec_panel.setMinimumWidth(150)
        self.sec_panel.setVisible(False)
        self.setCentralWidget(self.splitter)
        self.plotter.set_background("white")
        self.sec_plotter.set_background("white")
        self.sec_plotter.enable_parallel_projection()

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
        v.addAction("Split 3D / section evenly", self.split_evenly)
        self._view_menu = v                 # panel show/hide toggles are added once the docks exist

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
        self.bias_label.setWordWrap(True)
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
        # process builds: the voxel cells the device is built from (exact,
        # sharp, shared faces identical) or the smooth level-set contours
        # (prettier, but every sharp edge is bevelled by up to a voxel and
        # neighbouring materials do not share one surface)
        self.smooth = QtWidgets.QCheckBox("smooth surfaces (level set)")
        self.smooth.setToolTip("Off: the voxel structure the device solve uses. On: the level-set contours.")
        self.smooth.toggled.connect(lambda _: self.render())
        fl.addRow("Field", self.field_combo)
        fl.addRow("Scale", self.scale_combo)
        fl.addRow("Colormap", self.cmap_combo)
        fl.addRow("Range", self.auto_range)
        fl.addRow("", rr)
        fl.addRow("", self.edges)
        fl.addRow("", self.smooth)
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
        self.region_tree.setMinimumHeight(90)
        self.region_tree.setMaximumHeight(130)
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
        self.section_combo = QtWidgets.QComboBox()
        self.section_combo.addItems(["off", "x", "y", "z"])
        self.section_combo.setToolTip("Show the x/y/z slice face-on in a 2D view on the right, where "
                                      "right-click probes inside the device and cut lines lie in the plane")
        self.section_combo.currentTextChanged.connect(self._on_section)
        sl.addWidget(QtWidgets.QLabel("Section view"), 3, 0, 1, 2)
        sl.addWidget(self.section_combo, 3, 2)
        lay.addWidget(self.slice_box)

        lay.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def _build_cut_controls(self):
        """Cut-line controls, left of the 1D plot they drive (always in
        view, unlike the bottom of the scrolling Controls panel)."""
        box = QtWidgets.QWidget()
        cl = QtWidgets.QFormLayout()
        self.cut_p0 = QtWidgets.QLineEdit()
        self.cut_p1 = QtWidgets.QLineEdit()
        self.cut_p0.setPlaceholderText("x, y[, z] um")
        self.cut_p1.setPlaceholderText("x, y[, z] um")
        self.cut_p0.returnPressed.connect(self._on_cut_plot)
        self.cut_p1.returnPressed.connect(self._on_cut_plot)
        self.pick_btn = QtWidgets.QPushButton("Pick 2 points")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip("Then right-click the start and end points (in the section view for "
                                 "points inside a 3D device)")
        self.pick_btn.toggled.connect(self._on_pick_toggle)
        plot_btn = QtWidgets.QPushButton("Plot")
        plot_btn.setToolTip("Plot the cut between the typed From and To points")
        plot_btn.clicked.connect(self._on_cut_plot)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setToolTip("Remove the cut line, its points and all curves")
        clear_btn.clicked.connect(self._on_cut_clear)
        self.hcut_btn = QtWidgets.QPushButton("H cut")
        self.vcut_btn = QtWidgets.QPushButton("V cut")
        tip = ("{} cut across the whole section (or 2D device) through the last point you right-clicked "
               "(probe or Pick); through the middle if there is none")
        self.hcut_btn.setToolTip(tip.format("Horizontal"))
        self.vcut_btn.setToolTip(tip.format("Vertical"))
        self.hcut_btn.clicked.connect(lambda: self.axis_cut("h"))
        self.vcut_btn.clicked.connect(lambda: self.axis_cut("v"))
        self.overlay = QtWidgets.QCheckBox("Overlay curves")
        self.overlay.setToolTip("On: each new curve is added to the plot instead of replacing it - step the "
                                "bias or move the cut to compare. Changing field or Clear starts over.")
        self.overlay.toggled.connect(self._on_overlay)
        grid = QtWidgets.QGridLayout()
        for i, b in enumerate((self.hcut_btn, self.vcut_btn, self.pick_btn, plot_btn, clear_btn)):
            grid.addWidget(b, i // 2, i % 2)
        cl.addRow("From", self.cut_p0)
        cl.addRow("To", self.cut_p1)
        cl.addRow(grid)
        cl.addRow(self.overlay)
        left = QtWidgets.QWidget()
        left.setLayout(cl)
        left.setFixedWidth(300)
        h = QtWidgets.QHBoxLayout(box)
        h.setContentsMargins(4, 4, 4, 4)
        h.addWidget(left)
        h.addWidget(self.cutplot, 1)
        return box

    def _build_bottom(self):
        self.cutplot = CutPlot(self)
        self.cutplot.setMinimumHeight(160)
        d = QtWidgets.QDockWidget("Cut line", self)
        d.setWidget(self._build_cut_controls())
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
        self._view_menu.addSeparator()
        for dk in self.findChildren(QtWidgets.QDockWidget):
            dk.setFeatures(dk.features() | QtWidgets.QDockWidget.DockWidgetClosable)
            self._view_menu.addAction(dk.toggleViewAction())

    # --- data ----------------------------------------------------------------
    def _open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open result", "", "VTK results (*.pvd *.vtu)")
        if path:
            self.open(path)

    def open(self, path):
        self.session = model.Session(path)
        self.setWindowTitle(f"tcadview - {os.path.basename(path)}")
        n = len(self.session)
        self.bias_box.setEnabled(n > 1)
        self.bias_slider.blockSignals(True)
        self.bias_slider.setRange(0, n - 1)
        self.bias_slider.setValue(0)
        self.bias_slider.blockSignals(False)
        st0 = self.session.state(0)
        self.idx = n - 1 if st0.is_process else 0          # a build opens on its final structure
        if self.idx:
            self.bias_slider.blockSignals(True)
            self.bias_slider.setValue(self.idx)
            self.bias_slider.blockSignals(False)
        self.bias_box.setTitle("Process step" if st0.is_process else "Bias")
        st = self.state
        self.region_visible, self.region_opacity = {}, {}
        self._sync_regions()
        fields = st.fields()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(fields)
        default = model.MATERIAL_FIELD if st.is_process else \
            "ElectrostaticPotential" if "ElectrostaticPotential" in fields else \
            ("DopingConcentration" if "DopingConcentration" in fields else fields[0])
        self.field_combo.setCurrentText(default)
        self.field_combo.blockSignals(False)
        self.scale_combo.blockSignals(True)
        self.scale_combo.setCurrentText(model.default_scale(default))
        self.scale_combo.blockSignals(False)
        self.slice_box.setVisible(st.dim == 3)
        self.section_combo.blockSignals(True)
        self.section_combo.setCurrentText("off")
        self.section_combo.blockSignals(False)
        self.section = None
        self.sec_panel.setVisible(False)
        self.last_probe = None
        self.cut = None
        self.cutplot.clear()
        if hasattr(self.plotter, "iren") and self.plotter.iren is not None:
            if st.dim == 2:
                self.plotter.enable_2d_style()
            else:
                self.plotter.enable_trackball_style()
        self._camera_set = False
        self._update_bias_label()
        self._update_slice_labels()
        self.render()
        self.statusBar().showMessage("Right-click probes a point" + (
            "; for points inside the device use Slices > Section view" if st.dim == 3 else ""))

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
        return [k for k, n in enumerate(self.state.region_names) if self.region_visible.get(n, True)]

    def _opacity(self, k):
        return self.region_opacity.get(self.state.region_names[k], 1.0)

    def _sync_regions(self):
        """Region tree = the current state's regions (a process step can add
        or remove some); visibility and opacity are kept by name."""
        st = self.state
        rows = [(nm, st.region_material(k)) for k, nm in enumerate(st.region_names)]
        if getattr(self, "_tree_rows", None) == rows:
            return
        self._tree_rows = rows
        self.region_tree.blockSignals(True)
        self.region_tree.clear()
        for k, (nm, mat) in enumerate(rows):
            it = QtWidgets.QTreeWidgetItem([nm, mat])
            it.setData(0, QtCore.Qt.UserRole, k)
            it.setCheckState(0, QtCore.Qt.Checked if self.region_visible.get(nm, True) else QtCore.Qt.Unchecked)
            if st.is_process:
                it.setIcon(1, self._swatch(model.material_color(mat)))
            self.region_tree.addTopLevelItem(it)
        self.region_tree.resizeColumnToContents(0)
        self.region_tree.blockSignals(False)

    @staticmethod
    def _swatch(color):
        pm = QtGui.QPixmap(12, 12)
        pm.fill(QtGui.QColor(color))
        return QtGui.QIcon(pm)

    def _sync_fields(self):
        """Field list = the current state's fields (doping appears at the
        first implant of a build), keeping the selection when it exists."""
        fields = self.state.fields()
        if [self.field_combo.itemText(i) for i in range(self.field_combo.count())] == fields:
            return
        cur = self.field
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(fields)
        self.field_combo.setCurrentText(cur if cur in fields else fields[0])
        self.field_combo.blockSignals(False)

    def set_state(self, i):
        if self.session is None:
            return
        self.idx = int(np.clip(i, 0, len(self.session) - 1))
        self._sync_regions()
        self._sync_fields()
        self._update_bias_label()
        self._rebuild_section()
        self.render()
        if self.cut is not None:
            self._plot_cut()

    def _update_bias_label(self):
        info = self.session.states[self.idx]
        if self.state.is_process:
            self.bias_label.setText(f"{info.name}   [step {self.idx + 1}/{len(self.session)}]")
        elif len(self.session) > 1:
            self.bias_label.setText(f"{info.name}   [{self.idx + 1}/{len(self.session)}]")
        else:
            self.bias_label.setText(f"{info.name}  (only one bias point saved; for the whole sweep "
                                    f"rerun the driver with --vtk all)")

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
        if st.is_process:
            axes = [a for a in "xyz" if self.slice_on[a].isChecked()]
            smooth = self.smooth.isChecked()
            if not axes:
                return [g for g in [self._process_surface(k)] if g is not None]
            cut = st.slice_region if smooth else st.voxel_slice
            return [g for g in (cut(k, a, self._slice_origin(a)) for a in axes) if g is not None]
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

    def _process_surface(self, k):
        st = self.state
        return st.surface(k) if self.smooth.isChecked() else st.voxel_surface(k)

    def _add_field(self, plotter, geom, k, lo, hi, title, bar):
        """Draw region k's geometry `geom` coloured by the current field (or
        by its material)."""
        st, name = self.state, self.field
        if name == model.MATERIAL_FIELD:
            plotter.add_mesh(geom, color=model.material_color(st.region_material(k)), opacity=self._opacity(k),
                             show_edges=self.edges.isChecked(), edge_color="black", line_width=0.5,
                             smooth_shading=geom.n_cells > 0 and st.is_process and self.smooth.isChecked(),
                             reset_camera=False)
            return
        m = geom.copy(deep=False)
        assoc = st.association(name)
        vals = model.scale_values(st.values(name, geom), self.scale)
        if assoc == "point":
            m.point_data["_display"] = vals
        else:
            m.cell_data["_display"] = vals
        plotter.add_mesh(
            m, scalars="_display", preference=assoc, cmap=self.cmap_combo.currentText(), clim=(lo, hi),
            nan_color=NAN_COLOR, opacity=self._opacity(k), show_edges=self.edges.isChecked(),
            edge_color="black", line_width=0.5, show_scalar_bar=not bar,
            scalar_bar_args=dict(title=title, vertical=False, position_x=0.2, position_y=0.03,
                                 width=0.6, height=0.09, fmt="%.3g", color="black",
                                 title_font_size=14, label_font_size=12, n_labels=6),
            reset_camera=False)

    @staticmethod
    def _clear(plotter):
        for t in list(plotter.scalar_bars.keys()):
            plotter.remove_scalar_bar(t, render=False)
        plotter.clear_actors()

    def _add_marker(self, plotter, p):
        plotter.add_mesh(pv.PolyData(np.asarray(p, dtype=float)[None, :]), color="black", point_size=10,
                         render_points_as_spheres=True, name="probe", reset_camera=False)

    def render(self):
        if self.session is None:
            return
        st, name, scale = self.state, self.field, self.scale
        if not name:
            return
        lo, hi = self.clim()
        title = model.scale_label(name, scale, st.units.get(name, ""))
        cam = self.plotter.camera_position if self._camera_set else None
        self._clear(self.plotter)
        bar = False
        sliced = st.dim == 3 and any(self.slice_on[a].isChecked() for a in "xyz")
        for k in self.visible_regions():
            for geom in self._display_geometry(k):
                self._add_field(self.plotter, geom, k, lo, hi, title, bar)
                bar = bar or name != model.MATERIAL_FIELD
            if sliced and st.is_process:
                surf = self._process_surface(k)
                if surf is not None:                       # faint context around the slices
                    self.plotter.add_mesh(surf, color=model.material_color(st.region_material(k)), opacity=0.08,
                                          reset_camera=False)
            elif sliced:
                self.plotter.add_mesh(st.region(k).extract_feature_edges(), color="gray", opacity=0.4,
                                      reset_camera=False)
        if self.section is not None:
            self.plotter.add_mesh(pv.lines_from_points(self.section.outline3d()), color="black",
                                  line_width=2, reset_camera=False)
        if self.cut is not None:
            self.plotter.add_mesh(pv.Line(*self.cut), color="magenta", line_width=3, reset_camera=False)
        if self.last_probe is not None:
            self._add_marker(self.plotter, self.last_probe)
        self.plotter.add_text(st.label or self.session.states[self.idx].name, position="upper_left",
                              font_size=10, color="black", name="label")
        if cam is None:
            self._reset_camera()
        else:
            self.plotter.camera_position = cam
        self.plotter.render()
        self.render_section(lo, hi, title)

    def render_section(self, lo=None, hi=None, title=None):
        """The 2D section view: the plane face-on, coloured on the same
        range as the 3D view, with the probe and in-plane cut lines."""
        sec = self.section
        if sec is None:
            return
        if lo is None:
            lo, hi = self.clim()
            title = model.scale_label(self.field, self.scale, self.state.units.get(self.field, ""))
        pl = self.sec_plotter
        cam = pl.camera_position if self._sec_camera_set else None
        self._clear(pl)
        bar = False
        for k in self.visible_regions():
            g = sec.region(k)
            if g is not None:
                self._add_field(pl, g, k, lo, hi, title, bar)
                bar = bar or self.field != model.MATERIAL_FIELD
                pl.add_mesh(g.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                    manifold_edges=False, non_manifold_edges=False),
                            color="gray", line_width=1, reset_camera=False)
        flat = lambda p: np.append(sec.to2d(p), 0.0)                 # noqa: E731
        if self.cut is not None and sec.contains(self.cut[0]) and sec.contains(self.cut[1]):
            pl.add_mesh(pv.Line(flat(self.cut[0]), flat(self.cut[1])), color="magenta", line_width=3,
                        reset_camera=False)
        if self.last_probe is not None and sec.contains(self.last_probe):
            self._add_marker(pl, flat(self.last_probe))
        u, v = sec.axis_labels
        self.sec_title.setText(f"Section {sec.axis} = {sec.pos:.4g} um  ({u} across, {v} down)")
        self.sec_title.setToolTip("Right-click probes a point inside the device; Pick 2 points and H/V cut "
                                  "make cut lines in this plane")
        if cam is None:
            self._reset_section_camera()
        else:
            pl.camera_position = cam
        pl.render()

    def _reset_section_camera(self):
        """Face-on. x- and z-sections have depth (y) as vertical, pointing
        down as in 2D; the y-section is a plan view (x right, z up)."""
        if self.section is None:
            return
        up = np.array((0, 1, 0) if self.section.axis == "y" else (0, -1, 0), dtype=float)
        self.sec_plotter.view_vector((0, 0, 1) if self.section.axis == "y" else (0, 0, -1), viewup=up)
        self.sec_plotter.reset_camera()
        # leave the bottom ~15% for the colour bar: zoom out a little and
        # move the camera down (the device moves up)
        cam = self.sec_plotter.camera
        cam.parallel_scale = cam.parallel_scale / 0.85
        shift = -0.15 * cam.parallel_scale * up
        cam.focal_point = tuple(np.add(cam.focal_point, shift))
        cam.position = tuple(np.add(cam.position, shift))
        self._sec_camera_set = True

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
        self._reset_section_camera()

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
            self.cutplot.held = []        # held curves are of another field
            self._plot_cut()

    def _on_scale(self, _):
        self.auto_range.setChecked(True)
        self.render()
        if self.cut is not None:
            self.cutplot.held = []
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
        self.region_visible[self.state.region_names[k]] = item.checkState(0) == QtCore.Qt.Checked
        self.render()

    def _selected_regions(self):
        return [it.data(0, QtCore.Qt.UserRole) for it in self.region_tree.selectedItems()]

    def _on_region_select(self):
        sel = self._selected_regions()
        if sel:
            self.opacity.blockSignals(True)
            self.opacity.setValue(int(round(100 * self._opacity(sel[0]))))
            self.opacity.blockSignals(False)

    def _on_opacity(self, v):
        for k in self._selected_regions():
            self.region_opacity[self.state.region_names[k]] = v / 100.0
        self.render()

    def _update_slice_labels(self):
        for a in "xyz":
            self.slice_lbl[a].setText(f"{self._slice_origin(a):.4g} um")

    def _on_slice(self, ax):
        self.slice_lbl[ax].setText(f"{self._slice_origin(ax):.4g} um")
        on_section = self.section is not None and self.section.axis == ax
        if on_section:
            self._rebuild_section()
        if self.slice_on[ax].isChecked() or on_section:
            self.render()

    def _rebuild_section(self):
        """(Re)cut the section for the current state and slider position."""
        ax = self.section_combo.currentText()
        if self.session is None or ax == "off" or self.state.dim != 3:
            self.section = None
        else:
            self.section = model.Section(self.state, ax, self._slice_origin(ax))

    def _on_section(self, ax):
        self._sec_camera_set = False
        self._rebuild_section()
        was_open = self.sec_panel.isVisible()
        self.sec_panel.setVisible(self.section is not None)
        if self.section is not None and not was_open:
            self.split_evenly()
        self.render()

    def split_evenly(self):
        w = max(self.splitter.width(), 2)
        self.splitter.setSizes([w // 2, w - w // 2])

    # --- probe and cut line ---------------------------------------------------------
    def _on_right_click(self, pos):
        if self.session is None:
            return
        p = np.array(pos, dtype=float)
        if self.state.dim == 2:
            p[2] = 0.0
        self._use_point(p)

    def _on_section_right_click(self, pos):
        """A right-click in the section view: the in-plane point, mapped
        back to 3D (exactly on the plane)."""
        if self.section is None:
            return
        self._use_point(self.section.to3d(np.asarray(pos, dtype=float)[:2]))

    def _use_point(self, p):
        """A picked 3D point: a cut-line end in Pick mode, else a probe."""
        if self._picking:
            self.last_probe = p
            self._pick_cut.append(p)
            (self.cut_p0 if len(self._pick_cut) == 1 else self.cut_p1).setText(self._coord_text(p))
            if len(self._pick_cut) == 2:
                self.pick_btn.setChecked(False)
                self._on_cut_plot()
            return
        self.probe(p)

    def _coord_text(self, p):
        return ", ".join(f"{v:.5g}" for v in (p[:2] if self.state.dim == 2 else p))

    def probe(self, p):
        p = np.asarray(p, dtype=float)
        res = self.state.probe(p)
        coord = self._coord_text(p)
        if res is None:
            self.statusBar().showMessage(f"({coord}) um: outside the mesh")
            return None
        self.last_probe = p
        rows = [("position (um)", coord), ("region", res["region"] or "-")]
        rows += [(n + (f" ({self.state.units[n]})" if n in self.state.units else ""), _fmt(v))
                 for n, v in res["fields"].items()]
        self.probe_table.setRowCount(len(rows))
        for r, (a, b) in enumerate(rows):
            self.probe_table.setItem(r, 0, QtWidgets.QTableWidgetItem(a))
            self.probe_table.setItem(r, 1, QtWidgets.QTableWidgetItem(b))
        self.probe_table.resizeColumnToContents(0)
        v = res["fields"].get(self.field)
        self.statusBar().showMessage(f"({coord}) um  {res['region']}:  {self.field} = {_fmt(v)}"
                                     f"      (H cut / V cut run through this point)")
        self._add_marker(self.plotter, p)
        self.plotter.render()
        sec = self.section
        if sec is not None:
            if sec.contains(p):
                self._add_marker(self.sec_plotter, np.append(sec.to2d(p), 0.0))
            else:
                self.sec_plotter.remove_actor("probe")
            self.sec_plotter.render()
        return res

    def axis_cut(self, direction):
        """Horizontal ("h") or vertical ("v") cut through the last probed
        point, across the section if one is shown (its in-plane axes),
        else across a 2D device (x / y)."""
        if self.session is None:
            return
        p, sec = self.last_probe, self.section
        if self.state.dim == 3 and sec is None:
            QtWidgets.QMessageBox.information(
                self, "H/V cut", "In 3D, open a Section view first (Slices > Section view), then "
                "right-click a point in it. The cut runs across that section.")
            return
        if p is None or (sec is not None and not sec.contains(p)):
            b = self.state.grid.bounds
            p = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
            if sec is not None:
                p[sec.n] = sec.pos
            self.statusBar().showMessage("H/V cut through the middle (right-click a point to choose where)")
        if sec is not None:
            p0, p1 = sec.axis_cut(sec.to2d(p), direction)
        else:
            p0, p1 = model.device_axis_cut(self.state, p, direction)
        self.cut_p0.setText(self._coord_text(p0))
        self.cut_p1.setText(self._coord_text(p1))
        self.cut = (p0, p1)
        self.render()
        self._plot_cut()

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
        """Plot against the moving coordinate for an axis-aligned cut (e.g.
        depth y for a vertical cut), else against distance along the cut."""
        st = self.state
        p0, p1 = self.cut
        if self.field == model.MATERIAL_FIELD:
            self.cutplot.clear("Cut line: choose a field to plot (e.g. DopingConcentration)")
            return
        d, v = st.cut_line(p0, p1, self.field)
        i = model.cut_axis(p0, p1)
        nd = 2 if st.dim == 2 else 3
        if i is not None:
            x = p0[i] + d * np.sign(p1[i] - p0[i])
            xlabel = f"{'xyz'[i]} (um)"
            fixed = ", ".join(f"{'xyz'[j]}={p0[j]:.4g}" for j in range(nd) if j != i)
        else:
            x, xlabel = d, "distance along cut (um)"
            fixed = f"({self._coord_text(p0)}) -> ({self._coord_text(p1)})"
        bias = st.label or self.session.states[self.idx].name
        self.cutplot.plot(x, v, self.field, st.units.get(self.field, ""), self.scale, f"{bias}  [{fixed}]",
                          xlabel=xlabel, hold=self.overlay.isChecked())

    def _on_cut_clear(self):
        self.cut = None
        self.cut_p0.clear()
        self.cut_p1.clear()
        self.pick_btn.setChecked(False)
        self._pick_cut = []
        self.cutplot.clear()
        self.render()

    def _on_overlay(self, on):
        self.statusBar().showMessage(
            "Overlay on: new curves are added - step the bias or make another cut to compare" if on
            else "Overlay off: each new curve replaces the plot")
        if not on and self.cut is not None:
            self._plot_cut()

    def _screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", "tcadview.png", "PNG (*.png)")
        if path:
            self.plotter.screenshot(path)

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._docks_sized:             # dock sizes only apply once shown
            self._docks_sized = True
            self.resizeDocks([self._cut_dock], [230], QtCore.Qt.Vertical)
            self.resizeDocks([self._probe_dock], [300], QtCore.Qt.Horizontal)

    def closeEvent(self, ev):
        self.plotter.close()
        self.sec_plotter.close()
        super().closeEvent(ev)
