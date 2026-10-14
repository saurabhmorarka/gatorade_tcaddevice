"""Regenerate the README figures in docs/img/.

    python3 docs/make_readme_figures.py            # everything
    python3 docs/make_readme_figures.py 3d         # only the virtual-fab renders
    python3 docs/make_readme_figures.py 2d         # only the 2D NMOS field maps
    python3 docs/make_readme_figures.py 2d-old     # 2D diode / MOS-cap maps (not in the README)

3d: builds configs/fab_planar.yaml (out/fab_planar/) if needed and renders
the exact voxel faces the solver meshes (tcadview's process view).
2d: reads the VTK files the 2D drivers write with --vtk (run them first, see
the README).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
IMG = os.path.join(ROOT, "docs", "img")

HIDE_3D = {"ild"}                 # the ILD encloses everything; leave it out to see the device


# --- 3D: virtual-fab structure ------------------------------------------------
def _fab_session():
    from tcadview import model
    pvd = os.path.join(ROOT, "out", "fab_planar", "build.pvd")
    if not os.path.exists(pvd):
        from fab.__main__ import main as fab_main
        fab_main(["build", os.path.join(ROOT, "configs", "fab_planar.yaml"), "--out", os.path.dirname(pvd)])
    return model.Session(pvd)


def _plotter(shape=(1, 1), size=(1400, 1100)):
    import pyvista as pv
    p = pv.Plotter(off_screen=True, shape=shape, window_size=size, border=False)
    p.set_background("white")
    return p


def _camera(p, grid, zoom=1.0):
    # y is depth (positive down): look from the drain side, above, front
    c = np.array(grid.center)
    p.camera.focal_point = c
    p.camera.position = c + np.array([0.20, -0.17, -0.26])
    p.camera.up = (0.0, -1.0, 0.0)
    p.reset_camera()
    p.camera.zoom(zoom)


def _materials(p, st, regions, grid=None, skip=()):
    from tcadview import model
    names = list(st.grid.field_data["region_names"])
    for k, name in enumerate(names):
        if name in HIDE_3D or name in skip:
            continue
        surf = model._voxel_boundary(grid if grid is not None else st.grid, k)
        if surf is None:
            continue
        p.add_mesh(surf, color=model.material_color(st.region_material(k)), show_edges=False,
                   lighting=True, ambient=0.25, diffuse=0.8, specular=0.1)
        regions.append(name)


def _label(p, text, size=18):
    p.add_text(text, position="upper_left", font_size=size, color="black")


def fig_fab_structure(sess):
    st = sess.state(len(sess) - 1)
    p = _plotter(size=(1400, 1050))
    _materials(p, st, [])
    _camera(p, st.grid, 1.1)
    _label(p, "Planar NMOS from the virtual fab (ILD hidden): Si, STI, gate oxide, TiN gate, SiN spacer, "
              "TiN S/D contacts", 12)
    p.screenshot(os.path.join(IMG, "fab_structure.png"))
    p.close()


def fig_fab_doping(sess):
    """Cutaway at mid-width: silicon coloured by net doping on every voxel
    face, the rest by material."""
    from tcadview import model
    st = sess.state(len(sess) - 1)
    g = st.grid
    nx, ny, nz = g.dimensions
    half = g.extract_subset((0, nx - 1, 0, ny - 1, nz // 2, nz - 1))     # far half: the camera sees the cut face
    p = _plotter(size=(1400, 1050))
    names = list(g.field_data["region_names"])
    si = names.index("substrate")
    surf = model._voxel_boundary(half, si)
    surf.point_data["net doping"] = model.scale_values(surf.point_data["DopingConcentration"], "signed log")
    p.add_mesh(surf, scalars="net doping", cmap="RdBu_r", clim=(-21, 21), ambient=0.3,
               scalar_bar_args=dict(title="sign(N) log10(1+|N|),  N = Nd - Na (cm^-3)", vertical=False,
                                    position_x=0.22, position_y=0.04, width=0.56, height=0.07,
                                    color="black", title_font_size=16, label_font_size=14, n_labels=5,
                                    fmt="%.0f"))
    _materials(p, st, [], grid=half, skip={"substrate"})
    _camera(p, g, 1.1)
    _label(p, "Mid-width cutaway: net doping (red n+, blue p) - S/D, extensions, contact implant, well", 12)
    p.screenshot(os.path.join(IMG, "fab_doping_cutaway.png"))
    p.close()


def fig_fab_steps(sess):
    labels = [str(sess.state(i).grid.field_data["label"][0]) for i in range(len(sess))]
    picks = ["STI CMP", "gate etch", "spacer etch", "drain contact fill"]
    titles = ["1. STI trench, fill, CMP", "2. gate stack + etch", "3. extensions, spacer",
              "4. S/D implant, ILD, contacts"]
    p = _plotter(shape=(1, 4), size=(2400, 650))
    last = sess.state(len(sess) - 1).grid
    for col, (lab, title) in enumerate(zip(picks, titles)):
        p.subplot(0, col)
        st = sess.state(labels.index(lab))
        _materials(p, st, [])
        _camera(p, last, 1.35)
        p.add_text(title, position="upper_left", font_size=14, color="black")
    p.screenshot(os.path.join(IMG, "fab_process_steps.png"))
    p.close()


def figs_3d():
    sess = _fab_session()
    fig_fab_structure(sess)
    fig_fab_doping(sess)
    fig_fab_steps(sess)


def main(argv):
    os.makedirs(IMG, exist_ok=True)
    which = set(argv) or {"3d", "2d"}
    if "3d" in which:
        figs_3d()
    if "2d" in which:
        figs_2d()
    if "2d-old" in which:
        figs_2d_old()


# --- 2D: field maps from the drivers' VTK output ------------------------------
def _vtu(rel):
    import pyvista as pv
    path = os.path.join(ROOT, "out", rel)
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run the 2D drivers with --vtk first (see the docstring)")
    return pv.read(path)


class _Map:
    """One 2D VTK state on a matplotlib triangulation (y = depth, down)."""

    def __init__(self, m, delaunay=False):
        """delaunay: re-triangulate the nodes (Delaunay) for display - the
        point-cloud meshes of the 2D diode/MOS-cap have long slivers that
        smear a gouraud plot into streaks. Oxide = triangles above y = 0."""
        import matplotlib.tri as mtri
        self.m = m
        self.x, self.y = m.points[:, 0], m.points[:, 1]
        if delaunay:
            self.tri = mtri.Triangulation(self.x, self.y)
            self.oxide = self.y[self.tri.triangles].mean(axis=1) < 0.0
        else:
            self.tri = mtri.Triangulation(self.x, self.y, m.cells.reshape(-1, 4)[:, 1:])
            mats = list(m.field_data["material_names"])
            self.oxide = np.array([mats[i] == "SiO2" for i in np.asarray(m.cell_data["material_id"])])
        names = list(m.field_data["region_names"])
        self.label = str(m.field_data["label"][0])
        self.names = names

    def semi(self):
        """Triangulation of the semiconductor only."""
        import matplotlib.tri as mtri
        t = mtri.Triangulation(self.x, self.y, self.tri.triangles)
        t.set_mask(self.oxide)
        return t

    def field(self, name):
        return np.asarray(self.m.point_data[name])

    def junction(self, ax, **kw):
        d = self.field("DopingConcentration")
        ax.tricontour(self.semi(), d, levels=[0.0], colors="white", linewidths=1.0, linestyles="--", **kw)

    def oxide_patch(self, ax):
        if self.oxide.any():
            import matplotlib.tri as mtri
            t = mtri.Triangulation(self.x, self.y, self.tri.triangles)
            t.set_mask(~self.oxide)
            ax.tripcolor(t, np.zeros(len(self.x)), cmap="Greys", vmin=-1, vmax=4, shading="flat")


def _potential(ax, fig, mp, lim=None, title="electrostatic potential (V)"):
    psi = mp.field("ElectrostaticPotential")
    t = mp.semi()
    im = ax.tripcolor(t, psi, cmap="RdBu_r", shading="gouraud")
    ax.tricontour(t, psi, levels=12, colors="k", linewidths=0.4)
    mp.oxide_patch(ax)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title, fontsize=10)


def _logdens(ax, fig, mp, name, title, vmin=0, vmax=20, cmap="viridis"):
    v = np.log10(np.maximum(mp.field(name), 10.0 ** vmin))
    im = ax.tripcolor(mp.semi(), v, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
    mp.oxide_patch(ax)
    mp.junction(ax)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title, fontsize=10)


def _current(ax, fig, mp, xlim, ylim, name="TotalCurrentDensity", title="|J| (A/cm^2) and current lines",
             vmin=None, nx=160, ny=80):
    import matplotlib.tri as mtri
    J = mp.field(name)[:, :2]
    mag = np.hypot(J[:, 0], J[:, 1])
    lg = np.log10(np.maximum(mag, 1e-30))
    hi = np.nanmax(lg)
    lo = hi - 8 if vmin is None else vmin
    t = mp.semi()
    im = ax.tripcolor(t, np.clip(lg, lo, hi), cmap="inferno", vmin=lo, vmax=hi, shading="gouraud")
    mp.oxide_patch(ax)
    gx, gy = np.linspace(*xlim, nx), np.linspace(*ylim, ny)
    X, Y = np.meshgrid(gx, gy)
    U = mtri.LinearTriInterpolator(t, J[:, 0])(X, Y)
    V = mtri.LinearTriInterpolator(t, J[:, 1])(X, Y)
    U, V = np.ma.filled(U, np.nan), np.ma.filled(V, np.nan)
    sp = np.hypot(U, V)
    ok = np.isfinite(sp) & (np.log10(np.maximum(sp, 1e-30)) > lo + 2)
    U[~ok], V[~ok] = np.nan, np.nan
    ax.streamplot(gx, gy, U, V, color="#7fe0ff", linewidth=0.6, density=1.4, arrowsize=0.7)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("log10", fontsize=8)
    ax.set_title(title, fontsize=10)


def _finish(axes, xlim, ylim, unit="um"):
    for ax in np.ravel(axes):
        ax.set_xlim(*xlim)
        ax.set_ylim(ylim[1], ylim[0])
        ax.set_aspect("auto")
        ax.set_xlabel(f"x ({unit})", fontsize=9)
        ax.set_ylabel(f"depth y ({unit})", fontsize=9)
        ax.tick_params(labelsize=8)


def fig_mosfet2d():
    import matplotlib.pyplot as plt
    rows = [("idvg_Vds1_+0.0000.vtu", "off"), ("idvg_Vds1_+1.2000.vtu", "on")]
    xlim, ylim = (0.0, 1.1), (-0.005, 0.25)
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.4), constrained_layout=True)
    for r, (f, tag) in enumerate(rows):
        mp = _Map(_vtu(os.path.join("input_mosfet_2d", "vtk", f)))
        _potential(axes[r, 0], fig, mp, title=f"{tag}: {mp.label}   potential (V)")
        _logdens(axes[r, 1], fig, mp, "eDensity", "log10 electron density (cm^-3), dashed: junction",
                 vmin=2, vmax=20.5)
        _current(axes[r, 2], fig, mp, xlim, (0.0, ylim[1]), name="eCurrentDensity",
                 title="log10 |Jn| (A/cm^2) and electron current lines")
    _finish(axes, xlim, ylim)
    fig.suptitle("2D NMOS, Lg = 0.5 um, Vds = 1 V: off (Vgs = 0) and on (Vgs = 1.2 V)", fontsize=12)
    fig.savefig(os.path.join(IMG, "mosfet2d_fields.png"), dpi=130)
    plt.close(fig)


def fig_diode2d():
    import matplotlib.pyplot as plt
    rev = _Map(_vtu(os.path.join("input_diode_2d", "vtk", "diode2d_-1.0000.vtu")), delaunay=True)
    fwd = _Map(_vtu(os.path.join("input_diode_2d", "vtk", "diode2d_+0.5000.vtu")), delaunay=True)
    xlim, ylim = (3.0, 17.0), (0.0, 7.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True)
    _potential(axes[0], fig, rev, title=f"reverse {rev.label}: potential (V)")
    E = np.asarray(rev.m.point_data["ElectricField"])[:, :2]
    rev.m.point_data["_absE"] = np.hypot(E[:, 0], E[:, 1])
    _logdens(axes[1], fig, rev, "_absE", f"reverse {rev.label}: log10 |E| (V/cm)", vmin=1, vmax=4.5,
             cmap="magma")
    _current(axes[2], fig, fwd, xlim, ylim, title=f"forward {fwd.label}: log10 |J| (A/cm^2), current lines")
    _finish(axes, xlim, ylim)
    fig.suptitle("2D diode: p-well in an n substrate", fontsize=12)
    fig.savefig(os.path.join(IMG, "diode2d_fields.png"), dpi=130)
    plt.close(fig)


def fig_mos2d():
    import matplotlib.pyplot as plt
    mp = _Map(_vtu(os.path.join("input_mos_2d", "vtk", "mos2d_+1.2000.vtu")), delaunay=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True)
    _potential(axes[0], fig, mp, title=f"inversion {mp.label}: potential (V)")
    _logdens(axes[1], fig, mp, "hDensity", "log10 hole density (cm^-3): depletion", vmin=0, vmax=17,
             cmap="magma")
    _logdens(axes[2], fig, mp, "eDensity", "log10 electron density (cm^-3): inversion layer", vmin=0,
             vmax=19)
    _finish(axes[:2], (4.0, 16.0), (-0.01, 0.8))
    _finish(axes[2:], (5.0, 15.0), (-0.01, 0.1))
    fig.suptitle("2D MOS capacitor (gate over x = 7-13 um, p substrate)", fontsize=12)
    fig.savefig(os.path.join(IMG, "mos2d_fields.png"), dpi=130)
    plt.close(fig)


def figs_2d():
    fig_mosfet2d()


def figs_2d_old():
    """The point-cloud-mesh 2D diode and MOS-cap: their solutions are noisy
    on that mesh, so these are not in the README until the drivers move to
    the quadtree mesh."""
    fig_diode2d()
    fig_mos2d()


if __name__ == "__main__":
    main(sys.argv[1:])
