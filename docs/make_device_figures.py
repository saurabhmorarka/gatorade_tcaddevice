"""Structure figures of the FinFET and nanosheet templates in docs/img/.

    python3 docs/make_device_figures.py              # all devices
    python3 docs/make_device_figures.py finfet       # one

Per device (builds configs/fab_<device>.yaml into out/fab_<device>/ if
needed):
  <device>_3d.png       the device, ILD hidden
  <device>_gatecut.png  cut through the gate centre: the fin / sheets
                        wrapped by the gate dielectric and metal
  <device>_cut.png      mid-width cutaway, silicon coloured by net doping
  <device>_cuts.png     2D cuts (gate cross-section, along the channel):
                        materials and net doping
and, for all six NMOS/PMOS FinFET/nanosheet configs together,
  sd_epi_cuts.png       S/D cross-sections (y-z) through the contact (the
                        plug into the epi) and between the contact and the
                        spacer (the epi shapes), the neighbouring cells'
                        epi (mirrored) faded; net doping

    python3 docs/make_device_figures.py sdcut        # that figure only

The quantum vs classical figures come from quantum/compare_3d.py.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs"))
IMG = os.path.join(ROOT, "docs", "img")

TITLES = {
    "finfet": "Bulk FinFET, CPP 48 nm, Lg 15 nm, fin 5 x 50 nm",
    "nanosheet": "Nanosheet (PTS isolation), 3 x 5 nm sheets 25 nm wide, CPP 48 nm, Lg 15 nm",
    "nanosheet_bdi": "Nanosheet (bottom dielectric isolation), 3 x 5 nm sheets 25 nm wide, CPP 48 nm, Lg 15 nm",
}
SEMI_NAMES = ("substrate", "sheets", "sd_epi")
EPI_COLOR, CONTACT_COLOR = "#5a7fa8", "#c9a227"   # set apart from the substrate / gate (both Si / TiN)


def _session(dev):
    from tcadview import model
    pvd = os.path.join(ROOT, "out", f"fab_{dev}", "build.pvd")
    if not os.path.exists(pvd):
        from fab.__main__ import main as fab_main
        fab_main(["build", os.path.join(ROOT, "configs", f"fab_{dev}.yaml"), "--out", os.path.dirname(pvd)])
    return model.Session(pvd)


def _camera(p, grid, offset, zoom):
    c = np.array(grid.center)
    p.camera.focal_point = c
    p.camera.position = c + np.array(offset)
    p.camera.up = (0.0, -1.0, 0.0)
    p.reset_camera()
    p.camera.zoom(zoom)


def fig_3d(dev, sess):
    import make_readme_figures as M
    st = sess.state(len(sess) - 1)
    p = M._plotter(size=(1300, 1100))
    M._materials(p, st, [], skip={"ild"})
    _camera(p, st.grid, (0.26, -0.20, -0.20), 1.15)
    M._label(p, TITLES[dev] + " (ILD hidden)", 13)
    p.screenshot(os.path.join(IMG, f"{dev}_3d.png"))
    p.close()


def fig_gatecut(dev, sess):
    import make_readme_figures as M
    st = sess.state(len(sess) - 1)
    g = st.grid
    nx, ny, nz = g.dimensions
    cut = g.extract_subset((nx // 2, nx - 1, 0, ny - 1, 0, nz - 1))
    p = M._plotter(size=(1300, 1100))
    M._materials(p, st, [], grid=cut, skip={"ild"})
    _camera(p, g, (-0.30, -0.14, -0.16), 1.25)
    M._label(p, TITLES[dev] + ": cut through the gate", 13)
    p.screenshot(os.path.join(IMG, f"{dev}_gatecut.png"))
    p.close()


def fig_doping_cut(dev, sess):
    import make_readme_figures as M
    from tcadview import model
    st = sess.state(len(sess) - 1)
    g = st.grid
    nx, ny, nz = g.dimensions
    half = g.extract_subset((0, nx - 1, 0, ny - 1, nz // 2, nz - 1))
    names = list(g.field_data["region_names"])
    semi = [k for k, n in enumerate(names) if n in SEMI_NAMES]
    p = M._plotter(size=(1300, 1100))
    for k in semi:
        surf = model._voxel_boundary(half, k)
        if surf is None:
            continue
        N = np.asarray(surf.point_data["DopingConcentration"])
        surf.point_data["net doping"] = np.sign(N) * np.clip(np.log10(np.maximum(np.abs(N), 1.0)) - 15, 0, 6)
        p.add_mesh(surf, scalars="net doping", cmap="RdBu_r", clim=(-6, 6), ambient=0.3,
                   show_scalar_bar=(k == semi[0]),
                   scalar_bar_args=dict(title="net doping: 0 = below 1e15, +-6 = 1e21 cm^-3 (red n, blue p)",
                                        vertical=False, position_x=0.2, position_y=0.03, width=0.6, height=0.06,
                                        color="black", title_font_size=14, label_font_size=12, n_labels=7,
                                        fmt="%.0f"))
    M._materials(p, st, [], grid=half, skip={"ild"} | {names[k] for k in semi})
    _camera(p, g, (0.26, -0.20, -0.20), 1.15)
    M._label(p, TITLES[dev] + ": mid-width cutaway, silicon by net doping", 13)
    p.screenshot(os.path.join(IMG, f"{dev}_cut.png"))
    p.close()


def fig_cuts(dev):
    """2D cuts from the fab structure itself: materials and net doping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from fab import flow
    from tcadview.model import MATERIAL_COLORS
    st, _ = flow.build(flow.load(os.path.join(ROOT, "configs", f"fab_{dev}.yaml")), log=lambda *a: None)
    lab = st.cell_labels()
    x, y, z = st.grid.axes
    mats = [r.material for r in st.regions]
    names = [r.name for r in st.regions]
    cols = ["#ffffff"] + [EPI_COLOR if n == "sd_epi" else CONTACT_COLOR if n.endswith("contact")
                          else MATERIAL_COLORS.get(m, "#cccccc") for m, n in zip(mats, names)]
    cmap = ListedColormap(cols)
    nd, na, net = st.net_doping()
    semi = st.semiconductor_nodes()
    ix = int(np.argmin(np.abs(x - (x[0] + x[-1]) / 2)))
    iz = int(np.argmin(np.abs(z - (z[0] + z[-1]) / 2)))
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw=dict(width_ratios=[1, 1.6]))

    def mat(a, L, h, v, xl):                         # L: (h cells, v cells)
        a.pcolormesh(h, -v, (L + 1).T, cmap=cmap, vmin=-0.5, vmax=len(cols) - 0.5, shading="flat")
        a.set(aspect="equal", xlabel=xl, ylabel="height (nm)")
        return set(np.unique(L)) - {-1}

    def dop(a, N, S, h, v, xl):
        val = np.where(S, np.sign(N) * np.clip(np.log10(np.maximum(np.abs(N), 1.0)) - 15, 0, 6), np.nan)
        a.set(aspect="equal", xlabel=xl, ylabel="height (nm)")
        return a.pcolormesh(h, -v, val.T, cmap="RdBu_r", vmin=-6, vmax=6, shading="nearest")

    used = mat(ax[0, 0], lab[min(ix, lab.shape[0] - 1)].T, z, y, "z (nm)")
    used |= mat(ax[0, 1], lab[:, :, min(iz, lab.shape[2] - 1)], x, y, "x (nm)")
    dop(ax[1, 0], net[ix].T, semi[ix].T, z, y, "z (nm)")
    im = dop(ax[1, 1], net[:, :, iz], semi[:, :, iz], x, y, "x (nm)")
    ax[0, 0].set_title("cut through the gate (y-z)")
    ax[0, 1].set_title("cut along the channel at mid-width (x-y)")
    ax[1, 0].set_title("net doping, gate cut")
    ax[1, 1].set_title("net doping, channel cut")
    cb = fig.colorbar(im, ax=ax[1, :], shrink=0.8)
    cb.set_ticks(range(-6, 7, 2))
    cb.set_ticklabels(["-1e21", "-1e19", "-1e17", "<1e15", "1e17", "1e19", "1e21"])
    cb.set_label("net doping (cm$^{-3}$): red donors, blue acceptors")
    ax[0, 1].legend(handles=[Patch(color=cols[k + 1], label=f"{names[k]} ({mats[k]})") for k in sorted(used)],
                    loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.suptitle(TITLES[dev] + ": materials and net doping")
    fig.savefig(os.path.join(IMG, f"{dev}_cuts.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


SD_DEVICES = [("finfet", "FinFET NMOS (Si:P blob)"), ("finfet_pmos", "FinFET PMOS (SiGe diamond, merged)"),
              ("nanosheet", "nanosheet PTS NMOS"), ("nanosheet_pmos", "nanosheet PTS PMOS"),
              ("nanosheet_bdi", "nanosheet BDI NMOS"), ("nanosheet_bdi_pmos", "nanosheet BDI PMOS")]


def fig_sd_cuts(x_cuts=(4.0, 10.0)):
    """S/D cross-sections of the six FinFET / nanosheet configs: materials
    at each x of x_cuts (through the contact, then between the contact and
    the outer spacer), the mirrored neighbouring cells faded (the side walls
    are symmetry planes), and the net doping at the last one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from fab import flow
    from tcadview.model import MATERIAL_COLORS
    nr = len(x_cuts) + 1
    fig, axs = plt.subplots(nr, len(SD_DEVICES), figsize=(4.0 * len(SD_DEVICES), 5 * nr), squeeze=False)
    legend = {}
    for j, (dev, title) in enumerate(SD_DEVICES):
        st, _ = flow.build(flow.load(os.path.join(ROOT, "configs", f"fab_{dev}.yaml")), log=lambda *a: None)
        lab = st.cell_labels()
        x, y, z = st.grid.axes
        names = [r.name for r in st.regions]
        cols = ["#ffffff"] + [EPI_COLOR if n == "sd_epi" else CONTACT_COLOR if n.endswith("contact")
                              else MATERIAL_COLORS.get(r.material, "#cccccc") for n, r in zip(names, st.regions)]
        cmap = ListedColormap(cols)
        for r, x_cut in enumerate(x_cuts):
            ix = int(np.argmin(np.abs(x - x_cut)))
            L = lab[min(ix, lab.shape[0] - 1)]                            # (y, z) cells
            a = axs[r, j]
            for zz, LL, alpha in ((2 * z[0] - z[::-1], L[:, ::-1], 0.4), (z, L, 1.0),
                                  (2 * z[-1] - z[::-1], L[:, ::-1], 0.4)):
                a.pcolormesh(zz, -y, LL + 1, cmap=cmap, vmin=-0.5, vmax=len(cols) - 0.5, shading="flat", alpha=alpha)
            for w in (z[0], z[-1]):
                a.axvline(w, color="k", lw=0.6, ls=":")
            a.set(aspect="equal", xlabel="z (nm)", ylabel="height (nm)" if j == 0 else None,
                  title=(f"{title}\n" if r == 0 else "") + f"S/D cut at x = {x[ix]:g} nm"
                  + (" (contact)" if r == 0 else ""))
            for k in set(np.unique(L)) - {-1}:
                legend.setdefault(f"{names[k]} ({st.regions[k].material})", cols[k + 1])
        nd, na, net = st.net_doping()
        semi = st.semiconductor_nodes()
        val = np.where(semi[ix], np.sign(net[ix]) * np.clip(np.log10(np.maximum(np.abs(net[ix]), 1.0)) - 15, 0, 6),
                       np.nan)
        im = axs[-1, j].pcolormesh(z, -y, val, cmap="RdBu_r", vmin=-6, vmax=6, shading="nearest")
        axs[-1, j].set(aspect="equal", xlabel="z (nm)", ylabel="height (nm)" if j == 0 else None,
                       title=f"net doping, x = {x[ix]:g} nm")
    cb = fig.colorbar(im, ax=axs[-1, :], shrink=0.6, location="bottom", pad=0.08)
    cb.set_ticks(range(-6, 7, 2))
    cb.set_ticklabels(["-1e21", "-1e19", "-1e17", "<1e15", "1e17", "1e19", "1e21"])
    cb.set_label("net doping (cm$^{-3}$): red donors, blue acceptors")
    fig.legend(handles=[Patch(color=c, label=n) for n, c in legend.items()], loc="upper center",
               ncol=min(len(legend), 7), fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("S/D epitaxy: NMOS Si:P blob, PMOS Si0.5Ge0.5:B {111} diamond (merged only where the pitch "
                 "forces it); contact plug into the epi", y=1.03)
    fig.savefig(os.path.join(IMG, "sd_epi_cuts.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(argv):
    os.makedirs(IMG, exist_ok=True)
    if argv == ["sdcut"]:
        fig_sd_cuts()
        print("wrote sd_epi_cuts")
        return
    for dev in (argv or list(TITLES)):
        sess = _session(dev)
        fig_3d(dev, sess)
        fig_gatecut(dev, sess)
        fig_doping_cut(dev, sess)
        fig_cuts(dev)
        print("wrote", dev)
    if not argv:
        fig_sd_cuts()
        print("wrote sd_epi_cuts")


if __name__ == "__main__":
    main(sys.argv[1:])
