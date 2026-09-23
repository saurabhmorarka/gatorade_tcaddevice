"""Plot library for structure+fields files (see structure_io.py for the
schema): device cross-section, textbook band diagrams, and charge-density
decomposition. New plot types belong here as this project grows.

Every function takes an already-loaded structure dict (from
structure_io.load_structure(), or built in-memory by a driver without
touching disk) so it doesn't care whether it was called from main.py,
mos_main.py, or the standalone CLI below. Each dispatches on doc["dim"] and
only implements dim=1 today; a future 2D/3D structure would add a dim==2/3
branch to each of these without touching the dim==1 code path.

Standalone usage (given only a *_structure.json file, e.g. shared by
someone else): `python3 plot.py out/diode_structure.json`
"""
import argparse
import os
import sys

import matplotlib
# Library use (imported by main.py/mos_main.py, which already picked a
# backend before importing this module) is unaffected either way; this
# only matters for standalone CLI use, where `--interactive` needs a real
# GUI backend instead of the headless one everything else here uses -
# and the backend must be chosen before matplotlib.pyplot is ever
# imported, i.e. before argparse has even run, hence the raw sys.argv check.
if "--interactive" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import structure_io as sio

C_GRID = "#c9c9c9"
BAND_FIELDS = ["Evac", "Ec", "Ev", "Ei", "Ef"]
CHARGE_FIELDS = ["fixed", "mobile", "net"]
REGION_COLORS = {
    ("semiconductor", "p"): "#f4a259",
    ("semiconductor", "n"): "#5fa8d3",
    ("insulator", None): "#d9d9d9",
}


def _region_color(region):
    key = (region.get("kind"), region.get("doping_type"))
    if key in REGION_COLORS:
        return REGION_COLORS[key]
    return REGION_COLORS.get((region.get("kind"), None), "#bbbbbb")


def _require_1d(doc, fn_name):
    if doc["dim"] != 1:
        raise NotImplementedError(f"{fn_name}: only dim=1 structures are supported so far "
                                   f"(got dim={doc['dim']})")


def plot_structure(doc, ax=None, xlim_um=None, label_regions=True):
    """Device cross-section: regions colored by material/doping type, with
    mesh node positions drawn as tick marks so the (adaptive) mesh
    refinement is visible directly - dense near junctions/interfaces,
    coarsening into the bulk. Pass `xlim_um` to zoom into a sub-range (e.g.
    a nm-scale oxide layer that would otherwise be invisible next to a
    um-scale substrate) - region labels are hidden in a zoom by default
    since they'd usually fall outside it."""
    _require_1d(doc, "plot_structure")
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 2.8))
    x_um = np.array(doc["grid"]["x_um"])
    if xlim_um is not None:
        mask = (x_um >= xlim_um[0]) & (x_um <= xlim_um[1])
        x_um = x_um[mask]

    for region in doc["regions"]:
        x0, x1 = region["x_range_um"]
        ax.axvspan(x0, x1, color=_region_color(region), alpha=0.85, lw=0, zorder=0)
        if label_regions and xlim_um is None:
            ax.text((x0 + x1) / 2.0, 1.4, region["name"], ha="center", va="bottom", fontsize=9)
        ax.axvline(x0, color="#555555", lw=1, zorder=1)
    ax.axvline(doc["regions"][-1]["x_range_um"][1], color="#555555", lw=1, zorder=1)

    ax.plot(x_um, np.zeros_like(x_um), "|", color="#222222", ms=16, mew=1.0, zorder=2)
    ax.set_ylim(-1, 2.2)
    ax.set_yticks([])
    xlim = xlim_um if xlim_um is not None else (doc["regions"][0]["x_range_um"][0],
                                                  doc["regions"][-1]["x_range_um"][1])
    ax.set_xlim(*xlim)
    ax.set_xlabel("x (um)")
    ax.set_title(f"{doc['device']} structure  ({len(x_um)} mesh nodes"
                 f"{' in view' if xlim_um is not None else ''})")
    if own_fig:
        fig.tight_layout()
    return ax


def _broadcast_material(value, n):
    arr = np.asarray(value, dtype=float)
    return arr if arr.ndim > 0 else np.full(n, float(arr))


def _band_energies(doc, bias_point):
    """Ec, Ev, Ei (midgap approximation), vacuum level, and electron/hole
    quasi-Fermi energies, all in eV, from psi/phin/phip (all in volts).

    The vacuum level is taken as the PRIMARY, continuous quantity -
    E_vacuum(x) = -psi(x) (arbitrary zero at psi=0) - and Ec/Ev are derived
    from it via each material's own electron affinity/bandgap (Anderson's
    rule: Ec = E_vacuum - chi, Ev = Ec - Eg). This matters wherever chi/Eg
    vary with position (e.g. the MOS-cap's oxide vs. substrate): deriving
    Ec/Ev from a shared, continuous vacuum level is what produces the
    correct conduction/valence-band offset at a material interface: doing
    it the other way around (splitting a shared, continuous Ei by +-Eg/2)
    would instead force E_vacuum itself to jump at the interface, which is
    unphysical. For a single uniform material (the diode) the two are
    equivalent up to a constant shift.

    Efn/Efp follow from the same psi-phin/psi-phip relations physics.py
    uses (n = ni*exp((psi-phin)/Vt), p = ni*exp((phip-psi)/Vt)): a single
    flat Ef when phin=phip=0, i.e. true equilibrium."""
    psi = np.array(bias_point["fields"]["psi"])
    n = len(psi)
    chi = _broadcast_material(doc["material"]["chi_eV"], n)
    Eg = _broadcast_material(doc["material"]["Eg_eV"], n)
    phin = np.nan_to_num(np.array(bias_point["fields"].get("phin", np.zeros(n)), dtype=float))
    phip = np.nan_to_num(np.array(bias_point["fields"].get("phip", np.zeros(n)), dtype=float))

    Evac = -psi
    Ec = Evac - chi
    Ev = Ec - Eg
    Ei = (Ec + Ev) / 2.0
    Efn = Ei + (psi - phin)
    Efp = Ei - (phip - psi)
    return Ec, Ev, Ei, Evac, Efn, Efp


def plot_bands(doc, bias_index=0, ax=None, xlim_um=None, fields=None):
    """Textbook band diagram: E_c, E_v, E_i, vacuum level, and the electron/
    hole quasi-Fermi levels (drawn as a single flat E_f when the structure
    is in true equilibrium, i.e. phin=phip=0 everywhere). E_i is the
    midgap approximation (Ec+Ev)/2, not the exact Nc/Nv-weighted intrinsic
    level - fine for a teaching plot, not for precision analytic
    comparison (see analytic.py/mos_analytic.py for that).

    `fields`: subset of BAND_FIELDS ("Evac","Ec","Ev","Ei","Ef") to draw;
    None (default) draws all of them."""
    _require_1d(doc, "plot_bands")
    fields = set(BAND_FIELDS) if fields is None else set(fields)
    bp = doc["bias_points"][bias_index]
    Ec, Ev, Ei, Evac, Efn, Efp = _band_energies(doc, bp)
    x_um = np.array(doc["grid"]["x_um"])
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    if "Evac" in fields:
        ax.plot(x_um, Evac, color="#999999", lw=1.2, ls=":", label="E_vacuum")
    if "Ec" in fields:
        ax.plot(x_um, Ec, color="#1f6feb", lw=2, label="E_c")
    if "Ev" in fields:
        ax.plot(x_um, Ev, color="#c2410c", lw=2, label="E_v")
    if "Ei" in fields:
        ax.plot(x_um, Ei, color="#888888", lw=1.2, ls="--", label="E_i (midgap approx.)")
    if "Ef" in fields:
        if np.allclose(Efn, Efp):
            ax.plot(x_um, Efn, color="#1a7f37", lw=1.8, label="E_f")
        else:
            ax.plot(x_um, Efn, color="#1a7f37", lw=1.8, label="E_fn (electron quasi-Fermi)")
            ax.plot(x_um, Efp, color="#9a6700", lw=1.8, ls="--", label="E_fp (hole quasi-Fermi)")

    if xlim_um is not None:
        ax.set_xlim(*xlim_um)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Energy (eV)")
    label = bp.get("label") or f"bias={bp['bias']:.3f}"
    ax.set_title(f"{doc['device']} band diagram, {label}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, color=C_GRID)
    if own_fig:
        fig.tight_layout()
    return ax


def plot_charge(doc, bias_index=0, ax=None, xlim_um=None, fields=None):
    """Charge-density decomposition: fixed (ionized dopant) charge vs.
    mobile carrier charge vs. their sum (net charge), all in units of
    elementary charges/cm^3 (not multiplied by q) so they're directly
    comparable to the doping numbers used elsewhere in this project.

    `fields`: subset of CHARGE_FIELDS ("fixed","mobile","net") to draw;
    None (default) draws all of them."""
    _require_1d(doc, "plot_charge")
    fields = set(CHARGE_FIELDS) if fields is None else set(fields)
    bp = doc["bias_points"][bias_index]
    x_um = np.array(doc["grid"]["x_um"])
    Cdop = np.array(doc["doping_cm3"])
    n = np.array(bp["fields"]["n"])
    p = np.array(bp["fields"]["p"])
    mobile = p - n
    net = Cdop + mobile

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    regime = bp.get("regime")
    if regime:
        ax.text(0.02, 0.96, f"regime: {regime}", transform=ax.transAxes, fontsize=9,
                va="top", ha="left", color="#555555",
                bbox=dict(boxstyle="round", fc="#f0f0f0", ec="none"))

    ax.axhline(0, color="#333333", lw=0.8)
    if "fixed" in fields:
        ax.plot(x_um, Cdop, color="#6a4c93", lw=1.8, ls="--", label="Fixed (ionized dopant) charge, N_D-N_A")
    if "mobile" in fields:
        ax.plot(x_um, mobile, color="#1f6feb", lw=1.8, label="Mobile carrier charge, p-n")
    if "net" in fields:
        ax.plot(x_um, net, color="#c2410c", lw=2.2, label="Net charge, (N_D-N_A)+(p-n)")

    # Interface (sheet) charge, e.g. a fixed Si/SiO2 Qit - a genuinely
    # different quantity (C/cm^2, areal) from the volumetric (cm^-3) traces
    # above, so it can't share their y-scale as a literal spike; drawn as a
    # labeled marker/arrow at its location instead, the usual textbook
    # convention for an interface charge sheet.
    for iface in doc.get("interfaces", []):
        y0, y1 = ax.get_ylim()
        yspan = y1 - y0 if y1 > y0 else 1.0
        color = "#c2185b"
        ax.annotate(
            f"Q$_{{it}}$ = {iface['Qit_cm2']:.3g} C/cm$^2$",
            xy=(iface["x_um"], y0 + 0.82 * yspan), xytext=(iface["x_um"], y0 + 0.98 * yspan),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
            ha="center", va="top", fontsize=8, color=color,
        )
        ax.axvline(iface["x_um"], color=color, lw=1.0, ls=":", alpha=0.6)

    if xlim_um is not None:
        ax.set_xlim(*xlim_um)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Charge density / q (cm^-3)")
    label = bp.get("label") or f"bias={bp['bias']:.3f}"
    ax.set_title(f"{doc['device']} charge density, {label}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, color=C_GRID)
    if own_fig:
        fig.tight_layout()
    return ax


PLOTTERS = {"structure": plot_structure, "bands": plot_bands, "charge": plot_charge}


def _interactive_show(axes):
    """Open one live matplotlib window covering all of `axes` side by side,
    with a single checkbox panel listing every labeled curve across all of
    them - so opening a structure file interactively just works, with
    every field already loaded and available to toggle on/off, rather than
    requiring the plot type to be picked on the command line first. Only
    makes sense with a real (non-Agg) backend, i.e. --interactive on the
    CLI below, which picks the backend before pyplot is ever imported."""
    from matplotlib.widgets import CheckButtons

    axes = list(axes)
    fig = axes[0].figure
    lines = [l for ax in axes for l in ax.get_lines()
             if l.get_label() and not l.get_label().startswith("_")]
    if lines:
        labels = [l.get_label() for l in lines]
        fig.subplots_adjust(right=0.82)
        check_ax = fig.add_axes([0.84, 0.3, 0.15, 0.4])
        check = CheckButtons(check_ax, labels, [l.get_visible() for l in lines])

        def toggle(label):
            # Toggle every line with this label (a curve can legitimately
            # appear on more than one of the shown axes).
            for l in lines:
                if l.get_label() == label:
                    l.set_visible(not l.get_visible())
            fig.canvas.draw_idle()

        check.on_clicked(toggle)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone plotter for gatorade_tcaddevice structure+fields JSON files "
                     "(see structure_io.py for the file format).")
    parser.add_argument("structure_path", help="Path to a *_structure.json file")
    parser.add_argument("--which", default="structure,bands,charge",
                         help="Comma-separated plot names to make: structure,bands,charge "
                              "(default: all three)")
    parser.add_argument("--bias-index", type=int, default=0,
                         help="Index into bias_points for the bands/charge plots (default: 0, "
                              "the first saved bias point)")
    parser.add_argument("--band-fields", default=None,
                         help=f"Comma-separated subset of {BAND_FIELDS} to draw on the band "
                              "diagram (default: all)")
    parser.add_argument("--charge-fields", default=None,
                         help=f"Comma-separated subset of {CHARGE_FIELDS} to draw on the "
                              "charge-density plot (default: all)")
    parser.add_argument("--out-dir", default=None,
                         help="Directory to write PNGs into (default: alongside the structure file)")
    parser.add_argument("--interactive", action="store_true",
                         help="Open one live window with every --which plot side by side and a "
                              "single checkbox panel to toggle any of their curves on/off - just "
                              "run with a structure file and this flag, no --which/--*-fields "
                              "needed up front. Requires a local display.")
    args = parser.parse_args()

    doc = sio.load_structure(args.structure_path)
    which = [n.strip() for n in args.which.split(",") if n.strip()]
    for name in which:
        if name not in PLOTTERS:
            raise SystemExit(f"Unknown plot name {name!r}; choose from {sorted(PLOTTERS)}")

    band_fields = args.band_fields.split(",") if args.band_fields else None
    charge_fields = args.charge_fields.split(",") if args.charge_fields else None

    def make(name, ax=None):
        kwargs = {} if name == "structure" else {"bias_index": args.bias_index}
        if name == "bands" and band_fields:
            kwargs["fields"] = band_fields
        if name == "charge" and charge_fields:
            kwargs["fields"] = charge_fields
        return PLOTTERS[name](doc, ax=ax, **kwargs)

    if args.interactive:
        fig, axes_raw = plt.subplots(1, len(which), figsize=(6.5 * len(which), 5.5), squeeze=False)
        axes = axes_raw[0]
        for ax, name in zip(axes, which):
            make(name, ax=ax)
        fig.tight_layout()
        _interactive_show(axes)
        return

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.structure_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.structure_path))[0]

    for name in which:
        ax = make(name)
        fig = ax.figure
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{stem}_{name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(out_path)


if __name__ == "__main__":
    main()
