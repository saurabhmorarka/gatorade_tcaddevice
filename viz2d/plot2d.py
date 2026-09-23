"""2D structure/mesh/field plotter - the 2D sibling to core/plot.py.
Renders device regions (colored rectangles), the point-cloud mesh (Delaunay
edges), and boundary-condition tags (contact/symmetry/free_surface points)
via plot_structure2d, and solved fields (psi/n/p/etc, per-point over the
triangulation) via plot_field2d.

Standalone usage: `python3 -m viz2d.plot2d out/diode2d_structure.json`
"""
import argparse
import os
import sys

import matplotlib
if "--interactive" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np

from core import structure_io as sio

REGION_COLORS = {
    "p": "#f4a259",
    "n": "#5fa8d3",
}
BC_COLORS = {
    "symmetry": "#4caf50",
    "free_surface": "#9e9e9e",
}
CONTACT_COLORS = ["#d62728", "#1f77b4", "#9467bd", "#8c564b"]


def _require_2d(doc, fn_name):
    if doc["dim"] != 2:
        raise NotImplementedError(f"{fn_name}: only dim=2 structures are supported "
                                   f"(got dim={doc['dim']})")


def _contact_color(name, contact_names):
    return CONTACT_COLORS[contact_names.index(name) % len(CONTACT_COLORS)]


def mesa_bbox_um(doc, margin_x_frac=0.15, margin_y_mult=3.0):
    """Bounding box (x0, x1, y0, y1), in um, of a mesa protrusion (a region
    with y_range_um[0] < 0, e.g. a MOS capacitor's oxide+gate stack) padded
    by a margin - or None if this structure has no mesa. Used to give the
    interactive viewer a dedicated zoomed inset, since a real oxide is
    routinely 100-1000x thinner than the substrate it sits on and is
    otherwise completely invisible at the structure's own true-scale plot
    (see mesh2d/geometry2d.py::TopMesa's module docstring)."""
    mesa_regions = [r for r in doc["regions"] if r["y_range_um"][0] < 0]
    if not mesa_regions:
        return None
    x0 = min(r["x_range_um"][0] for r in mesa_regions)
    x1 = max(r["x_range_um"][1] for r in mesa_regions)
    y0 = min(r["y_range_um"][0] for r in mesa_regions)
    width = x1 - x0
    height = -y0
    return (x0 - margin_x_frac * width, x1 + margin_x_frac * width,
            y0 - margin_y_mult * height, margin_y_mult * height)


def plot_structure2d(doc, ax=None, show_mesh=True, show_boundary=True, label_regions=True,
                      xlim=None, ylim=None, aspect="equal"):
    """Device cross-section + point-cloud mesh + boundary-condition tags.
    Returns (ax, layers) where `layers` maps a layer name ("regions", "mesh",
    "boundary") to the list of matplotlib artists in it, for toggling."""
    _require_2d(doc, "plot_structure2d")
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    x_um = np.array(doc["grid"]["x_um"])
    y_um = np.array(doc["grid"]["y_um"])
    layers = {"regions": [], "mesh": [], "boundary": []}

    for region in doc["regions"]:
        x0, x1 = region["x_range_um"]
        y0, y1 = region["y_range_um"]
        color = REGION_COLORS.get(region.get("doping_type"), "#bbbbbb")
        patch = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor="k",
                           linewidth=0.8, alpha=0.6, zorder=1)
        ax.add_patch(patch)
        layers["regions"].append(patch)
        if label_regions:
            txt = ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), region["name"],
                           ha="center", va="center", fontsize=8, zorder=3)
            layers["regions"].append(txt)

    mesh2d = doc.get("mesh2d", {})
    if show_mesh and mesh2d.get("triangles"):
        segs_x, segs_y = [], []
        seen = set()
        for (i, j, k) in mesh2d["triangles"]:
            for a, b in ((i, j), (j, k), (k, i)):
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                segs_x += [x_um[a], x_um[b], None]
                segs_y += [y_um[a], y_um[b], None]
        line, = ax.plot(segs_x, segs_y, color="#c9c9c9", linewidth=0.4, zorder=2,
                         label="_mesh")
        layers["mesh"].append(line)

    if show_boundary and mesh2d.get("boundary"):
        contact_names = sorted({b["bc_type"].split(":", 1)[1] for b in mesh2d["boundary"]
                                 if b["bc_type"].startswith("contact:")})
        by_type = {}
        for b in mesh2d["boundary"]:
            by_type.setdefault(b["bc_type"], []).append(b["point_index"])
        handles = []
        for bc_type, idxs in by_type.items():
            if bc_type.startswith("contact:"):
                name = bc_type.split(":", 1)[1]
                color = _contact_color(name, contact_names)
                label = f"contact: {name}"
            else:
                color = BC_COLORS.get(bc_type, "#000000")
                label = bc_type
            pts = ax.scatter(x_um[idxs], y_um[idxs], color=color, s=14, zorder=4, label=label)
            layers["boundary"].append(pts)
            handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                                   markersize=6, label=label))
        ax.legend(handles=handles, fontsize=7, loc="upper right")

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, max(x_um))
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(max(y_um), 0)  # y=0 is the top surface - keep it at the top of the plot
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um), depth from top surface")
    ax.set_aspect(aspect)
    ax.set_title(f"{doc['device']} 2D structure")
    if own_fig:
        fig.tight_layout()
    return ax, layers


def _interactive_show(ax, layers):
    from matplotlib.widgets import CheckButtons

    fig = ax.figure
    names = [n for n in layers if layers[n]]
    if names:
        fig.subplots_adjust(right=0.8)
        check_ax = fig.add_axes([0.82, 0.4, 0.16, 0.2])
        check = CheckButtons(check_ax, names, [True] * len(names))

        def toggle(name):
            for artist in layers[name]:
                artist.set_visible(not artist.get_visible())
            fig.canvas.draw_idle()

        check.on_clicked(toggle)
    plt.show()


def plot_field2d(points_um, triangles, values, ax=None, title=None, cmap="viridis",
                  log_scale=False, label=None):
    """Render a per-point field (psi, n, p, ...) over the triangulation via
    tripcolor (Gouraud-shaded, i.e. linearly interpolated across each
    triangle from its vertex values - the natural analog of core/plot.py's
    line plots for a field defined on an unstructured 2D mesh). `log_scale`
    plots log10(values) (for n/p, which span many decades) with a
    correctly-labeled colorbar."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    plot_values = np.log10(np.maximum(values, 1e-300)) if log_scale else values
    tpc = ax.tripcolor(points_um[:, 0], points_um[:, 1], triangles, plot_values,
                        shading="gouraud", cmap=cmap)
    cbar = ax.figure.colorbar(tpc, ax=ax)
    if label:
        cbar.set_label(f"log10({label})" if log_scale else label)

    ax.set_xlim(points_um[:, 0].min(), points_um[:, 0].max())
    ax.set_ylim(points_um[:, 1].max(), points_um[:, 1].min())
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um), depth from top surface")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if own_fig:
        fig.tight_layout()
    return ax


FIELD_SPECS = {
    "psi": dict(cmap="RdBu_r", log_scale=False, label="psi (V)"),
    "n": dict(cmap="viridis", log_scale=True, label="n (cm^-3)"),
    "p": dict(cmap="magma", log_scale=True, label="p (cm^-3)"),
    "phin": dict(cmap="RdBu_r", log_scale=False, label="phin (V)"),
    "phip": dict(cmap="RdBu_r", log_scale=False, label="phip (V)"),
    "Ex": dict(cmap="RdBu_r", log_scale=False, label="Ex (V/cm)"),
    "Ey": dict(cmap="RdBu_r", log_scale=False, label="Ey (V/cm)"),
}


def interactive_field_viewer(doc):
    """Full interactive viewer: pick which field to see (structure-only, or
    any saved field - psi/n/p/phin/phip) and which saved bias point, with
    the same structure/mesh/boundary layer toggles as plot_structure2d, plus
    a slicing tool - click two points on the main plot to draw a cut line
    and see that field's profile along it in the side panel. Requires a
    real (non-Agg) backend, i.e. running with --interactive on the CLI
    below."""
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
    from scipy.interpolate import griddata

    _require_2d(doc, "interactive_field_viewer")
    x_um = np.array(doc["grid"]["x_um"])
    y_um = np.array(doc["grid"]["y_um"])
    points_um = np.stack([x_um, y_um], axis=1)
    triangles = np.array(doc["mesh2d"]["triangles"])
    bias_points = doc["bias_points"]
    field_names = [k for k in FIELD_SPECS if bias_points and k in bias_points[0]["fields"]]

    # Layout: big main view on the left (fixed rect - never resized after
    # creation, which is what made the structure "keep getting smaller" -
    # every fig.colorbar(tpc, ax=ax_main) call had been quietly shrinking
    # ax_main to make room for a new colorbar, even after removing the old
    # one). A dedicated, fixed-rect colorbar axes fixes that permanently.
    # Controls live in a compact column top-right; the field picker is a
    # single always-visible button that pops a floating option list open
    # ON TOP of whatever else is below it (a real dropdown, not a
    # permanently-expanded RadioButtons box) - the previous always-open
    # list ate enough fixed vertical space that it routinely collided with
    # the bias slider/layer checkboxes/gate-stack inset/slice panel below
    # it, especially once Ex/Ey brought the field count to 8.
    mesa_bbox = mesa_bbox_um(doc)

    fig = plt.figure(figsize=(15, 7))
    ax_main = fig.add_axes([0.06, 0.08, 0.52, 0.88])
    ax_cbar = fig.add_axes([0.60, 0.08, 0.015, 0.88])

    col_x, col_w = 0.74, 0.24
    field_btn_ax = fig.add_axes([col_x, 0.91, col_w, 0.05])
    slider_ax = fig.add_axes([col_x + 0.02, 0.855, col_w - 0.04, 0.025])
    check_ax = fig.add_axes([col_x, 0.74, col_w, 0.09])

    if mesa_bbox is not None:
        ax_inset = fig.add_axes([col_x, 0.40, col_w, 0.28])
        ax_cut = fig.add_axes([col_x, 0.08, col_w, 0.28])
    else:
        ax_inset = None
        ax_cut = fig.add_axes([col_x, 0.08, col_w, 0.60])

    field_options = ["structure"] + field_names
    n_fields = len(field_options)
    menu_height = min(0.30, 0.045 * n_fields)
    menu_ax = fig.add_axes([col_x, 0.91 - menu_height, col_w, menu_height])
    menu_ax.set_zorder(10)  # draws on top of check_ax/ax_inset/ax_cut while open
    menu_ax.patch.set_edgecolor("black")
    menu_ax.patch.set_linewidth(1.0)
    menu_ax.set_visible(False)

    state = {"field": "structure", "bias_idx": len(bias_points) // 2 if bias_points else 0,
             "cut_pts": [], "show_mesh": True, "show_boundary": True}

    def field_values():
        if state["field"] == "structure" or not bias_points:
            return None
        vals = bias_points[state["bias_idx"]]["fields"][state["field"]]
        return np.array([v if v is not None else np.nan for v in vals], dtype=float)

    def _draw_into(ax, xlim=None, ylim=None, aspect="equal"):
        """Renders the current field/structure state into `ax` - shared by
        the main view and the mesa zoomed inset so the two never drift out
        of sync with each other."""
        values = field_values()
        if values is not None:
            spec = FIELD_SPECS[state["field"]]
            plot_values = np.log10(np.maximum(values, 1e-300)) if spec["log_scale"] else values
            tpc = ax.tripcolor(x_um, y_um, triangles, plot_values, shading="gouraud",
                                cmap=spec["cmap"])
            if state["show_mesh"]:
                plot_structure2d(doc, ax=ax, show_mesh=True, show_boundary=False, label_regions=False)
            Va = bias_points[state["bias_idx"]]["bias"]
            ax.set_title(f"{state['field']} @ Va={Va:+.3f}V", fontsize=9)
        else:
            tpc = None
            plot_structure2d(doc, ax=ax, show_mesh=state["show_mesh"],
                              show_boundary=state["show_boundary"], label_regions=(ax is ax_main))
        ax.set_xlabel("x (um)", fontsize=8)
        ax.set_ylabel("y (um)", fontsize=8)
        ax.set_aspect(aspect)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        return tpc

    def redraw_main():
        xlim, ylim = ax_main.get_xlim(), ax_main.get_ylim()
        had_view = ax_main.has_data()
        ax_main.clear()
        ax_cbar.clear()
        tpc = _draw_into(ax_main)
        if tpc is not None:
            cbar = fig.colorbar(tpc, cax=ax_cbar)
            spec = FIELD_SPECS[state["field"]]
            cbar.set_label(f"log10({spec['label']})" if spec["log_scale"] else spec["label"])
        else:
            ax_cbar.axis("off")

        if had_view:
            ax_main.set_xlim(xlim)
            ax_main.set_ylim(ylim)
        else:
            ax_main.set_xlim(x_um.min(), x_um.max())
            ax_main.set_ylim(y_um.max(), y_um.min())
        if len(state["cut_pts"]) == 2:
            (x0, y0), (x1, y1) = state["cut_pts"]
            ax_main.plot([x0, x1], [y0, y1], "k--", linewidth=1.5, marker="x")

        if ax_inset is not None:
            ax_inset.clear()
            x0, x1, y0, y1 = mesa_bbox
            # aspect="auto" (not "equal") is what makes the mesa visible at
            # all: a real oxide is routinely 100-1000x thinner than the
            # gate footprint is wide, so an equal-aspect view of it is a
            # hairline. Letting the inset's y-axis stretch to fill a
            # roughly square panel is the same "vertically exaggerated,
            # not to scale" convention real device cross-section diagrams
            # use for a thin gate stack.
            _draw_into(ax_inset, xlim=(x0, x1), ylim=(y1, y0), aspect="auto")
            ax_inset.set_title("gate stack (y exaggerated, not to scale)", fontsize=8)

        fig.canvas.draw_idle()

    def redraw_cut():
        ax_cut.clear()
        if len(state["cut_pts"]) == 2 and state["field"] != "structure" and bias_points:
            (x0, y0), (x1, y1) = state["cut_pts"]
            s = np.linspace(0, 1, 200)
            cut_x = x0 + s * (x1 - x0)
            cut_y = y0 + s * (y1 - y0)
            dist_um = s * np.hypot(x1 - x0, y1 - y0)
            values = field_values()
            spec = FIELD_SPECS[state["field"]]
            cut_vals = griddata(points_um, values, np.stack([cut_x, cut_y], axis=1), method="linear")
            if spec["log_scale"]:
                ax_cut.semilogy(dist_um, np.maximum(cut_vals, 1e-300))
            else:
                ax_cut.plot(dist_um, cut_vals)
            ax_cut.set_xlabel("distance along cut (um)")
            ax_cut.set_ylabel(spec["label"])
            ax_cut.set_title("Slice")
            ax_cut.grid(alpha=0.3)
        else:
            ax_cut.set_title("Shift-click two points\non the left plot to slice", fontsize=9)
        fig.canvas.draw_idle()

    def on_click(event):
        # Plain click/drag is left free for the toolbar's own pan/zoom;
        # shift-click marks a slice endpoint instead, so the two don't
        # fight over the same gesture.
        if event.inaxes != ax_main or not event.key or "shift" not in event.key:
            return
        state["cut_pts"].append((event.xdata, event.ydata))
        if len(state["cut_pts"]) > 2:
            state["cut_pts"] = [state["cut_pts"][-1]]
        redraw_main()
        redraw_cut()

    def on_scroll(event):
        # Scroll-wheel zoom centered on the cursor, since it's not always
        # obvious the toolbar's own zoom-rectangle button (top of window)
        # is there to click first.
        if event.inaxes != ax_main:
            return
        scale = 0.85 if event.button == "up" else 1 / 0.85
        xlim, ylim = ax_main.get_xlim(), ax_main.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        ax_main.set_xlim(xdata - (xdata - xlim[0]) * scale, xdata + (xlim[1] - xdata) * scale)
        ax_main.set_ylim(ydata - (ydata - ylim[0]) * scale, ydata + (ylim[1] - ydata) * scale)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    field_btn = Button(field_btn_ax, f"field: structure ▾")
    radio = RadioButtons(menu_ax, field_options, active=0)
    for label in radio.labels:
        label.set_fontsize(8)

    def toggle_menu(event):
        menu_ax.set_visible(not menu_ax.get_visible())
        fig.canvas.draw_idle()

    field_btn.on_clicked(toggle_menu)

    def on_field(label):
        state["field"] = label
        field_btn.label.set_text(f"field: {label} ▾")
        menu_ax.set_visible(False)
        redraw_main()
        redraw_cut()

    radio.on_clicked(on_field)

    check_ax.set_title("layers", fontsize=9)
    check = CheckButtons(check_ax, ["mesh", "boundary"], [True, True])
    for label in check.labels:
        label.set_fontsize(8)

    def on_check(label):
        key = "show_mesh" if label == "mesh" else "show_boundary"
        state[key] = not state[key]
        redraw_main()

    check.on_clicked(on_check)

    if len(bias_points) > 1:
        slider = Slider(slider_ax, "bias idx", 0, len(bias_points) - 1,
                         valinit=state["bias_idx"], valstep=1)

        def on_slider(val):
            state["bias_idx"] = int(val)
            redraw_main()
            redraw_cut()

        slider.on_changed(on_slider)
    else:
        slider_ax.axis("off")

    hint_y = 0.38 if mesa_bbox is not None else 0.70
    fig.text(col_x, hint_y, "Scroll to zoom, drag toolbar's pan tool to pan,\n"
                             "shift-click twice on the plot to slice.",
              fontsize=8, va="top")

    redraw_main()
    redraw_cut()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone plotter for gatorade_tcaddevice 2D structure+mesh JSON files.")
    parser.add_argument("structure_path", help="Path to a *_structure.json file (dim=2)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--interactive", action="store_true",
                         help="Write a standalone interactive-viewer HTML file (Plotly-based: "
                              "real dropdown field picker, bias slider, layer toggles, click-to-"
                              "slice) and open it in the default browser - see viz2d/plot2d_web.py. "
                              "Falls back to a plain structure-only view if the file has no "
                              "bias_points.")
    parser.add_argument("--mpl-interactive", action="store_true",
                         help="Use the older matplotlib-window viewer instead of the Plotly HTML "
                              "one (kept for reference/offline use with no browser available).")
    args = parser.parse_args()

    doc = sio.load_structure(args.structure_path)

    if args.mpl_interactive:
        if doc.get("bias_points"):
            interactive_field_viewer(doc)
        else:
            ax, layers = plot_structure2d(doc)
            _interactive_show(ax, layers)
        return

    if args.interactive:
        import webbrowser
        from viz2d.plot2d_web import build_interactive_html
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.structure_path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.structure_path))[0]
        html_path = os.path.join(out_dir, f"{stem}_viewer.html")
        build_interactive_html(doc, html_path)
        print(html_path)
        webbrowser.open("file://" + os.path.abspath(html_path))
        return

    ax, layers = plot_structure2d(doc)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.structure_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.structure_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_structure.png")
    ax.figure.savefig(out_path, dpi=150)
    print(out_path)


if __name__ == "__main__":
    main()
