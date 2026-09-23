"""A genuinely interactive 2D structure/field viewer built on Plotly.js
instead of matplotlib widgets - replaces viz2d/plot2d.py's
interactive_field_viewer, which hit real limits of matplotlib's widget
system (RadioButtons has no true dropdown, hidden-axes ghosting on some
backends, and every control's position had to be hand-placed in figure
fractions with no layout engine to keep them from overlapping).

This writes a SELF-CONTAINED .html file (Plotly.js loaded from a CDN,
everything else inlined) with a native HTML <select> for field choice, a
native <input type=range> slider for the bias point, real checkboxes for
the mesh/boundary layers, and a click-to-slice tool - all driven by a
small hand-written JS "app" at the bottom of the file rather than trying
to force Plotly's own (also fairly limited) updatemenu/slider widgets to
juggle two independent pieces of state (field+bias selection vs layer
toggles) at once.

Field data is pre-interpolated onto a regular grid (scipy.interpolate.griddata,
the same tool the old matplotlib slice tool used) once per (field, bias
point) at file-write time and embedded as JSON - Plotly's Heatmap trace
then renders it, and the same grid is resampled bilinearly IN JAVASCRIPT
for the slice tool, so opening the file needs nothing but a browser (no
running Python process, unlike the old matplotlib window).
"""
import json
import os

import numpy as np
from scipy.interpolate import griddata

REGION_COLORS = {"p": "#f4a259", "n": "#5fa8d3"}
INSULATOR_COLOR = "#cfcfcf"
BC_COLORS = {"symmetry": "#4caf50", "free_surface": "#9e9e9e"}
CONTACT_COLORS = ["#d62728", "#1f77b4", "#9467bd", "#8c564b"]

FIELD_SPECS = {
    # default_log: which scale (Linear/Log) the lin/log toggle starts on for
    # this field - n/p span many decades so log is the useful default;
    # everything else (potential/field components, which are signed) starts
    # linear. The toggle itself is NOT restricted by this - the user can
    # switch either way for any field, viewing log10(|value|) when in log
    # mode (sign is lost, same convention the 1D I-V semilog plots use).
    "psi": dict(colorscale="RdBu", default_log=False, label="psi (V)"),
    "n": dict(colorscale="Viridis", default_log=True, label="n (cm^-3)"),
    "p": dict(colorscale="Magma", default_log=True, label="p (cm^-3)"),
    "phin": dict(colorscale="RdBu", default_log=False, label="phin (V)"),
    "phip": dict(colorscale="RdBu", default_log=False, label="phip (V)"),
    "Ex": dict(colorscale="RdBu", default_log=False, label="Ex (V/cm)"),
    "Ey": dict(colorscale="RdBu", default_log=False, label="Ey (V/cm)"),
}


def _region_color(region):
    if region.get("kind") == "insulator":
        return INSULATOR_COLOR
    return REGION_COLORS.get(region.get("doping_type"), "#bbbbbb")


def _mesa_bbox(regions, margin_x_frac=0.15, margin_y_mult=3.0):
    mesa_regions = [r for r in regions if r["y_range_um"][0] < 0]
    if not mesa_regions:
        return None
    x0 = min(r["x_range_um"][0] for r in mesa_regions)
    x1 = max(r["x_range_um"][1] for r in mesa_regions)
    y0 = min(r["y_range_um"][0] for r in mesa_regions)
    width = x1 - x0
    height = -y0
    return [x0 - margin_x_frac * width, x1 + margin_x_frac * width,
            y0 - margin_y_mult * height, margin_y_mult * height]


def _grid_field(x_um, y_um, values, xlim, ylim, nx, ny):
    """Interpolates the RAW (linear) field values onto a regular grid - no
    log transform here. The lin/log toggle is applied client-side in JS
    instead (see transformGrid() in the HTML template), so switching scale
    doesn't need a second copy of the data or a page reload."""
    gx = np.linspace(xlim[0], xlim[1], nx)
    gy = np.linspace(ylim[0], ylim[1], ny)
    GX, GY = np.meshgrid(gx, gy)
    Z = griddata(np.stack([x_um, y_um], axis=1), values, (GX, GY), method="linear")
    Z_list = [[None if not np.isfinite(v) else round(float(v), 6) for v in row] for row in Z]
    return {"x": gx.round(6).tolist(), "y": gy.round(6).tolist(), "z": Z_list}


def build_interactive_html(doc, out_path, grid_res=(160, 110), inset_grid_res=(120, 120)):
    """Writes a standalone interactive-viewer HTML file for `doc` (a loaded
    dim=2 structure, see core/structure_io.py) to out_path. Returns out_path."""
    if doc["dim"] != 2:
        raise NotImplementedError("build_interactive_html: only dim=2 structures are supported")

    x_um = np.array(doc["grid"]["x_um"], dtype=float)
    y_um = np.array(doc["grid"]["y_um"], dtype=float)
    regions = doc["regions"]
    mesh2d = doc.get("mesh2d", {})
    bias_points = doc.get("bias_points", [])
    triangles = mesh2d.get("triangles", [])

    xlim = (float(x_um.min()), float(x_um.max()))
    ylim = (float(y_um.min()), float(y_um.max()))
    mesa_bbox = _mesa_bbox(regions)

    field_names = [k for k in FIELD_SPECS if bias_points and k in bias_points[0]["fields"]]

    # --- Mesh edges (one line trace, None-separated segments) ---
    seen = set()
    mesh_x, mesh_y = [], []
    for (i, j, k) in triangles:
        for a, b in ((i, j), (j, k), (k, i)):
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            mesh_x += [x_um[a], x_um[b], None]
            mesh_y += [y_um[a], y_um[b], None]

    # --- Boundary points, grouped by bc_type ---
    boundary = mesh2d.get("boundary", [])
    contact_names = sorted({b["bc_type"].split(":", 1)[1] for b in boundary
                             if b["bc_type"].startswith("contact:")})
    by_type = {}
    for b in boundary:
        by_type.setdefault(b["bc_type"], []).append(b["point_index"])
    boundary_traces = []
    for bc_type, idxs in by_type.items():
        if bc_type.startswith("contact:"):
            name = bc_type.split(":", 1)[1]
            color = CONTACT_COLORS[contact_names.index(name) % len(CONTACT_COLORS)]
            label = f"contact: {name}"
        else:
            color = BC_COLORS.get(bc_type, "#000000")
            label = bc_type
        boundary_traces.append({
            "x": x_um[idxs].round(6).tolist(), "y": y_um[idxs].round(6).tolist(),
            "color": color, "label": label,
        })

    region_shapes = [{
        "x0": r["x_range_um"][0], "x1": r["x_range_um"][1],
        "y0": r["y_range_um"][0], "y1": r["y_range_um"][1],
        "color": _region_color(r), "name": r["name"],
    } for r in regions]

    # --- Pre-interpolated field grids, one per (field, bias point) ---
    nx, ny = grid_res
    field_data = {}
    for field in field_names:
        spec = FIELD_SPECS[field]
        per_bias = []
        for bp in bias_points:
            vals = np.array([v if v is not None else np.nan for v in bp["fields"][field]], dtype=float)
            grid = _grid_field(x_um, y_um, vals, xlim, ylim, nx, ny)
            if mesa_bbox is not None:
                grid["inset"] = _grid_field(x_um, y_um, vals, tuple(mesa_bbox[0:2]), tuple(mesa_bbox[2:4]),
                                             inset_grid_res[0], inset_grid_res[1])
            per_bias.append(grid)
        field_data[field] = {"label": spec["label"], "default_log": spec["default_log"],
                              "colorscale": spec["colorscale"], "bias": per_bias}

    bias_labels = [bp["label"] for bp in bias_points]

    payload = {
        "device": doc.get("device", "device"),
        "xlim": list(xlim), "ylim": list(ylim), "mesa_bbox": mesa_bbox,
        "mesh": {"x": mesh_x, "y": mesh_y},
        "boundary": boundary_traces,
        "regions": region_shapes,
        "field_names": field_names,
        "field_data": field_data,
        "bias_labels": bias_labels,
    }

    html = _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


_HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>gatorade_tcaddevice 2D viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; display: flex; height: 100vh; }
  #controls { width: 300px; padding: 16px; box-sizing: border-box; border-right: 1px solid #ddd;
              overflow-y: auto; background: #fafafa; }
  #controls h3 { margin: 0 0 4px 0; font-size: 13px; color: #333; }
  #controls label { display: block; font-size: 12px; color: #555; margin: 14px 0 4px; }
  #controls select, #controls input[type=range] { width: 100%; }
  #controls .row { display: flex; align-items: center; gap: 6px; font-size: 13px; margin: 4px 0; }
  #biasLabel { font-size: 12px; color: #333; margin-top: 2px; }
  #main { flex: 1; display: flex; flex-direction: column; padding: 8px; min-width: 0; }
  #plots { flex: 1; display: flex; gap: 8px; min-height: 0; }
  #mainplot { flex: 2; min-width: 0; }
  #sidecol { flex: 1; display: flex; flex-direction: column; gap: 8px; min-width: 260px; }
  #insetplot, #sliceplot { flex: 1; min-height: 0; }
  .hint { font-size: 11px; color: #777; margin-top: 10px; line-height: 1.4; }
</style>
</head>
<body>
<div id="controls">
  <h3>gatorade_tcaddevice 2D viewer</h3>
  <label for="fieldSelect">field</label>
  <select id="fieldSelect"></select>

  <label for="biasSlider">bias point</label>
  <input type="range" id="biasSlider" min="0" max="0" step="1" value="0">
  <div id="biasLabel"></div>

  <label>layers</label>
  <div class="row"><input type="checkbox" id="meshToggle" checked> <label for="meshToggle" style="margin:0">mesh</label></div>
  <div class="row"><input type="checkbox" id="boundaryToggle" checked> <label for="boundaryToggle" style="margin:0">boundary</label></div>

  <label>scale (structure field &amp; slice)</label>
  <div class="row"><input type="radio" name="scale" id="scaleLin" checked> <label for="scaleLin" style="margin:0">Linear</label></div>
  <div class="row"><input type="radio" name="scale" id="scaleLog"> <label for="scaleLog" style="margin:0">Log (log10 of |value|)</label></div>

  <div id="fieldNote" class="hint"></div>

  <div class="hint">Click two points on the main plot (left) to draw a slice line - click-drag
  no longer zooms (it was eating the clicks), so scroll to zoom and use the toolbar's pan (hand)
  icon if you need to pan instead. The autoscale/home icon in the toolbar resets the view.</div>
</div>
<div id="main">
  <div id="plots">
    <div id="mainplot"></div>
    <div id="sidecol">
      <div id="insetplot"></div>
      <div id="sliceplot"></div>
    </div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;

const state = { field: "structure", biasIdx: Math.floor((DATA.bias_labels.length - 1) / 2),
                showMesh: true, showBoundary: true, cutPts: [], logScale: false };

const fieldSelect = document.getElementById("fieldSelect");
const opt0 = document.createElement("option");
opt0.value = "structure"; opt0.textContent = "structure"; fieldSelect.appendChild(opt0);
for (const f of DATA.field_names) {
  const opt = document.createElement("option");
  opt.value = f; opt.textContent = f; fieldSelect.appendChild(opt);
}
fieldSelect.value = state.field;

const biasSlider = document.getElementById("biasSlider");
biasSlider.max = Math.max(0, DATA.bias_labels.length - 1);
biasSlider.value = state.biasIdx;
document.getElementById("biasLabel").textContent = DATA.bias_labels.length
  ? DATA.bias_labels[state.biasIdx] : "(no bias points saved)";

function regionShapes() {
  return DATA.regions.map(r => ({
    type: "rect", x0: r.x0, x1: r.x1, y0: r.y0, y1: r.y1,
    line: { color: "black", width: 1 }, fillcolor: r.color, opacity: 0.6, layer: "below",
  }));
}

function meshTrace() {
  return { x: DATA.mesh.x, y: DATA.mesh.y, mode: "lines", type: "scattergl",
           line: { color: "#c9c9c9", width: 0.6 }, hoverinfo: "skip", showlegend: false,
           visible: state.showMesh };
}

function boundaryTraces() {
  return DATA.boundary.map(b => ({
    x: b.x, y: b.y, mode: "markers", type: "scattergl", name: b.label,
    marker: { color: b.color, size: 5 }, visible: state.showBoundary,
  }));
}

function cutLineTrace() {
  if (state.cutPts.length !== 2) return null;
  return { x: [state.cutPts[0][0], state.cutPts[1][0]], y: [state.cutPts[0][1], state.cutPts[1][1]],
           mode: "lines+markers", type: "scatter", line: { color: "black", dash: "dash" },
           marker: { symbol: "x", size: 8, color: "black" }, showlegend: false };
}

function currentFieldGrid(forInset) {
  if (state.field === "structure") return null;
  const g = DATA.field_data[state.field].bias[state.biasIdx];
  return forInset ? g.inset : g;
}

// Log mode plots log10(|value|) (sign is lost - same convention the 1D
// I-V semilog plots already use elsewhere in this project) rather than
// needing a second precomputed copy of every grid; applied client-side so
// switching scale is instant and doesn't touch the underlying data.
function transformValue(v) {
  if (v === null || v === undefined) return null;
  if (!state.logScale) return v;
  const av = Math.abs(v);
  return av > 0 ? Math.log10(av) : null;
}

function transformGrid(grid) {
  const z = grid.z.map(row => row.map(transformValue));
  let zmin = Infinity, zmax = -Infinity;
  for (const row of z) for (const v of row) {
    if (v !== null) { if (v < zmin) zmin = v; if (v > zmax) zmax = v; }
  }
  if (!isFinite(zmin)) { zmin = 0; zmax = 1; }
  return { x: grid.x, y: grid.y, z, zmin, zmax };
}

function fieldLabel() {
  const label = DATA.field_data[state.field].label;
  return state.logScale ? "log10(|" + label + "|)" : label;
}

// Plotly's default tick formatting abbreviates large numbers with SI
// prefixes (1.2e5 shows as "120k", 3.4e6 as "3.4M") - fine for everyday
// plots, but wrong for device-physics quantities like Ex/Ey (up to ~1e6
// V/cm) which should read in proper scientific notation. Only applied in
// Linear mode; Log mode's values are already small (roughly -6 to 6, being
// log10 of the real quantity) so plain formatting is correct there.
const SCI_TICKFORMAT = ".2e";
function valueTickformat() { return state.logScale ? undefined : SCI_TICKFORMAT; }

// doubleClick: false - a quick double-click while trying to place the two
// slice points otherwise triggers Plotly's built-in "reset to autorange"
// zoom, which looks like an unwanted zoom jump right when clicking to
// slice. scrollZoom stays on as the one deliberate way to zoom (plus the
// toolbar's zoom/pan buttons); dragmode:false (set per-plot below) is what
// stops a plain click-drag from opening a zoom-box instead of registering
// as a slice point.
const PLOT_CONFIG = { responsive: true, scrollZoom: true, doubleClick: false, displaylogo: false };

// x increases rightward, y increases DOWNWARD (y=0 is the top/free surface,
// larger y is deeper into the device - see mesh2d/geometry2d.py's own
// domain convention) - spelled out in words (not an arrow glyph: Plotly
// rotates the y-axis title text 90 degrees, which rotates a "↓" glyph right
// along with it into something that reads as sideways, not down) since the
// downward-y convention is the opposite of the usual "up is positive"
// intuition and matters for reading Ex/Ey's sign correctly.
const X_AXIS_TITLE = "x (um), increases rightward";
const Y_AXIS_TITLE = "y (um), increases downward into device";

// Every redraw (a click to mark a slice point, toggling a layer, switching
// field/bias/scale) calls Plotly.react with a FRESH layout object - if that
// layout hard-codes the axis range back to the full domain every time, any
// zoom/pan the user had already done gets silently discarded on the very
// next click, which looks exactly like "clicking makes it zoom in/out by
// itself". Fixed by remembering the plot's own current range (captured via
// the plotly_relayout event, which fires for scroll-zoom, toolbar zoom/pan,
// AND the autoscale/home button alike) and reusing it on every subsequent
// render instead of the original full-domain default.
let mainRange = null;
let insetRange = null;

function baseLayout(xlim, ylim, title) {
  const xr = mainRange ? mainRange.x : xlim;
  const yr = mainRange ? mainRange.y : [ylim[1], ylim[0]];
  return {
    margin: { l: 55, r: 10, t: 30, b: 40 },
    dragmode: false,  // plain click now always fires plotly_click for slicing instead of
                        // starting a zoom-box drag - scroll or the toolbar's zoom/pan tools
                        // still work, they just aren't the default left-click gesture anymore
    xaxis: { title: { text: X_AXIS_TITLE, font: { size: 12 } }, range: xr, tickfont: { size: 10 } },
    yaxis: { title: { text: Y_AXIS_TITLE, font: { size: 12 } },
             range: yr, scaleanchor: "x", scaleratio: 1, tickfont: { size: 10 } },
    title: { text: title, font: { size: 12 } },
    shapes: state.field === "structure" ? regionShapes() : [],
  };
}

function updateFieldNote() {
  const notes = {
    Ex: "Ex = -dψ/dx: positive Ex points in +x (rightward).",
    Ey: "Ey = -dψ/dy: positive Ey points in +y (downward, into the device - NOT upward).",
    psi: "ψ: electrostatic potential (V).",
  };
  document.getElementById("fieldNote").textContent = notes[state.field] || "";
}

function renderMain() {
  const traces = [];
  const rawGrid = currentFieldGrid(false);
  const grid = rawGrid ? transformGrid(rawGrid) : null;
  if (grid) {
    traces.push({ x: grid.x, y: grid.y, z: grid.z, type: "heatmap", zmin: grid.zmin, zmax: grid.zmax,
                  colorscale: DATA.field_data[state.field].colorscale,
                  colorbar: { title: fieldLabel(), titlefont: {size: 11}, tickformat: valueTickformat() },
                  hovertemplate: "x=%{x:.3f}<br>y=%{y:.3f}<br>value=%{z:.4g}<extra></extra>" });
  }
  traces.push(meshTrace());
  for (const t of boundaryTraces()) traces.push(t);
  const cl = cutLineTrace();
  if (cl) traces.push(cl);

  const title = grid ? (state.field + " @ " + (DATA.bias_labels[state.biasIdx] || "")) : (DATA.device + " structure");
  const layout = baseLayout(DATA.xlim, DATA.ylim, title);
  Plotly.react("mainplot", traces, layout, PLOT_CONFIG);
  updateFieldNote();
}

function renderInset() {
  const el = document.getElementById("insetplot");
  if (!DATA.mesa_bbox) { el.style.display = "none"; return; }
  el.style.display = "";
  const traces = [];
  const rawGrid = currentFieldGrid(true);
  const grid = rawGrid ? transformGrid(rawGrid) : null;
  if (grid) {
    traces.push({ x: grid.x, y: grid.y, z: grid.z, type: "heatmap", zmin: grid.zmin, zmax: grid.zmax,
                  colorscale: DATA.field_data[state.field].colorscale, showscale: false,
                  hovertemplate: "x=%{x:.4f}<br>y=%{y:.5f}<br>value=%{z:.4g}<extra></extra>" });
  }
  if (state.showMesh) traces.push(meshTrace());
  const bx0 = DATA.mesa_bbox[0], bx1 = DATA.mesa_bbox[1], by0 = DATA.mesa_bbox[2], by1 = DATA.mesa_bbox[3];
  const xr = insetRange ? insetRange.x : [bx0, bx1];
  const yr = insetRange ? insetRange.y : [by1, by0];
  const layout = {
    margin: { l: 55, r: 10, t: 30, b: 40 },
    dragmode: false,
    xaxis: { title: { text: X_AXIS_TITLE, font: { size: 11 } }, range: xr, tickfont: { size: 9 } },
    yaxis: { title: { text: Y_AXIS_TITLE, font: { size: 11 } }, range: yr, tickfont: { size: 9 } },
    // aspect NOT locked here (unlike the main plot) - exaggerates the thin
    // mesa on purpose, see the module docstring.
    title: { text: "gate stack (y exaggerated, not to scale)", font: { size: 11 } },
    shapes: state.field === "structure" ? regionShapes() : [],
  };
  Plotly.react("insetplot", traces, layout, PLOT_CONFIG);
}

function bilinear(g, px, py) {
  const gx = g.x, gy = g.y, z = g.z;
  if (px < gx[0] || px > gx[gx.length - 1] || py < gy[0] || py > gy[gy.length - 1]) return null;
  let i = Math.floor((px - gx[0]) / (gx[1] - gx[0]));
  i = Math.min(Math.max(i, 0), gx.length - 2);
  let j = Math.floor((py - gy[0]) / (gy[1] - gy[0]));
  j = Math.min(Math.max(j, 0), gy.length - 2);
  const tx = (px - gx[i]) / (gx[i + 1] - gx[i]);
  const ty = (py - gy[j]) / (gy[j + 1] - gy[j]);
  const z00 = z[j][i], z10 = z[j][i + 1], z01 = z[j + 1][i], z11 = z[j + 1][i + 1];
  if (z00 === null || z10 === null || z01 === null || z11 === null) return null;
  return z00 * (1 - tx) * (1 - ty) + z10 * tx * (1 - ty) + z01 * (1 - tx) * ty + z11 * tx * ty;
}

function renderSlice() {
  const el = document.getElementById("sliceplot");
  if (state.cutPts.length !== 2 || state.field === "structure") {
    Plotly.react("sliceplot", [], { margin: { l: 45, r: 10, t: 30, b: 35 },
      title: { text: "shift-click ignored here - just click twice on the main plot to slice", font: { size: 10 } } });
    return;
  }
  const grid = currentFieldGrid(false);   // raw (linear) grid - bilinear interpolation happens
                                            // in linear space, THEN the log transform is applied
                                            // to the sampled scalar (not the other way around)
  const [x0, y0] = state.cutPts[0], [x1, y1] = state.cutPts[1];
  const n = 200;
  const dist = [], vals = [];
  const totalDist = Math.hypot(x1 - x0, y1 - y0);
  for (let k = 0; k < n; k++) {
    const s = k / (n - 1);
    const px = x0 + s * (x1 - x0), py = y0 + s * (y1 - y0);
    dist.push(s * totalDist);
    vals.push(transformValue(bilinear(grid, px, py)));
  }
  Plotly.react("sliceplot", [{ x: dist, y: vals, mode: "lines", type: "scatter" }], {
    margin: { l: 45, r: 10, t: 30, b: 35 },
    xaxis: { title: "distance along cut (um)" },
    yaxis: { title: fieldLabel(), tickformat: valueTickformat() },
    title: { text: "slice", font: { size: 12 } },
  });
}

function renderAll() { renderMain(); renderInset(); renderSlice(); }

const scaleLin = document.getElementById("scaleLin");
const scaleLog = document.getElementById("scaleLog");

fieldSelect.addEventListener("change", () => {
  state.field = fieldSelect.value;
  // Each field has its own sensible default (n/p start in Log, since they
  // span many decades; everything else starts Linear) - re-applied every
  // time the field changes rather than preserving whatever the user last
  // picked, so switching to n/p doesn't silently stay on a stale choice.
  const defaultLog = state.field !== "structure" && DATA.field_data[state.field].default_log;
  state.logScale = !!defaultLog;
  scaleLog.checked = state.logScale;
  scaleLin.checked = !state.logScale;
  renderAll();
});
biasSlider.addEventListener("input", () => {
  state.biasIdx = parseInt(biasSlider.value, 10);
  document.getElementById("biasLabel").textContent = DATA.bias_labels[state.biasIdx] || "";
  renderAll();
});
document.getElementById("meshToggle").addEventListener("change", (e) => { state.showMesh = e.target.checked; renderAll(); });
document.getElementById("boundaryToggle").addEventListener("change", (e) => { state.showBoundary = e.target.checked; renderMain(); });
scaleLin.addEventListener("change", () => { if (scaleLin.checked) { state.logScale = false; renderAll(); } });
scaleLog.addEventListener("change", () => { if (scaleLog.checked) { state.logScale = true; renderAll(); } });

renderAll();

document.getElementById("mainplot").on("plotly_click", (ev) => {
  if (!ev.points || !ev.points.length) return;
  const p = ev.points[0];
  state.cutPts.push([p.x, p.y]);
  if (state.cutPts.length > 2) state.cutPts = [state.cutPts[state.cutPts.length - 1]];
  renderMain();
  renderSlice();
});

// Remember the user's current zoom/pan (scroll-zoom, toolbar zoom/pan, or
// the autoscale/home button all fire this) so the NEXT render (triggered
// by a slice click, a layer toggle, a field/bias change, ...) reuses it
// instead of snapping back to the full-domain view - see baseLayout()'s
// comment for why that reset was exactly what made a plain click look
// like it was randomly zooming the plot in or out.
document.getElementById("mainplot").on("plotly_relayout", () => {
  const gd = document.getElementById("mainplot");
  if (gd.layout && gd.layout.xaxis && gd.layout.yaxis
      && gd.layout.xaxis.range && gd.layout.yaxis.range) {
    mainRange = { x: gd.layout.xaxis.range.slice(), y: gd.layout.yaxis.range.slice() };
  }
});
if (DATA.mesa_bbox) {
  // Only attach if the inset was actually initialized as a Plotly graph
  // (renderInset() skips Plotly.react entirely when there's no mesa) -
  // .on() doesn't exist on a plain, never-plotted div.
  document.getElementById("insetplot").on("plotly_relayout", () => {
    const gd = document.getElementById("insetplot");
    if (gd.layout && gd.layout.xaxis && gd.layout.yaxis
        && gd.layout.xaxis.range && gd.layout.yaxis.range) {
      insetRange = { x: gd.layout.xaxis.range.slice(), y: gd.layout.yaxis.range.slice() };
    }
  });
}
</script>
</body>
</html>
"""
