"""Data side of tcadview, with no Qt: open .vtu/.pvd results written by
solver2d/vtk_export.py, split them into regions, scale fields, probe points
and sample cut lines. The window (tcadview/app.py) only displays what this
module computes, so all of it is testable headless.

A Session is one opened file: a list of states (one per bias point; a
single .vtu is a one-state session). States load lazily and are cached.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from core import vtk_io

SCALES = ("linear", "log", "signed log")

# fields whose natural scale is logarithmic (Sentaurus names, as written by
# solver2d/vtk_export.py)
LOG_FIELDS = {"eDensity", "hDensity", "AbsElectricField"}
SIGNED_LOG_FIELDS = {"DopingConcentration", "SpaceCharge"}
HIDDEN_FIELDS = {"region_id", "material_id", "contact_id", "vtkOriginalPointIds", "vtkOriginalCellIds",
                 "_display"}


def default_scale(name):
    if name in LOG_FIELDS:
        return "log"
    if name in SIGNED_LOG_FIELDS:
        return "signed log"
    return "linear"


def scale_values(a, scale):
    """Values as displayed. log: log10|a| (zeros -> NaN); signed log:
    sign(a) log10(1 + |a|), which is continuous through 0 and ~ log10|a| for
    the doping/charge magnitudes TCAD deals in."""
    a = np.asarray(a, dtype=float)
    if scale == "linear":
        return a
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(a) > 0, np.log10(np.abs(a)), np.nan)
    if scale == "signed log":
        return np.sign(a) * np.log10(1.0 + np.abs(a))
    raise ValueError(f"unknown scale {scale!r}")


def scale_label(name, scale, unit=""):
    u = f" ({unit})" if unit else ""
    if scale == "log":
        return f"log10 |{name}|{u}"
    if scale == "signed log":
        return f"slog10 {name}{u}"
    return f"{name}{u}"


@dataclass
class StateInfo:
    bias: float
    path: str
    name: str


@dataclass
class State:
    """One loaded state: the full grid, its regions, and dataset metadata."""
    grid: pv.UnstructuredGrid
    region_names: list
    material_names: list
    contact_names: list
    contact_bias: dict
    label: str
    units: dict
    _regions: dict = field(default_factory=dict)

    @property
    def dim(self):
        """2 for a flat (z = 0) mesh, 3 otherwise."""
        b = self.grid.bounds
        return 2 if b[5] - b[4] < 1e-12 * max(1.0, b[1] - b[0], b[3] - b[2]) else 3

    def region(self, k):
        """Cells of region k as their own grid (cached)."""
        if k not in self._regions:
            rid = self.grid.cell_data["region_id"]
            self._regions[k] = self.grid.extract_cells(np.flatnonzero(rid == k))
        return self._regions[k]

    def region_material(self, k):
        rid = self.grid.cell_data["region_id"]
        mid = self.grid.cell_data["material_id"][rid == k]
        return self.material_names[int(mid[0])] if len(mid) and mid[0] >= 0 else ""

    def fields(self):
        """Displayable field names: point fields first, then cell fields."""
        pd = [n for n in self.grid.point_data.keys() if n not in HIDDEN_FIELDS]
        cd = [n for n in self.grid.cell_data.keys() if n not in HIDDEN_FIELDS and n not in pd]
        return pd + cd

    def association(self, name):
        return "point" if name in self.grid.point_data.keys() else "cell"

    def values(self, name, grid=None):
        """Scalar values of a field on `grid` (default the full grid); a
        vector field gives its magnitude."""
        g = self.grid if grid is None else grid
        a = g.point_data[name] if name in g.point_data.keys() else g.cell_data[name]
        a = np.asarray(a, dtype=float)
        return np.linalg.norm(a, axis=1) if a.ndim == 2 else a

    def range(self, name, scale, regions=None):
        """Finite (min, max) of the displayed values over `regions` (all if
        None); (0, 1) if there is nothing finite."""
        ks = range(len(self.region_names)) if regions is None else regions
        vals = [scale_values(self.values(name, self.region(k)), scale) for k in ks]
        v = np.concatenate([x[np.isfinite(x)] for x in vals]) if vals else np.zeros(0)
        if not len(v):
            return 0.0, 1.0
        lo, hi = float(v.min()), float(v.max())
        return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)

    def probe(self, xyz):
        """All fields interpolated at point xyz (um). Returns
        {"region": name or None, "fields": {name: value}}; None outside the
        mesh."""
        pt = pv.PolyData(np.atleast_2d(np.asarray(xyz, dtype=float)))
        cid = self.grid.find_containing_cell(pt.points[0])
        if cid < 0:
            return None
        s = pt.sample(self.grid, tolerance=1e-9)
        out = {}
        for n in self.fields():
            if self.association(n) == "point" and n in s.point_data.keys():
                v = np.asarray(s.point_data[n][0], dtype=float)
            else:
                v = np.asarray(self.grid.cell_data[n][cid], dtype=float)
            out[n] = float(np.linalg.norm(v)) if v.ndim else float(v)
        rid = int(self.grid.cell_data["region_id"][cid])
        return {"region": self.region_names[rid] if rid >= 0 else None, "cell": int(cid), "fields": out}

    def cut_line(self, p0, p1, name, n=400):
        """Field sampled along the segment p0 -> p1 (um). Returns (distance
        um, values); NaN outside the mesh or where the field is undefined.
        Point fields are interpolated; cell fields take the containing
        cell's value (piecewise constant, as they are stored)."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        t = np.linspace(0.0, 1.0, n)
        pts = p0 + t[:, None] * (p1 - p0)
        d = t * np.linalg.norm(p1 - p0)
        cells = self.grid.find_containing_cell(pts)
        inside = cells >= 0
        if self.association(name) == "point":
            s = pv.PolyData(pts).sample(self.grid, tolerance=1e-9)
            a = np.asarray(s.point_data[name], dtype=float)
            v = np.linalg.norm(a, axis=1) if a.ndim == 2 else a
        else:
            v = np.full(n, np.nan)
            v[inside] = self.values(name)[cells[inside]]
        return d, np.where(inside, v, np.nan)


def load_state(path):
    """Read one .vtu written by solver2d/vtk_export.py."""
    grid = pv.read(path)
    fd = grid.field_data

    def strings(key):
        return [str(x) for x in fd[key]] if key in fd.keys() else []

    cnames = strings("contact_names")
    bias = np.asarray(fd["contact_bias"], dtype=float) if "contact_bias" in fd.keys() else None
    units = dict(u.split(":", 1) for u in strings("field_units") if ":" in u)
    return State(grid=grid, region_names=strings("region_names"), material_names=strings("material_names"),
                 contact_names=cnames,
                 contact_bias=dict(zip(cnames, bias.tolist())) if bias is not None else {},
                 label=(strings("label") or [""])[0], units=units)


class Session:
    """An opened .pvd (sweep) or .vtu (single state)."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pvd":
            self.states = [StateInfo(b, f, n or os.path.basename(f)) for b, f, n in vtk_io.read_pvd(path)]
        elif ext == ".vtu":
            self.states = [StateInfo(0.0, self.path, os.path.basename(path))]
        else:
            raise ValueError(f"{path}: expected a .pvd or .vtu file")
        if not self.states:
            raise ValueError(f"{path}: no datasets")
        self._cache = {}

    def __len__(self):
        return len(self.states)

    def state(self, i):
        if i not in self._cache:
            if len(self._cache) > 32:           # bound memory on long 3D sweeps
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = load_state(self.states[i].path)
        return self._cache[i]
