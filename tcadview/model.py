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
# positive fields that vary exponentially between nodes (Scharfetter-Gummel):
# probes and cut lines interpolate log10 of them, as the colour map does, so
# a coarse mesh gives a smooth exponential profile, not linear cliffs
LOG_INTERP_FIELDS = {"eDensity", "hDensity"}
HIDDEN_FIELDS = {"region_id", "material_id", "contact_id", "vtkOriginalPointIds", "vtkOriginalCellIds",
                 "_display", "_log_eDensity", "_log_hDensity"}


# process-step snapshots (fab/export.py): regions are stored as level sets
# ls_<region>; "Material" colours each region by its material instead of a field
LEVELSET_PREFIX = "ls_"
MATERIAL_FIELD = "Material"
MATERIAL_COLORS = {"Si": "#9fb4cc", "SiO2": "#f4b8c8", "Poly": "#c0504d", "SiN": "#86c48a", "Metal": "#e2b33c",
                   "SiGe": "#b393d6", "Ge": "#6f8fb5", "HfO2": "#f2a25c", "TiN": "#8c8c8c", "W": "#a0a0b0"}
_EXTRA_COLORS = ("#7fc7c2", "#d98cb3", "#c9b27c", "#8fa3d9", "#b5c96f")


def material_color(name):
    if name in MATERIAL_COLORS:
        return MATERIAL_COLORS[name]
    if name.startswith("SiGe"):
        return MATERIAL_COLORS["SiGe"]
    return _EXTRA_COLORS[sum(map(ord, name)) % len(_EXTRA_COLORS)]


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
    kind: str = ""
    _regions: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    @property
    def is_process(self):
        """A process-step snapshot (voxel image with level sets)."""
        return self.kind == "process"

    def levelset(self, k):
        return LEVELSET_PREFIX + self.region_names[k]

    def _contour_grid(self):
        """The image with every level set forced 'outside' on the domain walls,
        so a region touching a wall contours to a closed surface (cached)."""
        if "contour_grid" not in self._cache:
            g = self.grid.copy(deep=False)
            dims = g.dimensions
            for k in range(len(self.region_names)):
                name = self.levelset(k)
                if name not in g.point_data.keys():
                    continue
                a = np.array(g.point_data[name], dtype=np.float32).reshape(dims[::-1])   # [k, j, i]
                for ax in range(3):
                    sl = [slice(None)] * 3
                    for end in (0, -1):
                        sl[ax] = end
                        a[tuple(sl)] = np.maximum(a[tuple(sl)], 0.5)
                g.point_data[name] = a.ravel()
            self._cache["contour_grid"] = g
        return self._cache["contour_grid"]

    def surface(self, k):
        """Region k's smooth closed boundary surface (process snapshots;
        cached), with the point fields interpolated onto it."""
        key = ("surface", k)
        if key not in self._cache:
            s = self._contour_grid().contour([-0.5], scalars=self.levelset(k))
            self._cache[key] = s if s.n_points else None
        return self._cache[key]

    def voxel_surface(self, k):
        """Region k's boundary exactly as the device sees it: the voxel
        faces between its cells (region_id) and any other region or gas
        (process snapshots; cached). Sharp edges, and the face two regions
        share is the same face in both - no bevels, seams or gaps. Point
        fields are taken at the grid nodes, not interpolated."""
        key = ("voxel_surface", k)
        if key not in self._cache:
            self._cache[key] = _voxel_boundary(self.grid, k)
        return self._cache[key]

    def voxel_slice(self, k, axis, pos):
        """Region k on the plane normal to `axis` at `pos`, cut along its
        voxel cells (process snapshots), or None."""
        key = ("slice", axis, round(float(pos), 12))
        if key not in self._cache:
            o = [0.0, 0.0, 0.0]
            o["xyz".index(axis)] = pos
            self._cache[key] = self.grid.slice(normal=axis, origin=o)
        sl = self._cache[key]
        if not sl.n_cells or "region_id" not in sl.cell_data.keys():
            return None
        ids = np.flatnonzero(np.asarray(sl.cell_data["region_id"]) == k)
        return sl.extract_cells(ids) if len(ids) else None

    def slice_region(self, k, axis, pos):
        """Region k on the plane normal to `axis` at `pos`, cut at its smooth
        boundary (process snapshots), or None."""
        key = ("slice", axis, round(float(pos), 12))
        if key not in self._cache:
            o = [0.0, 0.0, 0.0]
            o["xyz".index(axis)] = pos
            self._cache[key] = self.grid.slice(normal=axis, origin=o)
        sl = self._cache[key]
        if not sl.n_points:
            return None
        c = sl.clip_scalar(scalars=self.levelset(k), value=-0.5, invert=True)
        return c if c.n_points else None

    def node_labels(self):
        """Region index per grid node from the level sets (-1 = gas)."""
        if "labels" not in self._cache:
            ls = np.stack([np.asarray(self.grid.point_data[self.levelset(k)], dtype=np.int16)
                           for k in range(len(self.region_names))])
            lab = np.argmin(ls, axis=0)
            self._cache["labels"] = np.where(ls.min(axis=0) < 0, lab, -1)
        return self._cache["labels"]

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

    def fields(self, data_only=False):
        """Displayable field names: point fields first, then cell fields
        (process snapshots: "Material" first, unless data_only)."""
        pd = [n for n in self.grid.point_data.keys()
              if n not in HIDDEN_FIELDS and not n.startswith(LEVELSET_PREFIX) and not n.startswith("_log_")]
        cd = [n for n in self.grid.cell_data.keys() if n not in HIDDEN_FIELDS and n not in pd]
        extra = [MATERIAL_FIELD] if self.is_process and not data_only else []
        return extra + pd + cd

    def association(self, name):
        if name == MATERIAL_FIELD:
            return "material"
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
        if name == MATERIAL_FIELD:
            return 0.0, 1.0
        if self.is_process:                    # no per-region cell extraction on big voxel images
            sel = np.isin(self.node_labels(), list(ks))
            vals = [scale_values(self.values(name)[sel], scale)]
        else:
            vals = [scale_values(self.values(name, self.region(k)), scale) for k in ks]
        v = np.concatenate([x[np.isfinite(x)] for x in vals]) if vals else np.zeros(0)
        if not len(v):
            return 0.0, 1.0
        lo, hi = float(v.min()), float(v.max())
        return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)

    def _sample(self, pts):
        """Point fields interpolated at points (N, 3): {name: values}. The
        LOG_INTERP_FIELDS are interpolated in log10 (exact at nodes)."""
        g = self.grid
        for n in LOG_INTERP_FIELDS & set(g.point_data.keys()):
            key = "_log_" + n
            if key not in g.point_data.keys():
                with np.errstate(divide="ignore", invalid="ignore"):
                    a = np.asarray(g.point_data[n], dtype=float)
                    g.point_data[key] = np.where(a > 0, np.log10(np.where(a > 0, a, 1.0)), np.nan)
        s = pv.PolyData(np.atleast_2d(pts)).sample(g, tolerance=1e-9)
        out = {}
        for n in s.point_data.keys():
            if n.startswith("_log_"):
                continue
            a = np.asarray(s.point_data[n], dtype=float)
            if "_log_" + n in s.point_data.keys():
                a = 10.0 ** np.asarray(s.point_data["_log_" + n], dtype=float)
            out[n] = a
        return out

    def probe(self, xyz):
        """All fields interpolated at point xyz (um). Returns
        {"region": name or None, "fields": {name: value}}; None outside the
        mesh."""
        pt = pv.PolyData(np.atleast_2d(np.asarray(xyz, dtype=float)))
        cid = self.grid.find_containing_cell(pt.points[0])
        if cid < 0:
            return None
        s = self._sample(pt.points)
        out = {}
        for n in self.fields(data_only=True):
            if self.association(n) == "point" and n in s:
                v = s[n][0]
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
            a = self._sample(pts)[name]
            v = np.linalg.norm(a, axis=1) if a.ndim == 2 else a
        else:
            v = np.full(n, np.nan)
            v[inside] = self.values(name)[cells[inside]]
        return d, np.where(inside, v, np.nan)


def load_state(path):
    """Read one .vtu written by solver2d/vtk_export.py, or one .vti
    process snapshot written by fab/export.py."""
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
                 label=(strings("label") or [""])[0], units=units, kind=(strings("kind") or [""])[0])


class Session:
    """An opened .pvd (sweep) or .vtu (single state)."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pvd":
            self.states = [StateInfo(b, f, n or os.path.basename(f)) for b, f, n in vtk_io.read_pvd(path)]
        elif ext in (".vtu", ".vti"):
            self.states = [StateInfo(0.0, self.path, os.path.basename(path))]
        else:
            raise ValueError(f"{path}: expected a .pvd, .vtu or .vti file")
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


# in-plane axes (u, v) of a section normal to x, y or z; y is depth, so the
# x- and z-sections keep it as v and show the device the right way up
SECTION_AXES = {"x": (2, 1), "y": (0, 2), "z": (0, 1)}
AXIS_NAMES = "xyz"


class Section:
    """A planar cut of a 3D state, flattened into its own 2D mesh so it can be
    viewed, probed and cut like a 2D result. Display only: probes and cut
    lines map their in-plane points back to 3D and sample the full 3D grid
    (State.probe / State.cut_line), so the section adds no interpolation."""

    def __init__(self, state, axis, pos):
        self.state, self.axis, self.pos = state, axis, float(pos)
        self.n = AXIS_NAMES.index(axis)
        self.u, self.v = SECTION_AXES[axis]
        self._regions = {}

    @property
    def axis_labels(self):
        return AXIS_NAMES[self.u], AXIS_NAMES[self.v]

    def to3d(self, uv):
        uv = np.asarray(uv, dtype=float)
        p = np.zeros(uv.shape[:-1] + (3,))
        p[..., self.n] = self.pos
        p[..., self.u], p[..., self.v] = uv[..., 0], uv[..., 1]
        return p

    def to2d(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        return np.stack([xyz[..., self.u], xyz[..., self.v]], axis=-1)

    def contains(self, xyz, tol=1e-9):
        """True if 3D point xyz lies in the section plane."""
        b = self.state.grid.bounds
        scale = max(1.0, *(b[2 * i + 1] - b[2 * i] for i in range(3)))
        return abs(float(np.asarray(xyz)[self.n]) - self.pos) <= tol * scale

    def region(self, k):
        """Region k cut by the plane, as a flat mesh in (u, v, 0) with every
        field carried over (point fields interpolated, cell fields per cut
        cell); None if the plane misses the region."""
        if k not in self._regions:
            o = [0.0, 0.0, 0.0]
            o[self.n] = self.pos
            if self.state.is_process:
                s = self.state.slice_region(k, self.axis, self.pos)
            else:
                g = self.state.region(k)
                s = g.slice(normal=self.axis, origin=o) if g.n_cells else None
            if s is not None and s.n_points:
                s = s.copy()
                s.points = np.column_stack([self.to2d(s.points), np.zeros(s.n_points)])
            else:
                s = None
            self._regions[k] = s
        return self._regions[k]

    def bounds2d(self):
        """(umin, umax, vmin, vmax) of the device in the section plane."""
        b = self.state.grid.bounds
        return b[2 * self.u], b[2 * self.u + 1], b[2 * self.v], b[2 * self.v + 1]

    def outline3d(self):
        """The section rectangle (device bounds) as 3D corner points, closed."""
        u0, u1, v0, v1 = self.bounds2d()
        return self.to3d(np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1], [u0, v0]]))

    def axis_cut(self, uv, direction):
        """3D end points of the cut through in-plane point uv spanning the
        device along u ("h") or v ("v")."""
        u0, u1, v0, v1 = self.bounds2d()
        u, v = float(uv[0]), float(uv[1])
        ends = [[u0, v], [u1, v]] if direction == "h" else [[u, v0], [u, v1]]
        return self.to3d(np.array(ends))


def device_axis_cut(state, xyz, direction):
    """3D end points of the cut through xyz spanning a 2D device along x
    ("h") or y ("v")."""
    b = state.grid.bounds
    p0, p1 = np.array(xyz, dtype=float), np.array(xyz, dtype=float)
    i = 0 if direction == "h" else 1
    p0[i], p1[i] = b[2 * i], b[2 * i + 1]
    return p0, p1


def cut_axis(p0, p1, tol=1e-12):
    """Index of the coordinate a cut runs along if it is axis-aligned, else
    None (then plot against distance)."""
    d = np.abs(np.asarray(p1, float) - np.asarray(p0, float))
    moving = np.flatnonzero(d > tol * max(1.0, float(d.max())))
    return int(moving[0]) if len(moving) == 1 else None


def _voxel_boundary(grid, k):
    """PolyData of the voxel faces bounding the cells with region_id == k
    (domain walls included), outward-oriented quads on the grid nodes, with
    every point field copied from those nodes; None if region k is empty."""
    nx, ny, nz = grid.dimensions
    rid = np.asarray(grid.cell_data["region_id"]).reshape(nz - 1, ny - 1, nx - 1)     # [kz, jy, ix]
    m = np.pad(rid == k, 1)
    quads = []
    nshape = (nz, ny, nx)
    for a in range(3):                           # array axis 0 = z, 1 = y, 2 = x
        lo = np.moveaxis(m, a, 0)
        d_out = lo[:-1] & ~lo[1:]                # region below, not above: face normal +a
        d_in = ~lo[:-1] & lo[1:]
        for faces, flip in ((d_out, False), (d_in, True)):
            f = np.moveaxis(faces, 0, a)         # padded-cell index frame, face plane index along a
            idx = np.argwhere(f)
            if not len(idx):
                continue
            base = idx.copy()
            others = [b for b in range(3) if b != a]
            base[:, others] -= 1                 # padding offset in the in-plane axes
            # the face plane index along a is already the node index
            corners = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                c = base.copy()
                c[:, others[0]] += du
                c[:, others[1]] += dv
                corners.append(np.ravel_multi_index(c.T, nshape))
            q = np.stack(corners, axis=1)
            # array axes are (z, y, x): (u, v, a) is right-handed in xyz
            # only for a = y, so the winding flips with the axis and side
            if not (flip ^ (a == 1)):
                q = q[:, ::-1]
            quads.append(q)
    if not quads:
        return None
    q = np.concatenate(quads)
    nodes, inv = np.unique(q, return_inverse=True)
    iz, jy, ix = np.unravel_index(nodes, nshape)
    pts = np.asarray(grid.origin) + np.column_stack([ix, jy, iz]) * np.asarray(grid.spacing)
    faces = np.column_stack([np.full(len(q), 4), inv.reshape(q.shape)]).ravel()
    out = pv.PolyData(pts, faces)
    for name in grid.point_data.keys():
        out.point_data[name] = np.asarray(grid.point_data[name])[nodes]
    return out
