"""The virtual-fab structure: named regions (each a material and a level-set
function on a uniform node grid), 2D lithography masks, and accumulated
doping. Each process step is one geometric operation (fab/flow.py maps YAML
steps onto the methods here).

Coordinates (nm): x along the channel (source -> drain), y depth (positive
DOWN, y = 0 the original silicon surface, as in the 2D drivers), z across
the width. Process parameters speak in heights (positive UP): height = -y.

Regions never overlap: every deposit is clipped to the space outside the
existing solid, and etches only shrink regions.

Ties: inside is phi < 0, so the complement of a set is phi >= 0, written
as -phi - EPS. Without the EPS a node lying exactly on a shared boundary
(geometry on whole nm lands on nodes all the time) is in neither the set
nor its complement, and a mask edge leaves a one-voxel slot of nothing.
Removals are inclusive: an etch takes its boundary nodes too. Nodes are labelled by the
region with the most negative phi (node_labels), which partitions the
domain consistently even where two regions' fields disagree at sub-voxel
level.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.special import erfc

from fab import levelset

DONORS = {"P", "As", "Sb", "N"}           # N: donor in SiC; harmless for Si flows
ACCEPTORS = {"B", "In", "Al", "Ga"}
SEMICONDUCTORS = {"Si", "SiGe", "Ge", "SiC", "GaAs"}
CONDUCTORS = {"Poly", "Metal", "TiN", "TaN", "W", "Al", "Cu", "Co", "NiSi", "TiSi2"}
INSULATOR_EPS = {"SiO2": 3.9, "SiN": 7.5, "Si3N4": 7.5, "HfO2": 22.0, "Al2O3": 9.0, "SiON": 5.5}
EPS = 2.0 ** -20                          # nm (~1e-6, exact in binary); tie-break for complements


def is_semiconductor(material):
    return material in SEMICONDUCTORS or material.startswith("SiGe")      # SiGe0.3 etc.


def _min_above(f, depth, hy):
    """min of f(y - s) over s in [0, depth] along axis 1 (y grows downward,
    so this looks up). f is piecewise linear in y, so the minimum is at a
    node in the window or at its far end, which is interpolated."""
    if not np.isfinite(depth):
        return np.minimum.accumulate(f, axis=1)
    k = int(np.floor(depth / hy + 1e-9))
    frac = depth / hy - k
    out = ndimage.minimum_filter1d(f, size=k + 1, axis=1, origin=k // 2, mode="nearest") if k else f.copy()
    if frac > 1e-9:
        def up(m):                                  # f[i - m] along y, top row repeated above the domain
            return np.concatenate([np.repeat(f[:, :1], m, axis=1), f[:, :f.shape[1] - m]], axis=1)
        out = np.minimum(out, (1 - frac) * up(k) + frac * up(k + 1))
    return out


@dataclass
class Grid:
    """Uniform node grid: x0 + i*h along each axis (nm)."""
    lo: tuple
    hi: tuple
    h: tuple

    def __post_init__(self):
        self.h = np.asarray(self.h, dtype=float)
        self.lo = np.asarray(self.lo, dtype=float)
        n = np.round((np.asarray(self.hi, float) - self.lo) / self.h).astype(int) + 1
        self.shape = tuple(int(v) for v in n)
        self.hi = self.lo + (n - 1) * self.h
        self.axes = [self.lo[a] + np.arange(n[a]) * self.h[a] for a in range(3)]

    @property
    def x(self):
        return self.axes[0][:, None, None]

    @property
    def y(self):
        return self.axes[1][None, :, None]

    @property
    def z(self):
        return self.axes[2][None, None, :]

    @property
    def n_nodes(self):
        return int(np.prod(self.shape))


@dataclass
class Region:
    name: str
    material: str
    phi: np.ndarray                     # float64 level set at the nodes, < 0 inside (float64: the
                                        # EPS tie-break needs exact arithmetic near 0)


@dataclass
class Structure:
    grid: Grid
    regions: list = field(default_factory=list)
    masks: dict = field(default_factory=dict)
    donors: np.ndarray = None
    acceptors: np.ndarray = None
    contacts: list = field(default_factory=list)

    def __post_init__(self):
        if self.donors is None:
            self.donors = np.zeros(self.grid.shape)
            self.acceptors = np.zeros(self.grid.shape)

    # --- fields -------------------------------------------------------------------
    def region_index(self, name):
        for k, r in enumerate(self.regions):
            if r.name == name:
                return k
        return None

    def regions_of(self, materials=None, names=None):
        """Regions whose material is in `materials` or name in `names` (all
        regions if both are None)."""
        if materials is None and names is None:
            return list(self.regions)
        m, n = set(materials or ()), set(names or ())
        return [r for r in self.regions if r.material in m or r.name in n]

    def phi_union(self, regions):
        """Level set of the union of `regions` (+BIG if empty)."""
        if not regions:
            return np.full(self.grid.shape, levelset.BIG)
        out = regions[0].phi.astype(float)
        for r in regions[1:]:
            np.minimum(out, r.phi, out=out)
        return out

    def solid(self):
        return self.phi_union(self.regions)

    def mask2d(self, spec):
        """2D field over (x, z), negative where the mask is OPEN (where it lets
        the step act). spec: a mask name ("!name" inverts), a dict
        {rect: [x0, x1, z0, z1]} / {polygon: [[x, z], ...]} / {not: spec} /
        {union: [specs]}, or a list (union)."""
        X, Z = np.meshgrid(self.grid.axes[0], self.grid.axes[2], indexing="ij")
        if spec is None:
            return np.full(X.shape, -levelset.BIG)
        if isinstance(spec, str):
            if spec.startswith("!"):
                return -self.mask2d(spec[1:]) - EPS
            if spec not in self.masks:
                raise KeyError(f"unknown mask {spec!r} (define it with a pattern step first)")
            return self.masks[spec]
        if isinstance(spec, list):
            return np.min([self.mask2d(s) for s in spec], axis=0)
        if "rect" in spec:
            return levelset.polygon_sdf(X, Z, levelset.rect_poly(*spec["rect"]))
        if "polygon" in spec:
            return levelset.polygon_sdf(X, Z, spec["polygon"])
        if "not" in spec:
            return -self.mask2d(spec["not"]) - EPS
        if "union" in spec:
            return np.min([self.mask2d(s) for s in spec["union"]], axis=0)
        raise ValueError(f"bad mask spec {spec!r}")

    def _add(self, name, material, phi_new):
        phi_new = np.maximum(phi_new, -self.solid() - EPS)    # never overlap existing solid
        if (phi_new[:, 0, :] < 0).any():
            raise ValueError(f"deposit of {name!r} reaches the top of the domain (height "
                             f"{-self.grid.lo[1]:g} nm): raise the domain top (etches need gas above)")
        k = self.region_index(name)
        if k is None:
            self.regions.append(Region(name, material, np.array(phi_new, dtype=float)))
        else:
            if self.regions[k].material != material:
                raise ValueError(f"region {name!r} is {self.regions[k].material}, not {material}")
            self.regions[k].phi = np.minimum(self.regions[k].phi, phi_new)

    def _remove(self, phi_e, regions):
        """Remove {phi_e <= 0} (boundary included) from `regions`. Only nodes
        inside a region or on its surface (phi <= 0) change: a node already
        outside keeps its value, and with it the sub-voxel position of a
        surface the etch did not touch. Nodes exactly on a removed surface
        must change too, or they stay as stray zeros that later column
        minima (the anisotropic etch's visibility) would pick up."""
        for r in regions:
            r.phi = np.where(r.phi <= 0, np.maximum(r.phi, -phi_e), r.phi)
        self.regions = [r for r in self.regions if (r.phi < 0).any()]

    def _column(self, f2d):
        return f2d[:, None, :]

    # --- process operations ------------------------------------------------------------
    def substrate(self, material, name, top_height=0.0):
        """Fill everything below `top_height` (nm, up = positive)."""
        g = self.grid
        phi = np.broadcast_to(-top_height - g.y, g.shape)       # y_top - y < 0 below
        self._add(name, material, np.array(phi, dtype=float))

    def pattern(self, name, spec):
        """Define a lithography mask (2D, open where negative)."""
        self.masks[name] = self.mask2d(spec)

    def deposit(self, material, name, mode, thickness=None, to_height=None, on=None, mask=None):
        """mode conformal: everything within `thickness` of the solid (or of
        the `on` materials only - selective growth), outside the solid.
        mode fill: everything below `to_height`, outside the solid.
        mode directional: `thickness` straight up from each column's top.
        `mask` restricts the deposit to its open area."""
        g = self.grid
        if mode == "conformal":
            base = self.phi_union(self.regions_of(materials=on)) if on else self.solid()
            d = levelset.redistance(base, g.h, band=thickness + 3 * g.h.max())
            phi = d - thickness
        elif mode == "fill":
            phi = np.broadcast_to(-to_height - g.y, g.shape).astype(float)
        elif mode == "directional":
            top = levelset.column_top(self.solid(), g.axes[1])
            top = np.where(np.isfinite(top), top, g.axes[1][-1])
            phi = np.broadcast_to(self._column(top - thickness) - g.y, g.shape).astype(float)
        else:
            raise ValueError(f"unknown deposit mode {mode!r}")
        if mask is not None:
            phi = np.maximum(phi, self._column(self.mask2d(mask)))
        self._add(name, material, phi)

    def etch(self, mode, materials=None, depth=None, stop_on=None, mask=None, thickness=None, regions=None):
        """mode anisotropic: in each open column, remove the target materials
        above a front at `depth` below the column's top surface, never below
        the top of a `stop_on` material. mode isotropic: remove target
        material within `thickness` of the ambient (exposed surfaces).
        Targets: `materials` and/or named `regions` (default: everything
        except stop_on) - a region target etches one layer (e.g. the gate
        oxide) and leaves other regions of the same material (the STI)."""
        g = self.grid
        stop = set(stop_on or ())
        targets = self.regions_of(materials=materials, names=regions) if (materials or regions) else \
            [r for r in self.regions if r.material not in stop]
        if mode == "anisotropic":
            # removed = points with gas visible from above within `depth`
            # straight up: a vertical sliding minimum of the visible-gas
            # field. Visible gas = no solid at or above the point, i.e.
            # -cummin(solid) from the top of each column (shadowed under an
            # overhang). Both are minima along columns of the solid field,
            # so they keep the sub-voxel position of every vertical face and
            # etched edges inherit it (anything built from a per-column top
            # height places edges only to the nearest column).
            visible_gas = -np.minimum.accumulate(self.solid(), axis=1) - EPS
            phi_e = _min_above(visible_gas, np.inf if depth is None else depth, g.h[1])
            if stop:
                st = levelset.column_top(self.phi_union(self.regions_of(materials=stop)), g.axes[1])
                phi_e = np.maximum(phi_e, g.y - self._column(np.where(np.isfinite(st), st, 1e9)))
            if mask is not None:
                phi_e = np.maximum(phi_e, self._column(self.mask2d(mask)))
        elif mode == "isotropic":
            d = levelset.redistance(-self.solid(), g.h, band=thickness + 3 * g.h.max())   # > 0 inside solid
            phi_e = d - thickness
            if mask is not None:
                phi_e = np.maximum(phi_e, self._column(self.mask2d(mask)))
        else:
            raise ValueError(f"unknown etch mode {mode!r}")
        self._remove(np.broadcast_to(phi_e, g.shape), targets)

    def planarize(self, to_height):
        """CMP: remove everything above `to_height`."""
        self._remove(np.broadcast_to(self.grid.y + to_height, self.grid.shape), list(self.regions))

    def implant(self, species, rp, sigma, peak=None, dose=None, lateral_sigma=0.0, blocked_by=None, mask=None):
        """Analytic implant: Gaussian in depth below the top surface of the
        non-blocking solid (so dopant straggling under a mask edge lands in
        the silicon there), times a lateral erfc of the distance to the open
        area (blocked_by footprint and/or mask). peak in cm^-3, or dose in
        cm^-2 with sigma in nm."""
        g = self.grid
        if (peak is None) == (dose is None):
            raise ValueError("implant needs exactly one of peak, dose")
        if peak is None:
            peak = dose / (np.sqrt(2 * np.pi) * sigma * 1e-7)
        blockers = self.regions_of(materials=blocked_by) if blocked_by else []
        rest = [r for r in self.regions if r not in blockers]
        top = levelset.column_top(self.phi_union(rest), g.axes[1])
        depth = g.y - self._column(np.where(np.isfinite(top), top, 1e9))
        # A node ON the surface takes the surface value: the top from the
        # level set can sit a round-off above it, and a recessed contact's
        # floor nodes - the metal/silicon interface the contact model reads -
        # then missed a contact implant entirely (1e20 instead of 6e20).
        on_top = (depth < 0) & (depth >= -0.25 * g.h[1])
        depth = np.where(on_top, 0.0, depth)
        prof = np.where(depth >= 0, peak * np.exp(-0.5 * ((depth - rp) / sigma) ** 2), 0.0)
        s = self.mask2d(mask) if mask is not None else np.full(top.shape, -levelset.BIG)
        if blockers:
            # footprint of the blockers: min over each column of their
            # re-distanced field (set operations leave phi = 0 on surfaces
            # the blocker merely touches, which would read as "blocked")
            reach = 6.0 * lateral_sigma + 2.0 * g.h.max()             # erfc is 1 beyond this
            pb = levelset.local_distance(self.phi_union(blockers), g.h, reach)
            s = np.maximum(s, -pb.min(axis=1))                        # blocked columns -> positive
        if lateral_sigma > 0 and (s > 0).any() and (s < 0).any():
            s = levelset.redistance(s, g.h[[0, 2]])
            lat = 0.5 * erfc(s / (np.sqrt(2.0) * lateral_sigma))
        else:
            lat = (s < 0).astype(float)
        conc = prof * self._column(lat)
        self._add_doping(species, conc)

    def dope(self, species, conc, regions=None, materials=None):
        """Uniform doping inside the given regions/materials (all if none)."""
        rs = self.regions_of(materials=materials, names=regions)
        inside = self.phi_union(rs) < 0
        self._add_doping(species, np.where(inside, float(conc), 0.0))

    def _add_doping(self, species, conc):
        if species in DONORS:
            self.donors = self.donors + conc
        elif species in ACCEPTORS:
            self.acceptors = self.acceptors + conc
        else:
            raise ValueError(f"unknown dopant species {species!r} (donors {sorted(DONORS)}, "
                             f"acceptors {sorted(ACCEPTORS)})")

    # --- labels -------------------------------------------------------------------------
    def node_labels(self):
        """Region index of each node (-1 = ambient/gas)."""
        if not self.regions:
            return np.full(self.grid.shape, -1, dtype=np.int16)
        lab = np.full(self.grid.shape, -1, dtype=np.int16)
        best = np.zeros(self.grid.shape)
        for k, r in enumerate(self.regions):
            take = r.phi < best
            lab[take] = k
            best = np.where(take, r.phi, best)
        return lab

    def cell_labels(self, clean=True):
        """Region index of each voxel cell, from its 8 corners' mean phi.
        clean: remove one-voxel flakes (see _despeckle) - the labels the
        device mesh is built from must not carry voxel noise."""
        lab = self._raw_cell_labels()
        return _despeckle(lab)[0] if clean else lab

    def _raw_cell_labels(self):
        """Solid or gas from the cell mean of the union field (the field
        that defines the solid); which region, from the smallest cell mean
        of the region fields. Deciding gas from the region means alone
        leaves "nobody's" cells where a curved surface crosses a cell: the
        mean of a minimum is below the minimum of the means, so a filled
        ILD (exactly the complement of the old solid at every node) and
        the spacer it wraps can both read as outside."""
        nx, ny, nz = self.grid.shape
        lab = np.full((nx - 1, ny - 1, nz - 1), -1, dtype=np.int16)
        if not self.regions:
            return lab
        best = np.full(lab.shape, np.inf)
        for k, r in enumerate(self.regions):
            c = _cell_mean(r.phi)
            take = c < best
            lab[take] = k
            best = np.where(take, c, best)
        lab[_cell_mean(self.solid()) >= 0] = -1
        return lab

    def check(self):
        """Structure check (run after every build): the defects a device
        solve must not inherit. Returns a list of warning strings (empty =
        clean): one-voxel flakes removed from the labels, enclosed gas
        voids, small detached fragments of a region, and layers thinner
        than two voxels."""
        raw = self._raw_cell_labels()
        lab, n_flakes = _despeckle(raw)
        out = []
        if n_flakes:
            out.append(f"{n_flakes} one-voxel flake cell(s) relabelled to their surroundings")
        gas, n = ndimage.label(lab == -1)
        open_ids = set(np.unique(gas[:, 0, :])) - {0}
        voids = [i for i in range(1, n + 1) if i not in open_ids]
        if voids:
            sizes = ndimage.sum(lab == -1, gas, voids).astype(int)
            where = [tuple(int(v) for v in np.argwhere(gas == i)[0]) for i in voids[:3]]
            out.append(f"{len(voids)} enclosed gas void(s), sizes {sorted(sizes.tolist())[:8]} cells, "
                       f"e.g. at cell index {where}")
        for k, r in enumerate(self.regions):
            m = lab == k
            comp, nc = ndimage.label(m)
            if nc > 1:
                sizes = ndimage.sum(m, comp, range(1, nc + 1)).astype(int)
                small = sizes[sizes < max(64, 0.01 * sizes.max())]
                if len(small):
                    out.append(f"region {r.name!r}: {len(small)} detached fragment(s) of {sorted(small.tolist())[:6]} cells")
            thin = int(_thin_cells(m).sum())
            if thin:
                out.append(f"region {r.name!r}: {thin} cell(s) in a layer one voxel thick")
        return out

    def semiconductor_nodes(self, labels=None):
        lab = self.node_labels() if labels is None else labels
        semi = np.array([is_semiconductor(r.material) for r in self.regions] + [False])
        return semi[lab]                                         # lab = -1 picks the trailing False

    def net_doping(self, labels=None):
        """(donors, acceptors, net) at the nodes, zero outside semiconductors."""
        semi = self.semiconductor_nodes(labels)
        nd = np.where(semi, self.donors, 0.0)
        na = np.where(semi, self.acceptors, 0.0)
        return nd, na, nd - na

    def volume(self, name):
        """Region volume (nm^3) from its cell count (voxel accurate)."""
        k = self.region_index(name)
        return float((self.cell_labels() == k).sum() * np.prod(self.grid.h))


def _despeckle(lab, max_pass=4):
    """Relabel one-voxel flakes: a cell whose two neighbours along some axis
    carry the same label, different from its own, takes that label. A real
    thin film keeps its cells - its two sides are different materials (the
    gate oxide between silicon and gate) - while a flake sticking out of a
    staircased curved surface (gas on both sides) does not. Returns (labels,
    number of cells changed)."""
    lab = lab.copy()
    total = 0
    for _ in range(max_pass):
        changed = 0
        for a in range(3):
            v = np.moveaxis(lab, a, 0)                    # a view: writes go to lab
            lo, mid, hi = v[:-2], v[1:-1], v[2:]
            flip = (lo == hi) & (mid != lo)
            if flip.any():
                mid[flip] = lo[flip]
                changed += int(flip.sum())
        total += changed
        if not changed:
            break
    return lab, total


def _thin_cells(m):
    """Cells of mask m in a run of exactly one cell along some axis with
    the mask absent on both sides (a one-voxel-thick layer or fin)."""
    out = np.zeros(m.shape, bool)
    for a in range(3):
        v = np.moveaxis(m, a, 0)
        o = np.moveaxis(out, a, 0)
        o[1:-1] |= v[1:-1] & ~v[:-2] & ~v[2:]
    return out


def _cell_mean(p):
    """Mean of a node field over each voxel cell's 8 corners."""
    return (p[:-1, :-1, :-1] + p[1:, :-1, :-1] + p[:-1, 1:, :-1] + p[:-1, :-1, 1:]
            + p[1:, 1:, :-1] + p[1:, :-1, 1:] + p[:-1, 1:, 1:] + p[1:, 1:, 1:]) / 8
