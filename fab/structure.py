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
# HKstack: a ~0.6 nm SiO2 interlayer + ~1.4 nm HfO2 lumped into one 2 nm layer
# with the same EOT (0.85 nm: eps = 3.9 * 2.0 / 0.85). A 0.6 nm interlayer of
# its own needs a 0.25 nm voxel (two cells across); the lumped layer gives
# the same gate capacitance at 0.5 nm.
INSULATOR_EPS = {"SiO2": 3.9, "SiN": 7.5, "Si3N4": 7.5, "HfO2": 22.0, "Al2O3": 9.0, "SiON": 5.5, "HKstack": 9.2}
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
    parent: str = None                  # a layer carved out of another region (the epi's seed):
                                        # one-voxel slivers between the two merge (cell_labels)


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
        `mask` restricts the deposit to its open area; with mode conformal
        `to_height` caps it (a selective epitaxy grown to a height)."""
        g = self.grid
        if mode == "conformal":
            base = self.phi_union(self.regions_of(materials=on)) if on else self.solid()
            d = levelset.redistance(base, g.h, band=thickness + 3 * g.h.max())
            phi = d - thickness
            if to_height is not None:
                phi = np.maximum(phi, -to_height - g.y)
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

    def epitaxy(self, material, name, on, to_height, shape="facet", lateral=5.0, mask=None,
                facet_angle=54.7356, lattice=None):
        """Selective S/D epitaxy with a crystal shape in the cross-section
        across the width (y-z), extruded along x over the open area of
        `mask`, capped at `to_height` (a flat (100) top).

        Seeds: nodes of the `on` materials touching gas in the open area
        (the recessed fin, the fin/sheet ends under the spacer, a recess
        floor). Seeds overlapping in z form one group - a fin or a sheet
        stack - and each group grows one shape spanning its seeds in height,
        from its floor (the deepest of its lowest seed and the solid under
        it in the open area - a recessed fin's base, a recess floor, the
        bottom dielectric: the epi fills the trench bottom-up) to the cap:
          facet:   PMOS SiGe. {111} facets (facet_angle from the horizontal)
                   rise from the floor corners and fall from the cap at the
                   seed edges; at most `lateral` (nm) beyond the seed edge a
                   vertical side joins them (or they meet: a diamond).
          ellipse: NMOS Si:P blob. An ellipse `lateral` beyond the seed edge
                   at its widest, meeting the floor halfway out (a foot on
                   the STI, not a sliver along the fin's side), holding the
                   top corners and reaching the cap (taller than wide for a
                   fin or stack).
        The side walls of the domain in z are symmetry planes: each group's
        mirror images across them grow too, so epi from neighbouring fins
        merges when the pitch is tight. A merge or gap narrower than two
        voxels is an error (a sliver the mesh cannot resolve).

        lattice (nm, a multiple of the voxel in y and z; None = the voxel):
        the outer shape is drawn in lattice-sized blocks of the cross-section.
        A curved or slanted surface staircases at the voxel, and the device
        mesh (single-material boxes) must resolve every step: at 0.5 nm the
        epi's skin was half the added mesh. The seed envelope (where the epi
        meets the fin / sheets), the cap and the floor stay exact."""
        g = self.grid
        if shape not in ("facet", "ellipse"):
            raise ValueError(f"epitaxy shape must be facet or ellipse, not {shape!r}")
        hz = g.h[2]
        on_phi = self.phi_union(self.regions_of(materials=on))
        solid = self.solid()
        open2d = self.mask2d(mask) < 0 if mask is not None else np.ones((g.shape[0], g.shape[2]), bool)
        gas, _ = ndimage.label(solid > 0)                               # the ambient: gas open to the top
        gas = np.isin(gas, np.setdiff1d(np.unique(gas[:, 0, :]), [0])) & open2d[:, None, :]
        # on the surface up to the tie-break (a recess floor's corner node is
        # +EPS silicon, -EPS of the STI beside it), touching gas diagonally
        # included (that corner meets the trench only across the diagonal)
        seed = (on_phi <= EPS) & ndimage.binary_dilation(gas, structure=np.ones((3, 3, 3), bool))
        if not seed.any():
            raise ValueError(f"epitaxy {name!r}: no exposed {on} in the open area")
        y, z = g.axes[1], g.axes[2]
        y_cap = -to_height
        top = levelset.column_top(solid, y)                             # (x, z)
        zcols = seed.any(axis=(0, 1))
        lab, n = ndimage.label(zcols)
        groups = []
        for k in range(1, n + 1):
            iz = np.flatnonzero(lab == k)
            s = seed[:, :, iz[0]:iz[-1] + 1]
            jy = np.flatnonzero(s.any(axis=(0, 2)))
            floor = top[:, iz[0]:iz[-1] + 1][open2d[:, iz[0]:iz[-1] + 1]]
            floor = floor[np.isfinite(floor)]
            y_bot = max(y[jy[-1]], floor.max() if len(floor) else -np.inf)
            groups.append((z[iz[0]], z[iz[-1]], y[jy[0]], y_bot))
        t = np.tan(np.radians(facet_angle))
        polys, boxes = [], []
        for z0, z1, y_top, y_bot in groups:
            zc, hw = 0.5 * (z0 + z1), 0.5 * (z1 - z0)
            if y_bot <= y_cap:
                raise ValueError(f"epitaxy {name!r}: cap at height {to_height:g} is below the seed floor")
            if shape == "facet":
                L = min(lateral, (y_bot - y_cap) / (2 * t))           # facets meet before `lateral`: a diamond
                half = [(hw, y_bot), (hw + L, y_bot - L * t), (hw + L, y_cap + L * t), (hw, y_cap)]
                ext = hw + L
            else:
                a = hw + lateral
                kf = np.sqrt(1.0 - ((hw + 0.5 * lateral) / a) ** 2)  # the foot on the floor
                kt = np.sqrt(1.0 - (hw / a) ** 2)                     # the seed's top corners
                b = max((y_bot - y_top) / (kf + kt), (y_bot - y_cap) / (1 + kf))
                yc = y_bot - b * kf
                th = np.linspace(-np.pi / 2, np.pi / 2, 65)
                half = [(a * np.cos(q), yc + b * np.sin(q)) for q in th[::-1]]   # bottom to top
                ext = a
            right = [(zc + u, yy) for u, yy in half]
            left = [(zc - u, yy) for u, yy in reversed(half)]
            poly = right + left
            # the seed envelope down to the floor is always filled (the space
            # the S/D recess opened): a curved side through the floor's corner
            # leaves a one-voxel slot there
            box = [(z0, y_bot), (z1, y_bot), (z1, y_top), (z0, y_top)]
            for wall in (g.lo[2], g.hi[2]):
                gap = 2 * abs(wall - zc) - 2 * ext
                if abs(gap) < 2 * hz:
                    raise ValueError(f"epitaxy {name!r}: the epi and its mirror image across the side wall "
                                     f"z = {wall:g} {'overlap' if gap < 0 else 'are apart'} by only "
                                     f"{abs(gap):.2g} nm (< 2 voxels): change `lateral` or the pitch")
            for pp, out in ((poly, polys), (box, boxes)):
                out.append((pp, y_bot, z0))                            # (lattice anchored on the seed
                for wall in (g.lo[2], g.hi[2]):                          #  edge and the floor)
                    out.append(([(2 * wall - pz, py) for pz, py in pp], y_bot, 2 * wall - z1))

        def field(pts_y, pts_z, item):
            # nothing grows below its floor (an ellipse's lower arc would slip
            # down beside a bottom dielectric's edge)
            pp, yb, _ = item
            return np.maximum(levelset.polygon_sdf(pts_z, pts_y, pp), pts_y - yb)

        Y, Z = np.meshgrid(y, z, indexing="ij")
        if lattice:
            shapes = [_on_lattice(field, it, g, lattice, (it[1], it[2])) for it in polys]
        else:
            shapes = [field(Y, Z, it) for it in polys]
        f2 = np.min(shapes + [field(Y, Z, it) for it in boxes], axis=0)  # (y, z)
        phi = np.broadcast_to(f2[None, :, :], g.shape)
        phi = np.maximum(phi, -to_height - g.y)
        if mask is not None:
            # one voxel past the mask edge: the opening ends at a solid face
            # (the spacer), which clips the epi itself; a mask edge on the
            # face makes the cells there depend on two nearly equal fields
            phi = np.maximum(phi, self._column(self.mask2d(mask)) - g.h[0])
        self._add(name, material, phi)

    def epi_seed(self, region, name, material, on, thickness):
        """The seed (buffer) layer of a bilayer S/D epi: the first
        `thickness` nm of `region` grown on the `on` materials (the recessed
        silicon) becomes region `name` of `material` - a low-Ge SiGe seed
        under the high-Ge fill. The epi's outer shape is unchanged.

        Conformal growth: the seed spreads from the silicon nodes touching
        the epi, through the epi only (never across a dielectric), in box
        (max-norm) steps - its inner surface stays flat along flat silicon
        faces and square at their corners, no curved staircase for the
        device mesh. Its level set is a node distance to that surface."""
        g = self.grid
        k = self.region_index(region)
        if k is None:
            raise ValueError(f"epi_seed: no region {region!r}")
        if self.region_index(name) is not None:
            raise ValueError(f"epi_seed: region {name!r} exists already")
        epi = self.regions[k]
        lab = self.node_labels()
        inside = lab == k
        # growth starts at the epi's first nodes on the silicon (the
        # interface lies at them, to the node tie-break)
        si = np.isin(lab, [self.region_index(r.name) for r in self.regions_of(materials=on)])
        src = inside & ndimage.binary_dilation(si, structure=np.ones((3, 3, 3), bool))
        if not src.any():
            raise ValueError(f"epi_seed: {region!r} does not touch {on}")
        steps = [int(round(thickness / h)) for h in g.h]
        near = src.copy()
        for n in range(max(steps)):
            grow = near.copy()
            for a in range(3):
                if n < steps[a]:
                    grow |= ndimage.maximum_filter1d(grow, 3, axis=a, mode="constant")
            near = (grow & inside) | src
        near &= inside
        # each node keeps the epi's field in its own layer and a positive
        # distance to the other layer in the other one: min(seed, fill) is
        # exactly the old epi field, so the solid (and every later deposit,
        # etch and label decided from it) does not see the split
        d_out = ndimage.distance_transform_edt(~near, sampling=g.h) - 0.5 * min(g.h)
        d_in = ndimage.distance_transform_edt(near, sampling=g.h) - 0.5 * min(g.h)
        seed_phi = np.where(near, epi.phi, np.maximum(epi.phi, d_out))
        epi.phi = np.where(near, np.maximum(epi.phi, d_in), epi.phi)
        self.regions.insert(k + 1, Region(name, material, seed_phi, parent=region))
        self.regions = [r for r in self.regions if (r.phi < 0).any()]

    def contact_plug(self, regions, mask, floor_height, depth, shell):
        """Contact etch into a narrow S/D epi: below a contact trench whose
        floor is at `floor_height`, remove the `regions` (the epi) in a
        bullet across the width (y-z) - straight sides `shell` (nm) inside
        the epi's edges at the floor, a half-circle bottom (the blunt head),
        the tip `depth` below the floor - over the open area of `mask`. The
        contact fill then wraps metal down the epi's inside: a contact area
        of the plug's sides and bottom, not the epi top alone. One plug per
        piece of epi in z (a fin, a sheet stack)."""
        g = self.grid
        targets = self.regions_of(names=regions)
        phi_t = self.phi_union(targets)
        open2d = self.mask2d(mask) < 0
        y, z = g.axes[1], g.axes[2]
        j = int(np.argmin(np.abs(y - (-floor_height + g.h[1]))))        # the node row just below the floor
        row = (phi_t[:, j, :] < 0) & open2d
        lab, n = ndimage.label(row.any(axis=0))
        f2 = np.full((g.shape[1], g.shape[2]), levelset.BIG)
        Y, Z = np.meshgrid(y, z, indexing="ij")
        y_floor = -floor_height
        for k in range(1, n + 1):
            iz = np.flatnonzero(lab == k)
            zc = 0.5 * (z[iz[0]] + z[iz[-1]])
            r = 0.5 * (z[iz[-1]] - z[iz[0]]) - shell
            if r <= 0:
                continue
            yc = y_floor + max(depth - r, 0.0)                          # centre of the head
            ry = min(r, depth)                                          # a shallow plug: a half-ellipse
            th = np.linspace(0.0, np.pi, 33)
            head = [(zc + r * np.cos(t), yc + ry * np.sin(t)) for t in th]       # right -> bottom -> left
            poly = head + [(zc - r, g.lo[1] - 1.0), (zc + r, g.lo[1] - 1.0)]     # sides up out of the epi
            f2 = np.minimum(f2, levelset.polygon_sdf(Z, Y, poly))
        phi_e = np.maximum(np.broadcast_to(f2[None], g.shape), self._column(self.mask2d(mask)))
        self._remove(phi_e, targets)

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

    def epi_doping(self, species, peak, sx, sy, sz=None, regions=None):
        """Analytic in-situ doped S/D epitaxy: `peak` (cm^-3) inside the epi
        `regions`, and outside them a Gaussian tail of the distance to the
        epi, scaled per axis:

            N = peak * exp(-(dx/sx)^2/2 - (dy/sy)^2/2 - (dz/sz)^2/2)

        with (dx, dy, dz) the offset to the nearest epi point in the scaled
        metric - for a box-shaped epi exactly the separable
        exp(-dx^2/2sx^2) exp(-dy^2/2sy^2). sx (nm) sets the junction under
        the spacer (the overlap), sy the tail below the epi; sz defaults to
        sx. Only semiconductor nodes keep it (net_doping), and only those
        connected to the epi through semiconductor: the dopant does not
        cross a dielectric (bottom dielectric isolation under a nanosheet
        stack, a buried oxide) however thin."""
        g = self.grid
        sz = sx if sz is None else sz
        # the epi's nodes, its own surface included - but a deposit's level
        # set is also -EPS on surface nodes of the solid it grew on, all
        # within reach (an STI/fin interface far down): a surface node
        # counts only next to the epi's interior
        phi = self.phi_union(self.regions_of(names=regions))
        core = phi < -1e-3 * g.h.min()
        inside = core | ((phi <= 0) & ndimage.binary_dilation(core, structure=np.ones((3, 3, 3), bool)))
        if not inside.any():
            raise ValueError(f"epi_doping: no epi region among {regions}")
        d = ndimage.distance_transform_edt(~inside, sampling=(g.h[0] / sx, g.h[1] / sy, g.h[2] / sz))
        semi = self.semiconductor_nodes()
        comp, _ = ndimage.label(semi | inside)
        ids = np.unique(comp[inside])
        cut_off = semi & ~np.isin(comp, ids[ids > 0])           # semiconductor the epi does not reach
        self._add_doping(species, np.where(cut_off, 0.0, peak * np.exp(-0.5 * d ** 2)))

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
        if not clean:
            return lab
        lab = _despeckle(lab)[0]
        for k, r in enumerate(self.regions):
            if r.parent is not None and self.region_index(r.parent) is not None:
                lab = _merge_slivers(lab, k, self.region_index(r.parent))
        return lab

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
        # a layer carved out of a region (its `parent`: the epi's seed)
        # competes with the other regions as one with it - their union is
        # the parent's old field, and the pieces' own means would read
        # higher where their shared boundary meets the outer surface - and
        # then takes the cells where its mean is below the parent's
        children = {k: self.region_index(r.parent) for k, r in enumerate(self.regions)
                    if r.parent is not None and self.region_index(r.parent) is not None}
        family = {pk: [pk] + [ck for ck, p in children.items() if p == pk] for pk in set(children.values())}
        for k, r in enumerate(self.regions):
            if k in children:
                continue
            members = family.get(k, [k])
            phi = r.phi if len(members) == 1 else np.minimum.reduce([self.regions[m].phi for m in members])
            c = _cell_mean(phi)
            take = c < best
            lab[take] = k
            best = np.where(take, c, best)
            if len(members) > 1:
                own = np.full(lab.shape, np.inf)
                pick = np.full(lab.shape, k, dtype=np.int16)
                for m in members:
                    cm = _cell_mean(self.regions[m].phi)
                    sel = cm < own
                    pick[sel] = m
                    own = np.where(sel, cm, own)
                lab[take] = pick[take]
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
        for k, r in enumerate(self.regions):
            if r.parent is not None and self.region_index(r.parent) is not None:
                lab = _merge_slivers(lab, k, self.region_index(r.parent))
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


def _on_lattice(field, item, g, q, anchor):
    """A 2D (y, z) shape drawn in q x q blocks: the blocks of a lattice
    through `anchor` (y, z; a node), inside where field() is negative at
    their centres, returned as the signed distance (nm) on the voxel nodes
    to that union of blocks - zero on its faces, which lie on nodes (as for
    any flat face)."""
    r = [int(round(q / g.h[a])) for a in (1, 2)]
    if any(abs(r[i] * g.h[a] - q) > 1e-9 for i, a in enumerate((1, 2))):
        raise ValueError(f"epitaxy lattice {q:g} nm is not a multiple of the voxel {tuple(g.h[1:])}")
    ny, nz = g.shape[1] - 1, g.shape[2] - 1                         # voxel cells
    # each voxel cell -> its block (index along the lattice through the anchor)
    jy = np.floor((g.axes[1][:-1] + 0.5 * g.h[1] - anchor[0]) / q).astype(int)
    kz = np.floor((g.axes[2][:-1] + 0.5 * g.h[2] - anchor[1]) / q).astype(int)
    by, bz = np.unique(jy), np.unique(kz)
    CY, CZ = np.meshgrid(anchor[0] + q * (by + 0.5), anchor[1] + q * (bz + 0.5), indexing="ij")
    block = field(CY, CZ, item) < 0
    cell = block[np.searchsorted(by, jy)][:, np.searchsorted(bz, kz)]
    c = np.pad(cell, 1, mode="edge")                                 # (the domain edges continue the shape)
    n_in = c[:-1, :-1].astype(int) + c[1:, :-1] + c[:-1, 1:] + c[1:, 1:]   # inside cells around each node
    inside, face = n_in == 4, (n_in > 0) & (n_in < 4)
    if not face.any():
        return np.where(inside, -levelset.BIG, levelset.BIG)
    d = ndimage.distance_transform_edt(~face, sampling=g.h[1:])
    return np.where(inside, -d, d)


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


def _merge_slivers(lab, child, parent, max_pass=8):
    """One-voxel slivers between a layer carved out of a region (`child`,
    the epi's seed) and that region (`parent`): a later etch (the contact
    plug) cutting their shared boundary at a glancing angle leaves cells
    of one in a one-voxel layer with the other beside it. Both are the same
    epi, so the sliver is noise, not a film. First the child's slivers join
    the parent (repeated until none is left); then a parent sliver beside
    the child joins it only where that leaves no child sliver - one
    direction at a time, so a cell cannot flip back and forth."""
    lab = lab.copy()

    def slivers(own, other):
        out = np.zeros(lab.shape, bool)
        for ax in range(3):
            v, o = np.moveaxis(lab, ax, 0), np.moveaxis(out, ax, 0)
            lo, mid, hi = v[:-2], v[1:-1], v[2:]
            o[1:-1] |= (mid == own) & (lo != own) & (hi != own) & ((lo == other) | (hi == other))
        return out

    for _ in range(max_pass):
        s = slivers(child, parent)
        if not s.any():
            break
        lab[s] = parent
    s = slivers(parent, child)
    if s.any():
        trial = lab.copy()
        trial[s] = child
        lab_save, lab = lab, trial
        bad = slivers(child, parent) & s
        lab = lab_save
        lab[s & ~bad] = child
    return lab


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
