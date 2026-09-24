"""Blocky 2D device geometry: axis-aligned rectangular regions with a
painter's-algorithm override order (a later region overrides an earlier one
wherever they overlap - this is how a p+ square gets "dug into" an n
substrate rectangle), plus contact placement on the top/bottom domain
boundary.

All lengths are stored in cm (matching core/params.py's/core/mesh.py's
convention) - only the YAML config layer (mesh2d/config2d.py) works in um.

Domain convention: x in [0, width_cm], y in [0, height_cm] with y=0 the top
(free) surface and y=height_cm the bottom of the substrate. Left (x=0) and
right (x=width_cm) are always the `symmetry` boundary - see the module
docstring in mesh2d/boundary.py for why a side is never a contact surface.
Shape vocabulary is rectangles-only for now; tapered/other shapes are
explicitly future work (see the project's 2D/3D plan).

`TopMesa` generalizes the domain's outer boundary beyond a plain rectangle:
a mesa is a protrusion (e.g. a MOS capacitor's oxide+gate stack) that
extends ABOVE the flat substrate top surface (y<0) over a strict interior
x-range, the same way a real gate stack sits proud of the wafer surface
rather than being "dug into" it like a diode's doping region. A mesa's
own fill (its permittivity/doping) is declared as an ordinary `Region`
with y_range_cm[0] < 0 (the domain's signal for "this region protrudes
above the flat top, and should be treated as a boundary-shaping mesa, not
an embedded interior region") - `Domain2D` derives the stepped outer
boundary polygon from those regions plus the matching `top_mesas` entries,
so the two must describe the same x-range/height for a given mesa.
"""
from dataclasses import dataclass, field

import numpy as np

_EPS_REL = 1e-9  # relative tolerance for "on the rectangle boundary" tests


@dataclass
class Region:
    name: str
    x_range_cm: tuple      # (x0, x1)
    y_range_cm: tuple      # (y0, y1) - y0 < 0 marks a mesa-protrusion region (see module docstring)
    doping_type: str       # "p" | "n" (irrelevant for kind="insulator", but still required)
    concentration_cm3: float
    kind: str = "semiconductor"   # "semiconductor" | "insulator"
    eps_r: float = None    # relative permittivity override - required when kind="insulator"
                             # (e.g. an oxide's 3.9), ignored for "semiconductor" regions
                             # (those use their material's eps_r).
    grading_cm_per_decade: tuple = None  # (gx, gy): a GRADED doping region - full
                             # concentration inside the box, falling off outside it by one
                             # decade per gx (in x) / gy (in y) cm, and ADDED to the net
                             # doping painted so far (dopants compensate) instead of replacing
                             # it. Models a steep implanted/diffused S/D extension tail.
                             # None = the usual uniform, last-region-wins box.
    material: object = None  # semiconductor regions only: a core.params.Material for a
                             # different semiconductor (e.g. SiGe source/drains -> a 2D
                             # heterojunction, see semiconductor_material_index). None =
                             # leave the material as painted so far (doping-only region).


@dataclass
class TopMesa:
    """Geometric footprint of a protrusion above the flat substrate top
    surface (y=0) - purely for outer-boundary-shape/BC-tagging purposes;
    the mesa's actual material fill is a separate `Region` (see module
    docstring). Must not touch the domain's own left/right edges (a mesa is
    always a strict interior feature, like the diode's p+ square)."""
    x_range_cm: tuple      # (x0, x1)
    height_cm: float        # mesa extends from y=-height_cm (its own top) to y=0


@dataclass
class Contact:
    name: str
    surface: str            # "top" | "bottom"
    x_range_cm: tuple
    bias_role: str           # "anode" | "cathode"


@dataclass
class Domain2D:
    width_cm: float
    height_cm: float
    regions: list = field(default_factory=list)   # index 0 = base region (e.g. substrate)
    contacts: list = field(default_factory=list)
    top_mesas: list = field(default_factory=list)  # list of TopMesa

    def doping_at(self, x_cm, y_cm):
        """Signed net doping (Nd - Na, cm^-3) at point(s) (x_cm, y_cm) via
        last-region-wins rectangle membership. Points not covered by any
        region (shouldn't happen if region[0] spans the whole domain) get 0.
        A region with kind="insulator" always contributes net doping 0
        regardless of its doping_type/concentration_cm3 fields (moot for an
        insulator - see material_props_at for its eps/ni handling)."""
        x = np.asarray(x_cm, dtype=float)
        y = np.asarray(y_cm, dtype=float)
        net = np.zeros_like(x)
        tol_x = _EPS_REL * max(self.width_cm, 1e-30)
        tol_y = _EPS_REL * max(self.height_cm, 1e-30)
        for region in self.regions:
            if region.kind == "insulator":
                continue
            x0, x1 = region.x_range_cm
            y0, y1 = region.y_range_cm
            sign = 1.0 if region.doping_type == "n" else -1.0
            if region.grading_cm_per_decade is not None:
                gx, gy = region.grading_cm_per_decade
                dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
                dy = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
                net = net + sign * region.concentration_cm3 * 10.0 ** (-(dx / gx + dy / gy))
                continue
            mask = ((x >= x0 - tol_x) & (x <= x1 + tol_x)
                    & (y >= y0 - tol_y) & (y <= y1 + tol_y))
            net = np.where(mask, sign * region.concentration_cm3, net)
        return net

    def material_props_at(self, x_cm, y_cm, mat):
        """Returns (is_insulator, eps_r) per point(s), via the same last-
        region-wins painter's algorithm as doping_at. Points covered by no
        region (or only by "semiconductor" regions) get the Material's own
        eps_r and is_insulator=False; a point covered by an "insulator"
        region gets that region's eps_r and is_insulator=True.

        An insulator region's membership test is made STRICT (not
        tolerance-inclusive) on its y1 (semiconductor-facing) edge only -
        e.g. a mesa oxide's own y1=0 is exactly the substrate's own y0=0,
        the shared interface line. Using the usual tolerance-inclusive test
        there would let the insulator (processed after the base substrate
        region in the painter's-algorithm order) claim that shared line of
        nodes as ni=0/insulator - wrongly zeroing out the semiconductor's
        own topmost layer of nodes, exactly where inversion/accumulation
        charge is concentrated. The other three sides keep the usual
        inclusive tolerance (no competing semiconductor region touches
        them, so there's no ambiguity to resolve there)."""
        x = np.asarray(x_cm, dtype=float)
        y = np.asarray(y_cm, dtype=float)
        is_insulator = np.zeros(x.shape, dtype=bool)
        eps_r = np.full(x.shape, mat.eps_r, dtype=float)
        tol_x = _EPS_REL * max(self.width_cm, 1e-30)
        tol_y = _EPS_REL * max(self.height_cm, 1e-30)
        for region in self.regions:
            x0, x1 = region.x_range_cm
            y0, y1 = region.y_range_cm
            if region.kind == "insulator":
                mask = ((x >= x0 - tol_x) & (x <= x1 + tol_x)
                        & (y >= y0 - tol_y) & (y < y1))
                if region.eps_r is None:
                    raise ValueError(f"Region {region.name!r}: kind='insulator' requires eps_r")
                is_insulator = np.where(mask, True, is_insulator)
                eps_r = np.where(mask, region.eps_r, eps_r)
            else:
                mask = ((x >= x0 - tol_x) & (x <= x1 + tol_x)
                        & (y >= y0 - tol_y) & (y <= y1 + tol_y))
                was_insulator = is_insulator
                is_insulator = np.where(mask, False, is_insulator)
                if region.material is not None:
                    eps_r = np.where(mask, region.material.eps_r, eps_r)
                else:   # doping-only region: keep the semiconductor painted so far
                    eps_r = np.where(mask & was_insulator, mat.eps_r, eps_r)
        return is_insulator, eps_r

    def semiconductor_materials(self):
        """Distinct semiconductor materials of the regions, in first-use
        order - index k+1 in semiconductor_material_index; index 0 is the
        base Material passed to the mesh builder."""
        out = []
        for region in self.regions:
            if region.kind != "insulator" and region.material is not None and \
                    all(region.material is not m for m in out):
                out.append(region.material)
        return out

    def semiconductor_material_index(self, x_cm, y_cm):
        """Per point: 0 for the base semiconductor, k+1 for
        semiconductor_materials()[k]. Same last-region-wins painting and
        inclusive boundary tolerance as doping_at, so a node ON a doping
        boundary that is also a material boundary takes the later region's
        doping AND material (a metallurgical junction coinciding with a
        heterointerface stays consistent). Only regions with a material
        repaint it."""
        x = np.asarray(x_cm, dtype=float)
        y = np.asarray(y_cm, dtype=float)
        idx = np.zeros(x.shape, dtype=int)
        mats = self.semiconductor_materials()
        tol_x = _EPS_REL * max(self.width_cm, 1e-30)
        tol_y = _EPS_REL * max(self.height_cm, 1e-30)
        for region in self.regions:
            if region.kind == "insulator" or region.material is None:
                continue
            x0, x1 = region.x_range_cm
            y0, y1 = region.y_range_cm
            mask = ((x >= x0 - tol_x) & (x <= x1 + tol_x) & (y >= y0 - tol_y) & (y <= y1 + tol_y))
            k = next(i for i, m in enumerate(mats) if m is region.material)
            idx = np.where(mask, k + 1, idx)
        return idx

    def _mesa_at_x(self, x, tol_x):
        for mesa in self.top_mesas:
            x0, x1 = mesa.x_range_cm
            if x0 - tol_x <= x <= x1 + tol_x:
                return mesa
        return None

    def contains(self, x, y, tol=None):
        """True if (x,y) lies within the domain's own material (base
        rectangle, or within a mesa's protruding footprint)."""
        tol_x = tol if tol is not None else _EPS_REL * max(self.width_cm, 1e-30)
        tol_y = tol if tol is not None else _EPS_REL * max(self.height_cm, 1e-30)
        if -tol_y <= y <= self.height_cm + tol_y and -tol_x <= x <= self.width_cm + tol_x:
            return True
        mesa = self._mesa_at_x(x, tol_x)
        if mesa is not None and -mesa.height_cm - tol_y <= y <= tol_y:
            return True
        return False

    def outer_boundary(self):
        """Returns (vertices, edges, edge_role) tracing the domain's outer
        polygon counterclockwise-in-screen-coordinates (y grows downward):
        the base rectangle's top edge (y=0) with a notch spliced in for
        every top_mesas entry (sorted by x0), so a mesa protrudes above the
        flat top rather than being embedded in it. `edge_role` is one of
        "left" | "right" | "bottom" | "top" | "mesa_wall" per edge - "top"
        covers both the bare substrate top (outside any mesa) and a mesa's
        own top surface (both are contact-eligible); "mesa_wall" is a
        mesa's vertical side (always free_surface, never a contact - a
        contact is only ever carved out of a flat top/bottom surface)."""
        mesas = sorted(self.top_mesas, key=lambda m: m.x_range_cm[0])
        verts = [(0.0, 0.0)]
        roles = []
        x_cursor = 0.0
        for mesa in mesas:
            x0, x1 = mesa.x_range_cm
            verts.append((x0, 0.0)); roles.append("top")      # bare top up to the mesa
            verts.append((x0, -mesa.height_cm)); roles.append("mesa_wall")
            verts.append((x1, -mesa.height_cm)); roles.append("top")  # mesa's own top
            verts.append((x1, 0.0)); roles.append("mesa_wall")
            x_cursor = x1
        verts.append((self.width_cm, 0.0)); roles.append("top")   # bare top after the last mesa
        verts.append((self.width_cm, self.height_cm)); roles.append("right")
        verts.append((0.0, self.height_cm)); roles.append("bottom")
        roles.append("left")  # closing edge back to (0,0)

        n = len(verts)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return np.array(verts, dtype=float), np.array(edges, dtype=int), roles

    def junction_segments(self):
        """Line segments ((x0,y0),(x1,y1)) forming the metallurgical-junction
        boundary of every non-base region, excluding the parts that coincide
        with the domain's own outer boundary (those are a free surface / a
        contact, not an interior interface). This naturally also picks up a
        mesa region's OWN bottom edge (its oxide/semiconductor interface,
        y=0 under the mesa's footprint) since that edge is interior to the
        domain once the mesa protrudes above it - exactly the interface
        that needs tight mesh refinement, with no separate mechanism
        needed. Used to drive point-cloud refinement in mesh2d/pointcloud.py."""
        segs = []
        for region in self.regions[1:]:
            x0, x1 = region.x_range_cm
            y0, y1 = region.y_range_cm
            candidates = [
                ((x0, y0), (x1, y0)),  # top edge of the region
                ((x0, y1), (x1, y1)),  # bottom edge
                ((x0, y0), (x0, y1)),  # left edge
                ((x1, y0), (x1, y1)),  # right edge
            ]
            for (p0, p1) in candidates:
                if self._on_domain_boundary(p0) and self._on_domain_boundary(p1):
                    continue
                segs.append((p0, p1))
        return segs

    def _on_domain_boundary(self, p):
        return self.boundary_point_role(*p) is not None

    def boundary_point_role(self, x, y):
        """Returns the outer-boundary role of point (x,y) - one of "left",
        "right", "bottom", "top" (bare substrate top or a mesa's own top,
        both contact-eligible), "mesa_wall" (a mesa's vertical side, never a
        contact) - or None if the point is not on the domain's outer
        boundary at all (interior, including the y=0 interface line UNDER
        a mesa's footprint, which is interior once the mesa protrudes above
        it)."""
        tol_x = _EPS_REL * max(self.width_cm, 1e-30)
        tol_y = _EPS_REL * max(self.height_cm, 1e-30)
        if abs(y - self.height_cm) <= tol_y:
            return "bottom"
        if abs(x) <= tol_x and -tol_y <= y <= self.height_cm + tol_y:
            return "left"
        if abs(x - self.width_cm) <= tol_x and -tol_y <= y <= self.height_cm + tol_y:
            return "right"
        mesa = self._mesa_at_x(x, tol_x)
        if mesa is None:
            return "top" if abs(y) <= tol_y else None
        if abs(y + mesa.height_cm) <= tol_y:
            return "top"
        x0, x1 = mesa.x_range_cm
        if (abs(x - x0) <= tol_x or abs(x - x1) <= tol_x) and -mesa.height_cm - tol_y <= y <= tol_y:
            return "mesa_wall"
        return None

    def region_corners(self):
        """All rectangle corner points (domain + every region + every mesa),
        used to make sure the point cloud has vertices exactly on every
        geometric feature rather than relying on refinement alone to land
        near one."""
        pts = [(0.0, 0.0), (self.width_cm, 0.0),
               (0.0, self.height_cm), (self.width_cm, self.height_cm)]
        for region in self.regions:
            x0, x1 = region.x_range_cm
            y0, y1 = region.y_range_cm
            pts += [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        for mesa in self.top_mesas:
            x0, x1 = mesa.x_range_cm
            pts += [(x0, -mesa.height_cm), (x1, -mesa.height_cm)]
        for contact in self.contacts:
            x0, x1 = contact.x_range_cm
            y = self._contact_y(contact)
            pts += [(x0, y), (x1, y)]
        return pts

    def _contact_y(self, contact):
        if contact.surface == "bottom":
            return self.height_cm
        tol_x = _EPS_REL * max(self.width_cm, 1e-30)
        x_mid = 0.5 * (contact.x_range_cm[0] + contact.x_range_cm[1])
        mesa = self._mesa_at_x(x_mid, tol_x)
        return 0.0 if mesa is None else -mesa.height_cm
