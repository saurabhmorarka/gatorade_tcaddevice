"""Mesh-quality check, run unconditionally as part of every
mesh2d/mesh2d.py::build_mesh2d call - not an opt-in diagnostic, a
mandatory pre-flight gate, and not specific to any one device (it operates
on whatever triangulation mesh2d/pointcloud.py produced for whatever
Domain2D was given).

**What "mesh quality" means here, and why**: the standard requirement for
the Voronoi/box finite-volume method (see e.g. Fleischmann's TU Wien
thesis on device-simulation mesh generation,
https://www.iue.tuwien.ac.at/phd/fleischmann/node15.html) is a non-obtuse
triangulation - plain Delaunay only guarantees the angle SUM opposite a
shared edge is <=180 deg, not enough to keep every triangle's own
circumcenter inside itself, and a single obtuse angle pushes that
triangle's circumcenter outside it, which is exactly what makes the box
method's per-triangle corner-quad decomposition go negative
(mesh2d/fvgeometry.py's n_negative_subareas).

A literal, guaranteed non-obtuse bound for an arbitrary GRADED 2D mesh
turned out not to be something any mainstream tool actually delivers -
confirmed directly this session: Shewchuk's `triangle` library (adopted
in mesh2d/pointcloud.py specifically for its provable quality guarantee)
only bounds the MINIMUM angle (Ruppert's theorem), and even at the
practical reliability ceiling (~34 deg minimum) a graded mesh around a
rectangular doping step still came out ~13-16% obtuse (worst angle
~110-120 deg) - a real, literature-confirmed limitation, not an
implementation gap. A first attempt at closing that gap by hand (inserting
each obtuse triangle's circumcenter as a Steiner point and re-triangulating)
made things WORSE, not better (obtuse count diverged: 106 -> 368 over a
few iterations) - naive circumcenter insertion is a well-known-bad idea
for exactly this reason (Ruppert's/Chew's actual algorithms use more
careful "off-center" insertion and segment-encroachment checks, not plain
circumcenter insertion), so it was dropped rather than shipped as a
silently-unreliable fix.

**What this module actually checks and enforces**:
- The minimum angle across every triangle must be at least
  `min_angle_floor_deg` (default 15 deg - well below the ~30-32 deg target
  `triangle` is asked for, so this only fires if `triangle`'s own
  constrained refinement failed to hit its target, a real bug worth
  raising loudly, not a normal residual).
- The obtuse-triangle fraction (max angle > 90 deg) is reported (always,
  every run) as a WARNING, not a hard gate - per the finding above, some
  nonzero fraction is an expected, literature-confirmed property of a
  graded quality mesh, not a defect to block on. mesh2d/fvgeometry.py's
  cv_area_floor is what keeps the SOLVER safe against whatever obtuse
  residual remains, rather than pretending the mesh generator can make it
  exactly zero.
"""
import warnings

import numpy as np


OBTUSE_TOL_DEG = 1e-6


def _triangle_angles_deg(points, simplex):
    a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
    angles = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        v1, v2 = q - p, r - p
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    return angles


def mesh_quality_report(points, triangles):
    """Returns a dict: n_triangles, n_obtuse (max angle > 90 deg),
    obtuse_fraction, worst_max_angle_deg, worst_min_angle_deg."""
    all_angles = np.array([_triangle_angles_deg(points, s) for s in triangles])
    max_angles = all_angles.max(axis=1)
    min_angles = all_angles.min(axis=1)
    # A right angle computed in floating point comes out as 90 +- ~1e-11
    # deg; a quadtree mesh is ALL right/45-degree angles, so without a
    # tolerance every other triangle would be miscounted as obtuse.
    n_obtuse = int(np.sum(max_angles > 90.0 + OBTUSE_TOL_DEG))
    return {
        "n_triangles": len(triangles),
        "n_obtuse": n_obtuse,
        "obtuse_fraction": n_obtuse / max(1, len(triangles)),
        "worst_max_angle_deg": float(max_angles.max()) if len(max_angles) else 0.0,
        "worst_min_angle_deg": float(min_angles.min()) if len(min_angles) else 90.0,
    }


def check_mesh_quality(points, triangles, min_angle_floor_deg=15.0, forbid_obtuse=False):
    """Mandatory pre-flight gate - raises if the mesh violates the one
    thing that IS a real bug (a triangle far below `triangle`'s own
    quality target), warns (every run, not just when something looks
    wrong) about the informational obtuse-fraction residual.

    forbid_obtuse=True (used for the quadtree mesh style, which is non-
    obtuse by construction - see mesh2d/quadtree.py) turns any obtuse
    triangle into a hard error instead of a warning."""
    report = mesh_quality_report(points, triangles)
    if forbid_obtuse and report["n_obtuse"]:
        raise RuntimeError(
            f"mesh2d: {report['n_obtuse']} obtuse triangle(s) (worst angle "
            f"{report['worst_max_angle_deg']:.4f} deg) in a mesh style that must be non-obtuse "
            "by construction - a mesh-generation bug.")
    if report["worst_min_angle_deg"] < min_angle_floor_deg:
        raise RuntimeError(
            f"mesh2d: a triangle with minimum angle {report['worst_min_angle_deg']:.1f} deg "
            f"(below the {min_angle_floor_deg} deg floor) survived triangle's own quality "
            "refinement - a real mesh-generation bug (see mesh2d/pointcloud.py's 'q' flag), "
            "not an expected residual; investigate before trusting this mesh.")
    if report["n_obtuse"]:
        warnings.warn(
            f"mesh2d: {report['n_obtuse']}/{report['n_triangles']} triangles "
            f"({report['obtuse_fraction']:.1%}) are obtuse (worst angle "
            f"{report['worst_max_angle_deg']:.1f} deg) - an expected, literature-confirmed "
            "residual of graded quality meshing (see mesh2d/mesh_quality.py's module "
            "docstring), mitigated by mesh2d/fvgeometry.py's cv_area_floor, not eliminated.")
    return report
