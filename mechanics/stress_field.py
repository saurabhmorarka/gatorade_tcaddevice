"""A solved process-stress field on its voxel (3D) or pixel (2D) grid, as
the device solver samples it: the total stress per voxel (Voigt, GPa, the
Gauss-point mean of mechanics/fe2d.py / fe3d.py), the material of every
voxel, and the grid (lower corner `lo` and spacing `h`, nm, in the frame of
the structure it was solved on: x along the channel, y depth positive down,
z width).

sample(points_nm, materials) is a MASKED multilinear interpolation between
voxel centres: only voxels of the given materials (silicon, for the
piezoresistance of a silicon edge) take part, the weights renormalized, so
a point on a Si / oxide or Si / SiGe interface takes the silicon side's
stress and nothing from across the interface. A point whose whole stencil
is outside the material falls back to the nearest voxel of it.

Saved as .npz (save / load) so a `fab simulate --stress` run can reuse the
stress solve: the fab config, its parameters and the grid travel with the
field and are checked when it is loaded.
"""
import json

import numpy as np
from scipy import ndimage


class StressField:
    def __init__(self, sigma, lo, h, labels, materials, meta=None):
        """sigma (*shape, 6) GPa; lo (dim,) nm; h scalar nm; labels (*shape)
        int (0 = void); materials {label: material name}."""
        self.sigma = np.asarray(sigma, np.float64)
        self.labels = np.asarray(labels, np.int32)
        self.shape = self.labels.shape
        self.dim = self.labels.ndim
        if self.sigma.shape != self.shape + (6,):
            raise ValueError(f"stress {self.sigma.shape} does not match labels {self.shape} x 6")
        self.lo = np.asarray(lo, float).reshape(self.dim)
        self.h = float(h)
        self.materials = {int(k): str(v) for k, v in materials.items()}
        self.meta = dict(meta or {})
        self._nearest = {}

    # --- construction -------------------------------------------------------------
    @classmethod
    def from_planar_flow(cls, b, flow=None, par=None):
        """From a mechanics.planar_flow Builder after run_flow (2D, the final
        step's stress)."""
        from mechanics.planar_flow import MATS
        g = b.g
        sig = b.sh.total(b.sh.history[-1][1]).reshape(g.ny, g.nx, 6).transpose(1, 0, 2)
        mats = {k: v for k, v in MATS.items()}
        return cls(sig, (g.x0, g.y0), g.h, b.lab.T, mats,
                   meta=dict(source="planar_flow", flow=flow, par=par or {},
                             gate_x_nm=[float(b.xc - b.p["Lg"] / 2), float(b.xc + b.p["Lg"] / 2)]))

    @classmethod
    def from_fab_stress(cls, res, config=None, par=None, fab_set=None):
        """From a mechanics.fab_stress.run result (3D, the final state)."""
        g, sh = res["grid"], res["sh"]
        sig = sh.element_stress().reshape(g.shape + (6,))
        mats = {gid: mat for gid, (name, mat) in res["regions"].items()}
        names = {gid: name for gid, (name, mat) in res["regions"].items()}
        return cls(sig, g.lo, g.h, sh.labels, mats,
                   meta=dict(source="fab_stress", config=config, par=par or {}, fab_set=fab_set or {},
                             region_names={str(k): v for k, v in names.items()}))

    # --- io ------------------------------------------------------------------------
    def save(self, path):
        mats = json.dumps(self.materials)
        np.savez_compressed(path, sigma=self.sigma.astype(np.float32), labels=self.labels, lo=self.lo,
                            h=self.h, materials=mats, meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        mats = {int(k): v for k, v in json.loads(str(z["materials"])).items()}
        return cls(z["sigma"], z["lo"], float(z["h"]), z["labels"], mats, meta=json.loads(str(z["meta"])))

    # --- geometry ------------------------------------------------------------------
    def shifted(self, offset_nm):
        """The same field with its grid moved by offset_nm (dim,): device
        coordinate = field coordinate + offset."""
        out = StressField(self.sigma, self.lo + np.asarray(offset_nm, float), self.h, self.labels, self.materials,
                          dict(self.meta, offset_nm=list(map(float, np.atleast_1d(offset_nm)))))
        return out

    def material_mask(self, materials):
        keep = [k for k, v in self.materials.items() if v in set(materials)]
        return np.isin(self.labels, keep)

    def voxel_of(self, points_nm):
        """Index of the voxel containing each point (clipped to the grid)."""
        u = (np.asarray(points_nm, float)[:, :self.dim] - self.lo) / self.h
        return np.clip(np.floor(u).astype(np.int64), 0, np.array(self.shape) - 1)

    def inside(self, points_nm, tol=1e-6):
        p = np.asarray(points_nm, float)[:, :self.dim]
        hi = self.lo + np.array(self.shape) * self.h
        return np.all((p >= self.lo - tol) & (p <= hi + tol), axis=1)

    def _nearest_index(self, key, mask):
        if key not in self._nearest:
            _, idx = ndimage.distance_transform_edt(~mask, return_indices=True)
            self._nearest[key] = idx
        return self._nearest[key]

    def sample(self, points_nm, materials=("Si",)):
        """(P, 6) stress at the points (nm, (P, >= dim); extra coordinates
        ignored), masked to voxels of `materials` (see the module
        docstring)."""
        p = np.asarray(points_nm, float)[:, :self.dim]
        mask = self.material_mask(materials)
        if not mask.any():
            raise ValueError(f"the stress field has no voxels of {materials}")
        u = (p - self.lo) / self.h - 0.5                 # voxel-centre index coordinates
        i0 = np.floor(u).astype(np.int64)
        w = u - i0
        n = np.array(self.shape)
        acc = np.zeros((len(p), 6))
        wsum = np.zeros(len(p))
        for corner in np.ndindex(*(2,) * self.dim):
            c = np.array(corner)
            idx = i0 + c
            ok = np.all((idx >= 0) & (idx < n), axis=1)
            idx = np.clip(idx, 0, n - 1)
            wt = np.prod(np.where(c == 1, w, 1.0 - w), axis=1) * ok
            t = tuple(idx.T)
            wt = wt * mask[t]
            acc += wt[:, None] * self.sigma[t]
            wsum += wt
        out = np.empty_like(acc)
        good = wsum > 1e-12
        out[good] = acc[good] / wsum[good, None]
        if (~good).any():                                # nearest voxel of the material
            near = self._nearest_index(tuple(sorted(materials)), mask)
            v = tuple(self.voxel_of(p[~good]).T)
            src = tuple(near[a][v] for a in range(self.dim))
            out[~good] = self.sigma[src]
        return out
