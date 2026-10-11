"""Write fab structures as VTK images (.vti) plus a .pvd over the process
steps, for tcadview's process-step slider (and ParaView).

Each snapshot holds, on the voxel grid (coordinates in um, as the device
results):
  cell data   region_id, material_id (-1 = gas), per voxel
  point data  ls_<region>: the region's level set, clipped to +-127 steps of
              `levelset_scale` um and stored as int8 (compresses to almost
              nothing, and is all a smooth 0-contour needs);
              DopingConcentration / DonorConcentration / AcceptorConcentration
              (cm^-3, NaN outside semiconductors), once anything is doped
  field data  region_names, material_names, label (step name), kind
              ("process"), step index, contacts, field_units
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.vtk_io import write_pvd, write_vti

DOPING_UNITS = ["DopingConcentration:cm^-3", "DonorConcentration:cm^-3", "AcceptorConcentration:cm^-3"]


def write_snapshot(path, st, label, index):
    write_vti(path, **snapshot_arrays(st, label, index))


def snapshot_arrays(st, label, index):
    """Everything write_vti needs for one snapshot, as arrays owned by the
    snapshot (the structure keeps changing after this returns)."""
    g = st.grid
    qs = g.h.min() / 16.0                                   # level-set quantum, nm
    regions = st.regions
    materials = sorted({r.material for r in regions})
    cell = st.cell_labels()
    rmat = np.array([materials.index(r.material) for r in regions] + [-1], dtype=np.int16)
    point = {f"ls_{r.name}": np.clip(np.round(r.phi / qs), -127, 127).astype(np.int8) for r in regions}
    units = []
    if st.donors.any() or st.acceptors.any():
        labels = st.node_labels()
        nd, na, net = st.net_doping(labels)
        semi = st.semiconductor_nodes(labels)
        nan = np.float32(np.nan)                        # undefined outside semiconductors (grey in the viewer)
        point.update(DopingConcentration=np.where(semi, net, nan).astype(np.float32),
                     DonorConcentration=np.where(semi, nd, nan).astype(np.float32),
                     AcceptorConcentration=np.where(semi, na, nan).astype(np.float32))
        units = DOPING_UNITS
    contacts = [c["name"] + ":" + (c.get("region") or c.get("surface", "")) for c in st.contacts]
    field = {"region_names": [r.name for r in regions] or ["-"], "material_names": materials or ["-"],
             "label": label, "kind": "process", "step": [index], "levelset_scale": [qs * 1e-3],
             "contacts": contacts or ["-"], "field_units": units or ["-"]}
    return dict(origin=g.lo * 1e-3, spacing=g.h * 1e-3, shape=g.shape, point_data=point,
                cell_data={"region_id": cell, "material_id": rmat[cell]}, field_data=field)


class SnapshotWriter:
    """on_step callback for fab.flow.build: one .vti per step, then the .pvd.
    The arrays are taken on the build thread; compressing and writing them
    (zlib releases the GIL) runs on worker threads while the build goes on."""

    def __init__(self, out_dir, stem="build", every=True, workers=4):
        self.out_dir, self.stem, self.every = out_dir, stem, every
        self.entries = []
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.jobs = []
        os.makedirs(out_dir, exist_ok=True)

    def __call__(self, i, label, st):
        if not self.every:
            self.last = (i, label, st)
            return
        self._write(i, label, st)

    def _write(self, i, label, st):
        f = f"{self.stem}_{i:02d}.vti"
        arrays = snapshot_arrays(st, label, i)
        self.jobs.append(self.pool.submit(write_vti, os.path.join(self.out_dir, f), **arrays))
        self.entries.append({"file": f, "timestep": i, "name": f"{i:02d} {label}"})

    def finish(self):
        if not self.every and getattr(self, "last", None):
            self._write(*self.last)
        for j in self.jobs:
            j.result()                                       # re-raise any write error
        self.pool.shutdown()
        path = os.path.join(self.out_dir, f"{self.stem}.pvd")
        write_pvd(path, self.entries)
        return path
