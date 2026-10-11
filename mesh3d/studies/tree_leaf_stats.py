"""Build only the octree box tree of a fab device and print leaf counts and
median sizes per material (where the nodes go). Usage: python
mesh3d/studies/tree_leaf_stats.py configs/fab_planar.yaml"""
import sys, time, itertools; sys.path.insert(0, ".")
import numpy as np
import mesh3d.octree as O
from fab import flow
from fab.simulate import device_voxels, device_materials
cfg = flow.load(sys.argv[1])
st, _ = flow.build(cfg, log=lambda *a: None)
lab, net, origin, h = device_voxels(st, True)
mats = device_materials(st)
kinds = [m["kind"] for m in mats]
cn = sum(net[i:net.shape[0]-1+i, j:net.shape[1]-1+j, k:net.shape[2]-1+k] for i in (0,1) for j in (0,1) for k in (0,1))/8
t = time.time()
tgt = O.target_spacing(lab, kinds, cn, h, (2,.5,2), (2,2,2), (8,8,8), 1.35)
tree = O.BoxTree(lab, tgt, [16,32,16], semi_labels=[k for k,m in enumerate(mats) if m["kind"] in ("semi","conductor")])
tree.refine(log=print)
ids, LO, HI = tree.leaves(); ll = lab[LO[:,0],LO[:,1],LO[:,2]]
corners = np.concatenate([np.stack([np.where(o[a], HI[:,a], LO[:,a]) for a in range(3)],1) for o in itertools.product((0,1),repeat=3)])
print(f"tree {time.time()-t:.0f}s, nodes {len(np.unique(corners, axis=0))}")
names = [m["name"] for m in mats] + ["gas"]
for l in np.unique(ll):
    sel = ll == l; sz = (HI-LO)[sel]*h
    print(f"  {names[l]:15s} leaves {sel.sum():7d}  median size {np.median(sz,0)} nm")
