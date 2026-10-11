"""Debug aid: list edges whose facet area differs between its two cells
(all labels, then device labels only), with node positions and labels.
Usage: python mesh3d/studies/facet_asymmetry_debug.py configs/fab_planar.yaml"""
import sys; sys.path.insert(0, ".")
import numpy as np, mesh3d.octree as O
from fab import flow
from fab.simulate import device_voxels, device_materials
st, _ = flow.build(flow.load(sys.argv[1]), log=lambda *a: None)
lab, net, origin, h = device_voxels(st, True); mats = device_materials(st)
cn = sum(net[i:net.shape[0]-1+i, j:net.shape[1]-1+j, k:net.shape[2]-1+k] for i in (0,1) for j in (0,1) for k in (0,1))/8
tgt = O.target_spacing(lab, [m["kind"] for m in mats], cn, h, (2,.5,2), (2,2,2), (8,8,8), 1.35)
T = O.BoxTree(lab, tgt, [16,32,16], semi_labels=[k for k,m in enumerate(mats) if m["kind"] in ("semi","conductor")]); T.refine()
ids, LO, HI = T.leaves(); ll = lab[LO[:,0],LO[:,1],LO[:,2]]
d = O._dual(T, ids, LO, HI, ll, lab, h, print)
FN, FJ, FL, FA = d["facet"]; N = d["N"]
lo, hi = np.minimum(FN, FJ), np.maximum(FN, FJ); side = (FN > FJ).astype(int)
key = (lo*N+hi)*2+side
uk, inv = np.unique(key, return_inverse=True); s = np.bincount(inv, weights=FA)
pk, sd = uk//2, uk%2
up, pinv = np.unique(pk, return_inverse=True)
A2 = np.full((len(up),2), np.nan); A2[pinv, sd] = s
both = ~np.isnan(A2).any(1)
rel = np.abs(A2[:,0]-A2[:,1])/np.nanmax(A2,1)
w = np.argsort(-np.where(both, rel, 0))[:10]
box = set(d["box_nodes"].tolist())
print("pairs with rel diff > 1e-6:", int((both & (rel > 1e-6)).sum()), "of", int(both.sum()))
for k in w:
    i, j = up[k]//N, up[k]%N
    print(d["node_idx"][i], "box" if i in box else "clip", d["node_idx"][j], "box" if j in box else "clip", A2[k], f"{rel[k]:.3f}")
kinds = np.array([m["kind"] for m in mats] + ["gas"])
VN, VL, VV = d["vol"]
devv = np.isin(kinds[VL], ("semi", "insulator"))
cv = np.bincount(VN, weights=VV*devv, minlength=N)
names = [m["name"] for m in mats] + ["gas"]
keepf = np.isin(kinds[FL], ("semi", "insulator"))
key2 = key[keepf]; uk2, inv2 = np.unique(key2, return_inverse=True); s2 = np.bincount(inv2, weights=FA[keepf])
pk2, sd2 = uk2//2, uk2%2; up2, pinv2 = np.unique(pk2, return_inverse=True)
B = np.full((len(up2),2), np.nan); B[pinv2, sd2] = s2
bothB = ~np.isnan(B).any(1); relB = np.abs(B[:,0]-B[:,1])/np.nanmax(B,1)
bad = np.flatnonzero(bothB & (relB > 1e-6))
print("device-label asym pairs:", len(bad))
for k in bad[:8]:
    i, j = up2[k]//N, up2[k]%N
    li = sorted({names[l] for l in FL[(FN==i)&(FJ==j)]}); lj = sorted({names[l] for l in FL[(FN==j)&(FJ==i)]})
    print(d["node_idx"][i], "dev" if cv[i]>0 else "nondev", li, "|", d["node_idx"][j], "dev" if cv[j]>0 else "nondev", lj, B[k])
