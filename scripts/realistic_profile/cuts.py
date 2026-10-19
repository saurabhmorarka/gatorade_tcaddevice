import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fab import flow
COL={"gas":"#ffffff","substrate":"#9ecae1","sheets":"#6baed6","sti":"#c7e9c0","spacer":"#fdae6b","inner_spacer":"#fd8d3c",
     "gate_oxide":"#636363","gate":"#ffd92f","dummy_gate":"#e5c494","ild":"#f0f0f0","sd_epi":"#e7298a","sd_epi_seed":"#7570b3",
     "source_contact":"#66a61e","drain_contact":"#66a61e","bdi":"#a6761d"}
out=sys.argv[1]; cases=[("fab_finfet_pmos","ideal"),("fab_finfet_pmos","realistic"),("fab_nanosheet_pmos","ideal"),("fab_nanosheet_pmos","realistic")]
fig,axs=plt.subplots(4,3,figsize=(15,22))
for row,(c,prof) in enumerate(cases):
    cfg=flow.load(f"configs/{c}.yaml"); cfg["params"]["profile"]=prof
    if "finfet" in c and prof=="realistic": cfg["params"]["voxel"]=[0.5,0.5,0.25]
    st,_=flow.build(cfg, log=lambda *a: None); lab=st.cell_labels(); g=st.grid
    names=["gas"]+[r.name for r in st.regions]
    cmap=ListedColormap([COL.get(n,"#999999") for n in names])
    L=lab+1
    xs,ys,zs=[np.r_[g.axes[a]] for a in range(3)]
    ext_xy=[xs[0],xs[-1],-ys[-1],-ys[0]]; ext_yz=[zs[0],zs[-1],-ys[-1],-ys[0]]
    iz=lab.shape[2]//2; ixc=int(np.argmin(np.abs(xs-2.0))); ixs=int(np.argmin(np.abs(xs-9.0)))
    kw=dict(cmap=cmap,vmin=0,vmax=len(names)-1,interpolation="nearest",aspect="equal",origin="upper")
    axs[row,0].imshow(L[:,:,iz].T,extent=ext_xy,**kw); axs[row,0].set_title(f"{c} {prof}: x-y at the fin/stack centre",fontsize=9)
    axs[row,1].imshow(L[ixc,:,:],extent=ext_yz,**kw); axs[row,1].set_title("y-z at x=2 nm (S/D centre, through the contact)",fontsize=9)
    axs[row,2].imshow(L[ixs,:,:],extent=ext_yz,**kw); axs[row,2].set_title("y-z at x=9 nm (2.5 nm from the channel end)",fontsize=9)
    for a in axs[row]: a.set_ylim(-80,25); a.set_xlabel("nm"); a.set_ylabel("height nm")
    # heights along the centre column under the gate
    ixg=lab.shape[0]//2; col=lab[ixg,:,iz]; hc=-(ys[:-1]+g.h[1]/2)
    def span(nm):
        k=st.region_index(nm); m=col==k
        return (round(hc[m].min()-g.h[1]/2,2), round(hc[m].max()+g.h[1]/2,2)) if m.any() else None
    print(c,prof,{n:span(n) for n in ("gate","gate_oxide","substrate","sheets","sti")})
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=v,label=k) for k,v in COL.items() if k!="gas"],loc="lower center",ncol=8)
fig.savefig(out,dpi=70,bbox_inches="tight")
