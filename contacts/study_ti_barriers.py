"""Study A (plans/schottky_plan.md section 7): one Ti / TiN / TiSix contact
metal on both S/D types, and NiSi (the PMOS-leaning silicide) for
comparison -

  NMOS: n+ Si:P 7e20 (electron barrier Phi_Bn),
  PMOS: p+ strained Si0.5Ge0.5:B 5e20 on Si (hole barrier Phi_Bp).

How far is the plain work-function (Schottky-Mott) barrier from the
measured one, how far does Fermi-level pinning take it, and what contact
resistivity does each give (1D Schottky model: thermionic emission +
nonlocal tunneling)?

Run: .venv/bin/python -m contacts.study_ti_barriers [--out out/schottky]
"""
import argparse
import os
import numpy as np

from core.params import Material
from core.materials import AlloyMaterial, resolve_material
from core import material_db as db
from contacts.schottky import SchottkyContact
from contacts import newton_schottky_1d as ns

NMOS_DOPING = 7e20
PMOS_DOPING = -5e20
L_DEVICE = 30e-7        # cm, uniform-doped slab behind the contact
DV = 1e-3               # V, for rho_c = dV/dJ at V -> 0

# Work functions (eV): the material DB values; TiN's depends strongly on
# its composition/process (4.2-4.9 reported), the DB's 4.6 is used.
METALS = {"Ti": 4.33, "TiN": 4.6, "NiSi": 4.65}

# Cowley-Sze pinning fit for Si (Sze & Ng, Physics of Semiconductor
# Devices, 3rd ed., ch. 3): Phi_Bn = c2 (Phi_m - chi) + (1 - c2)(Eg - phi0),
# c2 = 0.27 (the pinning factor S), phi0 = 0.30 eV above Ev (the neutral
# level). TO VERIFY against the book's table.
S_SI = 0.27
PHI0_ABOVE_EV = 0.30

# Measured barriers on n-Si (eV), literature (see the plan / DEVELOPMENT_LOG):
#   TiN 0.49 (Wittmer & Melchior), 0.35-0.58 in other reports;
#   TiSix/n-Si(100) ~0.69, ~0.61 under a TiN cap;
#   NiSi 0.65-0.66 (Ni2Si/NiSi forward I-V, IBM; ~0.65 e / ~0.45 h,
#   Gupta et al.), up to 0.70 in other reports.
MEASURED_N_SI = {"TiN": (0.35, 0.58), "TiSix": (0.61, 0.69), "NiSi": (0.65, 0.70)}
# Measured hole barrier on p-Si: NiSi ~0.45 eV (= Eg - Phi_Bn, near-midgap
# pinning). Used to check the Eg - Phi_Bn transfer below.
MEASURED_P_SI = {"NiSi": 0.45}
# Measured rho_c (Ohm cm^2) and the doping it was measured at:
# Ti-silicided Si:P 2e21: 1.5e-9 (imec); TiSi on high-Ge SiGe:B ~1-1.5e-9
# (patent) and 5e-9 (thesis), doping not given: mapped at our 5e20.
MEASURED_RHO = {"NMOS": (1.5e-9, 1.5e-9, 2e21), "PMOS": (1.0e-9, 5e-9, -5e20)}


def materials():
    si = resolve_material(db.get("Silicon"))
    sige_props = db.derive_alloy("Si0.5Ge0.5_strained",
                                 AlloyMaterial("SixGe1-x", "Silicon", "Germanium", bowing_eV=0.36),
                                 0.5, strained_on="Silicon")
    return si, resolve_material(sige_props)


def rho_c(mat, doping, contact):
    return ns.contact_resistivity(mat, doping, contact, L=L_DEVICE, dV=DV)


def barrier_table(si, sige):
    """Phi_Bn on Si and Phi_Bp on SiGe for each metal and model."""
    cnl_si = si.chi_eV + si.Eg_eV - PHI0_ABOVE_EV      # neutral level below vacuum
    rows = []
    for metal, wf in METALS.items():
        for model in ("schottky_mott", "pinned"):
            kw = dict(workfunction_eV=wf, barrier_model=model, pinning_S=S_SI, cnl_eV=cnl_si)
            c = SchottkyContact(**kw)
            bn_si, _ = c.barriers(si.chi_eV, si.Eg_eV)
            # SiGe: strained Si0.5Ge0.5 keeps Si's chi (Delta Ec ~ 0); the
            # neutral level is assumed fixed relative to Ec (the measured
            # p-SiGe barriers fall by ~Delta Ev with Ge, i.e. Phi_Bn stays).
            _, bp_sige = c.barriers(sige.chi_eV, sige.Eg_eV)
            rows.append(dict(metal=metal, model=model, wf=wf, phi_bn_si=bn_si, phi_bp_sige=bp_sige))
    return rows


def measured_rows(si, sige):
    """Start from each silicide's measured n-Si barrier (mid-range) and
    carry it to SiGe as Phi_Bp = Eg(SiGe) - Phi_Bn: the pinned level stays
    put relative to Ec (strained Si0.5Ge0.5 keeps Si's chi), so the whole
    band-gap shrink lands on the hole barrier."""
    rows = []
    for metal in ("TiN", "TiSix", "NiSi"):
        a, b = MEASURED_N_SI[metal]
        bn = 0.5 * (a + b)
        rows.append(dict(metal=metal, model="measured", wf=np.nan, phi_bn_si=bn,
                         phi_bp_si=si.Eg_eV - bn, phi_bp_sige=sige.Eg_eV - bn))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/schottky")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    si, sige = materials()
    print(f"Si:   chi {si.chi_eV:.3f}  Eg {si.Eg_eV:.3f}  Nc {si.Nc:.2e}  Nv {si.Nv:.2e}")
    print(f"SiGe: chi {sige.chi_eV:.3f}  Eg {sige.Eg_eV:.3f}  Nc {sige.Nc:.2e}  Nv {sige.Nv:.2e}  (strained on Si)")

    # rho_c vs barrier for both S/D types: the curve every barrier estimate sits on.
    phis = np.round(np.arange(0.05, 0.851, 0.05), 3)
    curves = {}
    for dev, mat, dop in (("NMOS", si, NMOS_DOPING), ("PMOS", sige, PMOS_DOPING)):
        key = "barrier_n_eV" if dop > 0 else "barrier_p_eV"
        curves[dev] = np.array([rho_c(mat, dop, SchottkyContact(**{key: float(p)})) for p in phis])

    print("\nBarrier and rho_c per metal and model "
          f"(NMOS Si:P {NMOS_DOPING:.0e}, PMOS Si0.5Ge0.5:B {-PMOS_DOPING:.0e})")
    print(f"{'metal':6s} {'model':14s} {'Phi_m':>6s} | {'Phi_Bn(Si)':>10s} {'rho_c NMOS':>11s} | "
          f"{'Phi_Bp(SiGe)':>12s} {'rho_c PMOS':>11s}")
    rows = barrier_table(si, sige) + measured_rows(si, sige)
    for r in rows:
        rn = rho_c(si, NMOS_DOPING, SchottkyContact(barrier_n_eV=max(r["phi_bn_si"], 0.0)))
        rp = rho_c(sige, PMOS_DOPING, SchottkyContact(barrier_p_eV=max(r["phi_bp_sige"], 0.0)))
        r.update(rho_n=rn, rho_p=rp)
        wf = "  -   " if np.isnan(r["wf"]) else f"{r['wf']:6.2f}"
        print(f"{r['metal']:6s} {r['model']:14s} {wf} | {r['phi_bn_si']:10.3f} {rn:11.2e} | "
              f"{r['phi_bp_sige']:12.3f} {rp:11.2e}")
    print("\nMeasured on n-Si: " + ", ".join(f"{k} {a:.2f}-{b:.2f} eV" for k, (a, b) in MEASURED_N_SI.items()))
    for k, bp in MEASURED_P_SI.items():
        a, b = MEASURED_N_SI[k]
        print(f"Measured on p-Si: {k} {bp:.2f} eV; Eg(Si) - Phi_Bn gives "
              f"{si.Eg_eV - b:.2f}-{si.Eg_eV - a:.2f} eV")
    mats = {"NMOS": si, "PMOS": sige}
    meas_curves = {}
    for dev in ("NMOS", "PMOS"):
        lo, hi, dop_meas = MEASURED_RHO[dev]
        key = "barrier_n_eV" if dop_meas > 0 else "barrier_p_eV"
        c = curves[dev] if dop_meas in (NMOS_DOPING, PMOS_DOPING) else \
            np.array([rho_c(mats[dev], dop_meas, SchottkyContact(**{key: float(p)})) for p in phis])
        meas_curves[dev] = c
        # Barrier range the measured rho_c implies (rho_c rises with the barrier).
        b_lo = np.interp(np.log(lo), np.log(c), phis)
        b_hi = np.interp(np.log(hi), np.log(c), phis)
        print(f"{dev}: measured rho_c {lo:.1e}-{hi:.1e} Ohm cm^2 at {abs(dop_meas):.0e} "
              f"-> barrier {b_lo:.2f}-{b_hi:.2f} eV in this model")

    np.savez(os.path.join(args.out, "study_ti_barriers.npz"), phis=phis, **{f"rho_{k}": v for k, v in curves.items()})
    plot(args.out, phis, curves, meas_curves, rows)


def plot(out, phis, curves, meas_curves, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    blue, orange, green, ink, muted = "#2a78d6", "#eb6834", "#1b9e5a", "#0b0b0b", "#52514e"
    colors = {"schottky_mott": blue, "pinned": orange, "measured": green}
    markers = {"Ti": "o", "TiN": "s", "NiSi": "^", "TiSix": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    panels = ((axes[0], "NMOS", "phi_bn_si", "rho_n", "NMOS: n+ Si:P", NMOS_DOPING),
              (axes[1], "PMOS", "phi_bp_sige", "rho_p", "PMOS: p+ Si0.5Ge0.5:B (strained)", -PMOS_DOPING))
    for ax, dev, bkey, rkey, title, dop in panels:
        ax.semilogy(phis, curves[dev], color=ink, lw=2)
        ax.annotate(f"{dop:.0e}", (phis[-1], curves[dev][-1]), textcoords="offset points", xytext=(4, 0),
                    fontsize=8, color=ink, va="center")
        lo, hi, dop_meas = MEASURED_RHO[dev]
        if meas_curves[dev] is not curves[dev]:
            ax.semilogy(phis, meas_curves[dev], color=ink, lw=1.2, ls="--")
            ax.annotate(f"{abs(dop_meas):.0e}", (phis[-1], meas_curves[dev][-1]), textcoords="offset points",
                        xytext=(4, 0), fontsize=8, color=ink, va="center")
        ax.axhspan(lo * 0.95, hi * 1.05, color=muted, alpha=0.15, lw=0)
        ax.text(0.98, hi * 1.12 if dev == "NMOS" else lo * 1.08, f"measured rho_c ({abs(dop_meas):.0e})",
                color=muted, fontsize=8, ha="right", transform=ax.get_yaxis_transform())
        if dev == "NMOS":
            for (k, (a, b)), y in zip(MEASURED_N_SI.items(), (0.97, 0.97, 0.80)):
                ax.axvspan(a, b, color=muted, alpha=0.08, lw=0)
                ax.text(b + 0.01 if k == "NiSi" else (a + b) / 2, y, f"measured\n{k}", color=muted, fontsize=8,
                        ha="left" if k == "NiSi" else "center", va="top", transform=ax.get_xaxis_transform())
        for r in rows:
            col, mk = colors[r["model"]], markers[r["metal"]]
            xb = max(r[bkey], 0.0)
            ax.semilogy(xb, r[rkey], mk, ms=8, color=col, mec="white", mew=1.5, zorder=3)
        ax.set_title(title, fontsize=10, color=ink)
        ax.set_xlabel("barrier to the majority carrier (eV)", color=ink)
        ax.set_xlim(-0.02, 0.95)
        ax.grid(True, which="major", color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("specific contact resistivity (Ohm cm$^2$)", color=ink)
    axes[1].legend(handles=[Line2D([], [], color=ink, lw=2, label="1D model at the device doping"),
                            Line2D([], [], color=ink, lw=1.2, ls="--", label="1D model at the measured doping"),
                            Line2D([], [], marker="o", ls="", color=blue, label="Schottky-Mott"),
                            Line2D([], [], marker="o", ls="", color=orange, label="pinned, Cowley-Sze"),
                            Line2D([], [], marker="o", ls="", color=green,
                                   label="measured n-Si barrier (SiGe: Eg - Phi_Bn)"),
                            *[Line2D([], [], marker=m, ls="", color=muted, label=k) for k, m in markers.items()]],
                   fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    path = os.path.join(out, "study_ti_barriers.png")
    fig.savefig(path, dpi=150)
    print("figure:", path)


if __name__ == "__main__":
    main()
