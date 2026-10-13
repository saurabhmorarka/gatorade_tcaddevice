"""Device templates: layer 1 (a template name + parameters) expanded into
layer 2 (an explicit process flow) and layer 3 (the device definition).

Every number in a generated flow is an expression over the parameters, so
the dumped layer-2 YAML (python -m fab expand) stays readable and editable,
and changing a parameter in it still propagates.
"""

# --- planar bulk MOSFET -----------------------------------------------------------
# Finite width with STI on both sides (always: a planar device without its
# STI edges is not the real device). x: source -> drain, z: width, heights
# up from the original silicon surface. No halo implant (modern flows drop
# it; add one in the flow if a study needs it).
PLANAR_DEFAULTS = {
    "type": "nmos",
    # gate: a real conductor built by the flow; its whole surface on the
    # dielectrics (oxide, spacers, STI under the endcaps) is the gate terminal.
    # Default gate-first TiN; "Poly" gives a doped-poly gate (n+ for nmos).
    "gate_material": "TiN",
    "gate_wf": None,        # eV; None: band-edge-like default for the type (see planar())
    "contact_material": "TiN",  # source/drain contact metal (full volumes)
    # Source/drain contacts are Schottky by default (real devices: Ti/TiN on
    # degenerate epi; plans/schottky_plan.md): thermionic emission + tunneling,
    # as a sub-grid boundary condition (contacts/subgrid.py). "ohmic" = ideal.
    "contact_type": "schottky",
    # The metal's electron barrier (eV). 0.45: what the measured rho_c of
    # Ti(Si)/TiN on Si:P implies in this model (README, Study A). A p+
    # contact gets Eg - Phi_Bn of its own semiconductor (pinned to Ec).
    "contact_barrier_n_eV": 0.45,
    # Low-resistance contact engineering (real flows; plans/schottky_plan.md 9):
    # the contact trench is etched this far into the S/D silicon so the metal
    # also contacts the recess sidewalls (nm; 0 = a top contact only) ...
    "contact_recess": 5,
    # ... and a shallow high-dose contact implant through the opening keeps
    # the silicon degenerate where the metal touches (the S/D implant alone
    # leaves ~4e19 at the surface: rho_c ~7e-7 Ohm cm^2). peak 0 = off.
    "ct_peak": 7e20, "ct_rp": 2, "ct_sigma": 3, "ct_lateral": 2,
    "Lg": 30,               # gate length
    "tox": 1.5,             # gate oxide
    "gate_height": 40,
    "spacer": 8,            # spacer (SiN) thickness
    "L_sd": 40,             # source/drain length beyond the spacer
    "W": 60,                # active (channel) width
    "sti_width": 20,        # STI on each side of the active area
    "sti_depth": 50,
    "sti_overfill": 10,     # STI oxide fill above silicon before CMP
    "sub_depth": 100,       # silicon below the surface in the domain
    "endcap": 15,           # gate extension over the STI
    "contact_inset": 5,     # contact trench kept off the spacer and STI edge (ILD in between)
    "ild_material": "SiO2", # interlayer dielectric around the gate and contacts
    "ild_cap": 10,          # ILD above the gate top; the contacts rise to the ILD top, above the gate
    "ild_overfill": 5,      # ILD deposited this far above its final top, then CMP
    "top_margin": 4,        # gas above everything deposited
    "voxel": [1.0, 0.5, 1.0],
    "sub_conc": 1e15,       # substrate doping, cm^-3
    "well_peak": 3e18, "well_rp": 25, "well_sigma": 15,
    # implants as analytic Gaussians (peak cm^-3, Rp/sigma/lateral nm), tuned
    # for a ~30 nm gate: extension junction ~11 nm, deep S/D junction ~30 nm
    "ext_peak": 5e19, "ext_rp": 4, "ext_sigma": 3, "ext_lateral": 3,
    "sd_peak": 1e20, "sd_rp": 12, "sd_sigma": 7, "sd_lateral": 5,
}

PLANAR_DERIVED = {
    "Lx": "2*(L_sd + spacer) + Lg",
    "Lz": "W + 2*sti_width",
    "xg0": "L_sd + spacer",
    "xg1": "L_sd + spacer + Lg",
    "zw0": "sti_width",
    "zw1": "sti_width + W",
    "ild_top": "tox + gate_height + ild_cap",
}


def planar(params):
    """Layer 2 + 3 for the planar template (params: layer-1 values merged
    over PLANAR_DEFAULTS)."""
    p = dict(PLANAR_DEFAULTS, **params)
    p.update({k: v for k, v in PLANAR_DERIVED.items() if k not in params})
    if p["type"] not in ("nmos", "pmos"):
        raise ValueError(f"planar type must be nmos or pmos, not {p['type']!r}")
    n = p["type"] == "nmos"
    gm, cmat, ild = p["gate_material"], p["contact_material"], p["ild_material"]
    if p["gate_wf"] is None:                 # n+/p+ poly at the band edges; metals a little inside
        p["gate_wf"] = (4.05 if n else 5.17) if gm == "Poly" else (4.25 if n else 4.95)
    sub_sp, sd_sp = ("B", "As") if n else ("P", "B")
    process = [
        {"step": "substrate", "label": "substrate", "material": "Si", "name": "substrate"},
        {"step": "dope", "label": "substrate doping", "species": sub_sp, "conc": "sub_conc", "materials": ["Si"]},
        {"step": "pattern", "label": "active mask", "name": "active", "mask": {"rect": [-1, "Lx + 1", "zw0", "zw1"]}},
        {"step": "etch", "label": "STI trench etch", "mode": "anisotropic", "materials": ["Si"], "mask": "!active",
         "depth": "sti_depth"},
        {"step": "deposit", "label": "STI oxide fill", "mode": "fill", "material": "SiO2", "name": "sti",
         "to_height": "sti_overfill"},
        {"step": "planarize", "label": "STI CMP", "to_height": 0},
        {"step": "implant", "label": "well / Vt implant", "species": sub_sp, "peak": "well_peak", "rp": "well_rp",
         "sigma": "well_sigma"},
        # gate dielectric over the whole (flat, post-CMP) surface: on the
        # active area only it would leave the gate 1.5 nm lower over the
        # STI - a ledge at every active edge
        {"step": "deposit", "label": "gate dielectric", "mode": "conformal", "material": "SiO2", "name": "gate_oxide",
         "thickness": "tox"},
        {"step": "deposit", "label": "gate metal deposition", "mode": "fill", "material": gm, "name": "gate",
         "to_height": "tox + gate_height"},
        {"step": "pattern", "label": "gate mask", "name": "gate",
         "mask": {"rect": ["xg0", "xg1", "zw0 - endcap", "zw1 + endcap"]}},
        {"step": "etch", "label": "gate etch", "mode": "anisotropic", "materials": [gm], "mask": "!gate",
         "stop_on": ["SiO2"]},
        # strip the exposed gate dielectric only: etching all SiO2 also
        # recesses the STI below the silicon, exposing an active-edge step
        # that the spacer etch then leaves nitride stringers in
        {"step": "etch", "label": "gate dielectric strip", "mode": "anisotropic", "regions": ["gate_oxide"],
         "depth": "1.5*tox"},
        {"step": "implant", "label": "extension implant", "species": sd_sp, "peak": "ext_peak", "rp": "ext_rp",
         "sigma": "ext_sigma", "lateral_sigma": "ext_lateral", "blocked_by": [gm]},
        {"step": "deposit", "label": "spacer deposition", "mode": "conformal", "material": "SiN", "name": "spacer",
         "thickness": "spacer"},
        {"step": "etch", "label": "spacer etch", "mode": "anisotropic", "materials": ["SiN"], "depth": "1.15*spacer"},
        {"step": "implant", "label": "source/drain implant", "species": sd_sp, "peak": "sd_peak", "rp": "sd_rp",
         "sigma": "sd_sigma", "lateral_sigma": "sd_lateral", "blocked_by": [gm, "SiN"]},
        # ILD: fill around the gate stack, CMP to ild_cap above the gate.
        # Spacer and contacts are never separated by gas, and the contacts
        # come out taller than the gate - their sidewalls facing the gate
        # through spacer + ILD are much of the off-state capacitance.
        {"step": "deposit", "label": "ILD deposition", "mode": "fill", "material": ild, "name": "ild",
         "to_height": "ild_top + ild_overfill"},
        {"step": "planarize", "label": "ILD CMP", "to_height": "ild_top"},
        {"step": "pattern", "label": "contact masks", "name": "source_ct",
         "mask": {"rect": [-1, "xg0 - spacer - contact_inset", "zw0 + contact_inset", "zw1 - contact_inset"]}},
        {"step": "pattern", "label": "contact masks", "name": "drain_ct",
         "mask": {"rect": ["xg1 + spacer + contact_inset", "Lx + 1", "zw0 + contact_inset", "zw1 - contact_inset"]}},
        # contact trenches through the ILD only (down to the silicon; spacer, STI untouched)
        {"step": "etch", "label": "source contact etch", "mode": "anisotropic", "regions": ["ild"], "mask": "source_ct"},
        {"step": "etch", "label": "drain contact etch", "mode": "anisotropic", "regions": ["ild"], "mask": "drain_ct"},
    ]
    for side in ("source", "drain"):
        if p["contact_recess"]:
            process.append({"step": "etch", "label": f"{side} contact recess (into the S/D)", "mode": "anisotropic",
                            "materials": ["Si"], "mask": f"{side}_ct", "depth": "contact_recess"})
        if p["ct_peak"]:
            # blocked by the ILD only: just outside the opening the profile
            # is measured from the silicon top, so the lateral straggle dopes
            # the upper recess sidewalls too
            process.append({"step": "implant", "label": f"{side} contact implant", "species": sd_sp,
                            "peak": "ct_peak", "rp": "ct_rp", "sigma": "ct_sigma", "lateral_sigma": "ct_lateral",
                            "blocked_by": [ild], "mask": f"{side}_ct"})
    process += [
        {"step": "deposit", "label": "source contact fill", "mode": "fill", "material": cmat, "name": "source_contact",
         "to_height": "ild_top", "mask": "source_ct"},
        {"step": "deposit", "label": "drain contact fill", "mode": "fill", "material": cmat, "name": "drain_contact",
         "to_height": "ild_top", "mask": "drain_ct"},
    ]
    # headroom above the gate for the spacer liner (a deposit must not reach
    # the domain ceiling: the etches see gas only from above)
    domain = {"x": [0, "Lx"], "z": [0, "Lz"], "height": ["-sub_depth", "ild_top + ild_overfill + top_margin"],
              "voxel": "voxel"}
    # layer 3: terminals are process regions (whole interfaces), with a type:
    # "schottky" (barrier + tunneling at the metal/epi interface, default for
    # the source/drain) or "ohmic"
    sd = {"type": p["contact_type"]}
    if p["contact_type"] == "schottky":
        sd["barrier_n_eV"] = "contact_barrier_n_eV"
    device = {"contacts": [
        {"name": "source", "region": "source_contact", **sd},
        {"name": "drain", "region": "drain_contact", **sd},
        {"name": "gate", "region": "gate", "type": "gate", "workfunction_eV": "gate_wf"},
        {"name": "body", "surface": "bottom", "type": "ohmic"},
    ]}
    return {"params": p, "domain": domain, "process": process, "device": device}


TEMPLATES = {"planar": planar}
