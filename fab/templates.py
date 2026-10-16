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


# --- bulk FinFET -------------------------------------------------------------------------
# One fin in one gate pitch: x spans a contacted poly pitch (CPP) with the
# gate in the middle and half of each shared S/D contact at the ends, z one
# fin pitch with the fin in the middle (the side walls are symmetry planes:
# the neighbouring fins). Heights up from the fin top. Gate-first (same
# structure as a replacement-gate flow here); recessed fin + selective,
# in-situ doped S/D epitaxy (analytic profile, epi_doping); bulk with a
# punch-through stopper under the fin. 2 nm-node-class numbers: CPP 48 nm,
# Lg 15 nm, 5 nm spacer, fin 5 nm wide / 50 nm tall at a 24 nm pitch,
# EOT 0.85 nm.
FINFET_DEFAULTS = {
    "type": "nmos",
    "gate_material": "TiN",
    "gate_wf": None,          # eV; None: 4.40 (nmos) / 4.80 (pmos) (undoped channel: Vt set by the metal)
    "contact_material": "TiN",
    "contact_type": "schottky",
    "contact_barrier_n_eV": 0.45,
    "contact_recess": 5,      # contact trench into the epi (nm)
    "cpp": 48,                # contacted poly pitch = the x domain
    "Lg": 15,
    "spacer": 5,
    "fin_pitch": 24,          # the z domain
    "W_fin": 5,
    "H_fin": 50,              # fin height above the STI
    "sub_fin": 20,            # fin etch below the STI top (the fin base inside the STI)
    "sub_depth": 30,          # silicon below the fin etch bottom
    "tox": 2.0,               # gate dielectric (lumped IL + HfO2, EOT 0.85 nm: HKstack)
    "gate_dielectric": "HKstack",
    "gate_height": 30,        # gate metal above the fin top
    "contact_inset": 3,       # contact trench kept off the spacer (ILD in between)
    "ild_material": "SiO2",
    "ild_cap": 10, "ild_overfill": 5, "top_margin": 4,
    "voxel": [0.5, 0.5, 0.5],
    # device mesh: fine spacing (nm, x/y/z) at semiconductor interfaces and
    # junctions - the fin's gated walls face z, so z is fine too (the
    # planar default, 2 nm in z, put 5 nodes across a 5 nm fin)
    "mesh_h_fine": [1.0, 0.5, 0.5],
    "sub_conc": 1e15,         # substrate (and undoped fin) doping
    # punch-through stopper: peaked at the fin base, implanted before the
    # fin reveal (through the STI); the sub-fin under the gate is the
    # off-state leakage path without it
    "pts_peak": 3e18, "pts_sigma": 6,
    "sd_recess_stop": 5,      # the S/D fin recess stops this far above the STI (the epi stays above the PTS)
    # S/D epi: grown to epi_over above the fin top, in-situ doped peak
    # (NMOS Si:P 7e20, PMOS Si0.5Ge0.5:B 5e20); the tail outside the epi is
    # Gaussian with sx along the channel (the junction under the spacer) and
    # sy in depth
    "epi_over": 5, "epi_thickness": 18,
    "epi_peak": None, "epi_sx": 2.0, "epi_sy": 3.0,
}

FINFET_DERIVED = {
    "Lx": "cpp",
    "Lz": "fin_pitch",
    "xg0": "(cpp - Lg)/2",
    "xg1": "(cpp + Lg)/2",
    "zf0": "(fin_pitch - W_fin)/2",
    "zf1": "(fin_pitch + W_fin)/2",
    "W": "2*H_fin + W_fin",    # effective width per fin (currents are per um of it)
    "gate_top": "tox + gate_height",
    "ild_top": "tox + gate_height + ild_cap",
}


def _sd_epi_and_contacts(p, process, n, x_open_l, x_open_r):
    """S/D epitaxy (selective, from the exposed silicon, capped), its
    analytic doping, ILD, and the contact trenches + fill: shared by the
    FinFET and nanosheet flows. x_open_*: the S/D opening edges (outer
    spacer edges)."""
    epi_mat = "Si" if n else "SiGe"
    sd_sp = "P" if n else "B"
    cmat, ild = p["contact_material"], p["ild_material"]
    process += [
        {"step": "deposit", "label": "S/D epitaxy", "mode": "conformal", "material": epi_mat, "name": "sd_epi",
         "on": ["Si"], "thickness": "epi_thickness", "to_height": "epi_over", "mask": "!gate_sp"},
        {"step": "epi_doping", "label": "S/D epi in-situ doping", "species": sd_sp, "peak": "epi_peak",
         "sx": "epi_sx", "sy": "epi_sy", "regions": ["sd_epi"]},
        {"step": "deposit", "label": "ILD deposition", "mode": "fill", "material": ild, "name": "ild",
         "to_height": "ild_top + ild_overfill"},
        {"step": "planarize", "label": "ILD CMP", "to_height": "ild_top"},
        {"step": "pattern", "label": "contact masks", "name": "source_ct",
         "mask": {"rect": [-1, f"{x_open_l} - contact_inset", -1, "Lz + 1"]}},
        {"step": "pattern", "label": "contact masks", "name": "drain_ct",
         "mask": {"rect": [f"{x_open_r} + contact_inset", "Lx + 1", -1, "Lz + 1"]}},
        {"step": "etch", "label": "source contact etch", "mode": "anisotropic", "regions": ["ild"], "mask": "source_ct"},
        {"step": "etch", "label": "drain contact etch", "mode": "anisotropic", "regions": ["ild"], "mask": "drain_ct"},
    ]
    for side in ("source", "drain"):
        if p["contact_recess"]:
            process.append({"step": "etch", "label": f"{side} contact recess (into the epi)", "mode": "anisotropic",
                            "regions": ["sd_epi"], "mask": f"{side}_ct", "depth": "contact_recess"})
    process += [
        {"step": "deposit", "label": "source contact fill", "mode": "fill", "material": cmat, "name": "source_contact",
         "to_height": "ild_top", "mask": "source_ct"},
        {"step": "deposit", "label": "drain contact fill", "mode": "fill", "material": cmat, "name": "drain_contact",
         "to_height": "ild_top", "mask": "drain_ct"},
    ]


def _device_layer(p):
    sd = {"type": p["contact_type"]}
    if p["contact_type"] == "schottky":
        sd["barrier_n_eV"] = "contact_barrier_n_eV"
    return {"contacts": [
        {"name": "source", "region": "source_contact", **sd},
        {"name": "drain", "region": "drain_contact", **sd},
        {"name": "gate", "region": "gate", "type": "gate", "workfunction_eV": "gate_wf"},
        {"name": "body", "surface": "bottom", "type": "ohmic"},
    ]}


def _merge(defaults, derived, params, kind):
    p = dict(defaults, **params)
    p.update({k: v for k, v in derived.items() if k not in params})
    if p["type"] not in ("nmos", "pmos"):
        raise ValueError(f"{kind} type must be nmos or pmos, not {p['type']!r}")
    n = p["type"] == "nmos"
    if p.get("gate_wf") is None:
        p["gate_wf"] = 4.40 if n else 4.80
    if p.get("epi_peak") is None:
        p["epi_peak"] = 7e20 if n else 5e20
    return p, n


def finfet(params):
    """Layer 2 + 3 for the bulk FinFET template."""
    p, n = _merge(FINFET_DEFAULTS, FINFET_DERIVED, params, "finfet")
    gm, hk = p["gate_material"], p["gate_dielectric"]
    sub_sp = "B" if n else "P"
    process = [
        {"step": "substrate", "label": "substrate", "material": "Si", "name": "substrate"},
        {"step": "dope", "label": "substrate doping", "species": sub_sp, "conc": "sub_conc", "materials": ["Si"]},
        {"step": "pattern", "label": "fin mask", "name": "fin", "mask": {"rect": [-1, "Lx + 1", "zf0", "zf1"]}},
        {"step": "etch", "label": "fin etch", "mode": "anisotropic", "materials": ["Si"], "mask": "!fin",
         "depth": "H_fin + sub_fin"},
        {"step": "deposit", "label": "STI oxide fill", "mode": "fill", "material": "SiO2", "name": "sti",
         "to_height": 5},
        {"step": "planarize", "label": "STI CMP (stop on the fin)", "to_height": 0},
        {"step": "implant", "label": "punch-through stopper", "species": sub_sp, "peak": "pts_peak",
         "rp": "H_fin + 3", "sigma": "pts_sigma"},
        {"step": "etch", "label": "fin reveal (STI recess)", "mode": "anisotropic", "regions": ["sti"],
         "depth": "H_fin"},
        {"step": "deposit", "label": "gate dielectric (IL + HfO2)", "mode": "conformal", "material": hk,
         "name": "gate_oxide", "thickness": "tox"},
        {"step": "deposit", "label": "gate metal deposition", "mode": "fill", "material": gm, "name": "gate",
         "to_height": "gate_top"},
        {"step": "pattern", "label": "gate mask", "name": "gate", "mask": {"rect": ["xg0", "xg1", -1, "Lz + 1"]}},
        {"step": "etch", "label": "gate etch", "mode": "anisotropic", "materials": [gm], "mask": "!gate",
         "stop_on": [hk]},
        # the whole exposed dielectric, fin side walls included
        {"step": "etch", "label": "gate dielectric strip", "mode": "anisotropic", "regions": ["gate_oxide"],
         "mask": "!gate"},
        {"step": "deposit", "label": "spacer deposition", "mode": "conformal", "material": "SiN", "name": "spacer",
         "thickness": "spacer"},
        {"step": "pattern", "label": "gate + spacer footprint", "name": "gate_sp",
         "mask": {"rect": ["xg0 - spacer", "xg1 + spacer", -1, "Lz + 1"]}},
        # spacer etch with the fin side walls cleared (over-etch outside the
        # gate + spacer footprint), then the cap off the gate top
        {"step": "etch", "label": "spacer etch (fin side walls cleared)", "mode": "anisotropic",
         "materials": ["SiN"], "mask": "!gate_sp"},
        {"step": "etch", "label": "spacer etch (gate top)", "mode": "anisotropic", "materials": ["SiN"],
         "depth": "1.15*spacer"},
        # on the fin only: an anisotropic front `depth` below an STI column's
        # top would reach the substrate under the STI
        {"step": "etch", "label": "S/D fin recess", "mode": "anisotropic", "materials": ["Si"],
         "mask": {"not": {"union": ["gate_sp", "!fin"]}}, "depth": "H_fin - sd_recess_stop"},
    ]
    _sd_epi_and_contacts(p, process, n, "xg0 - spacer", "xg1 + spacer")
    domain = {"x": [0, "Lx"], "z": [0, "Lz"], "height": ["-(H_fin + sub_fin + sub_depth)",
                                                          "ild_top + ild_overfill + top_margin"], "voxel": "voxel"}
    return {"params": p, "domain": domain, "process": process, "device": _device_layer(p)}


# --- nanosheet (gate-all-around) ------------------------------------------------------------
# One stack of n_sheets Si sheets in one CPP, replacement-metal-gate flow:
# Si/SiGe superlattice, stack etch + STI, dummy gate + outer spacer, S/D
# recess, SiGe lateral recess + inner spacers, S/D epi from the sheet ends,
# ILD + CMP to the dummy gate, dummy-gate pull, channel release (SiGe
# removed), conformal high-k wrapping every sheet, metal fill between the
# sheets. The substrate mesa under the stack is gated too (bottom
# parasitic channel), held off by the punch-through stopper. Heights up
# from the top sheet's top. 2 nm-node-class numbers: CPP 48 nm, Lg 15 nm,
# 3 sheets 5 nm thick / 25 nm wide, 10 nm SiGe (the suspension gap), 5 nm
# inner and outer spacers.
NANOSHEET_DEFAULTS = {
    "type": "nmos",
    "gate_material": "TiN",
    "gate_wf": None,
    "contact_material": "TiN",
    "contact_type": "schottky",
    "contact_barrier_n_eV": 0.45,
    "contact_recess": 5,
    "cpp": 48,
    "Lg": 15,
    "spacer": 5,              # outer spacer
    "inner_spacer": 5,        # SiGe lateral recess = inner spacer length
    "n_sheets": 3,
    "t_sheet": 5,
    "W_sheet": 25,
    "t_sige": 10,             # sacrificial SiGe = sheet-to-sheet gap after release
    "sige_material": "SiGe",
    "stack_pitch": 46,        # the z domain (stack-to-stack pitch)
    "sub_fin": 20,            # stack etch into the substrate (mesa height under the STI top)
    "sub_depth": 30,
    "tox": 2.0,
    "gate_dielectric": "HKstack",
    "gate_height": 30,        # gate metal above the top sheet
    "contact_inset": 3,
    "ild_material": "SiO2",
    "ild_cap": 10, "ild_overfill": 5, "top_margin": 4,
    "voxel": [0.5, 0.5, 0.5],
    "mesh_h_fine": [1.0, 0.5, 0.5],      # sheet side walls face z (see the FinFET)
    "sub_conc": 1e15,
    # Isolation of the substrate mesa under the stack (the S/D epi reaches
    # it, a punch-through path 5-20 nm under the bottom sheet):
    #   pts: a heavy punch-through stopper only (1e19, sharp: implanted
    #        before the superlattice in a real flow)
    #   bdi: bottom dielectric isolation - a sacrificial layer under the
    #        superlattice, removed under the dummy gate and refilled with
    #        SiN; the epi then meets only the sheet ends (light PTS 1e18)
    "isolation": "pts",
    "t_bdi": 10,
    "bdi_material": "SiN",
    "pts_peak": None, "pts_sigma": 4,       # None: 1e19 (pts) / 1e18 (bdi)
    "epi_over": 5, "epi_thickness": 18,
    "epi_peak": None, "epi_sx": 2.0, "epi_sy": 4.0,
}

NANOSHEET_DERIVED = {
    "Lx": "cpp",
    "Lz": "stack_pitch",
    "xg0": "(cpp - Lg)/2",
    "xg1": "(cpp + Lg)/2",
    "zs0": "(stack_pitch - W_sheet)/2",
    "zs1": "(stack_pitch + W_sheet)/2",
    "H_stack": "n_sheets*(t_sheet + t_sige)",
    "H_sl": "H_stack + t_bdi",               # superlattice (+ the BDI layer) on the substrate
    "W": "2*n_sheets*(W_sheet + t_sheet)",   # effective width (currents per um of it)
    "gate_top": "gate_height",
    "ild_top": "gate_height + ild_cap",
}


def nanosheet(params):
    """Layer 2 + 3 for the nanosheet (GAA) template."""
    p, n = _merge(NANOSHEET_DEFAULTS, NANOSHEET_DERIVED, params, "nanosheet")
    p["n_sheets"] = int(p["n_sheets"])
    bdi = p["isolation"] == "bdi"
    if p["isolation"] not in ("pts", "bdi"):
        raise ValueError(f"nanosheet isolation must be pts or bdi, not {p['isolation']!r}")
    if not bdi:
        p["t_bdi"] = 0
    if p.get("pts_peak") is None:
        p["pts_peak"] = 1e18 if bdi else 1e19
    gm, hk, sg = p["gate_material"], p["gate_dielectric"], p["sige_material"]
    sub_sp = "B" if n else "P"
    process = [
        {"step": "substrate", "label": "substrate", "material": "Si", "name": "substrate", "top_height": "-H_sl"},
    ]
    if bdi:
        process.append({"step": "deposit", "label": "BDI sacrificial layer (high-Ge SiGe)", "mode": "fill",
                        "material": sg, "name": "bdi_sacrificial", "to_height": "-H_stack"})
    for k in range(p["n_sheets"]):           # bottom up: SiGe, then the Si sheet on it
        top_sige = f"-H_stack + {k + 1}*t_sige + {k}*t_sheet"
        process += [
            {"step": "deposit", "label": f"superlattice: SiGe {k + 1}", "mode": "fill", "material": sg,
             "name": "sige", "to_height": top_sige},
            {"step": "deposit", "label": f"superlattice: Si sheet {k + 1}", "mode": "fill", "material": "Si",
             "name": "sheets", "to_height": f"{top_sige} + t_sheet"},
        ]
    process += [
        {"step": "dope", "label": "substrate / sheet doping", "species": sub_sp, "conc": "sub_conc",
         "materials": ["Si", sg]},
        {"step": "pattern", "label": "stack mask", "name": "stack", "mask": {"rect": [-1, "Lx + 1", "zs0", "zs1"]}},
        {"step": "etch", "label": "stack etch", "mode": "anisotropic", "materials": ["Si", sg], "mask": "!stack",
         "depth": "H_sl + sub_fin"},
        {"step": "deposit", "label": "STI oxide fill", "mode": "fill", "material": "SiO2", "name": "sti",
         "to_height": 5},
        {"step": "planarize", "label": "STI CMP (stop on the stack)", "to_height": 0},
        {"step": "implant", "label": "punch-through stopper", "species": sub_sp, "peak": "pts_peak",
         "rp": "H_sl + 5", "sigma": "pts_sigma"},
        # with BDI the STI stays 2 nm above the mesa top: flush, the S/D epi
        # beside the stack would meet the mesa's top edge
        {"step": "etch", "label": "STI recess (stack reveal)", "mode": "anisotropic", "regions": ["sti"],
         "depth": "H_sl - 2" if bdi else "H_sl"},
        {"step": "deposit", "label": "dummy gate (poly)", "mode": "fill", "material": "Poly", "name": "dummy_gate",
         "to_height": "gate_top"},
        {"step": "pattern", "label": "gate mask", "name": "gate", "mask": {"rect": ["xg0", "xg1", -1, "Lz + 1"]}},
        {"step": "etch", "label": "dummy gate etch", "mode": "anisotropic", "regions": ["dummy_gate"],
         "mask": "!gate"},
    ]
    if bdi:
        # the sacrificial layer goes everywhere (the dummy gate holds the
        # stack), then a conformal SiN closes the cavity and a directional
        # etch clears the film seen from above: what is left is the SiN
        # under the stack. Region-selective, so the isotropic reach
        # only has to cover the layer's farthest point from a side wall.
        process += [
            {"step": "etch", "label": "BDI sacrificial layer removal", "mode": "isotropic",
             "regions": ["bdi_sacrificial"], "thickness": "W_sheet/2 + Lg/2 + spacer + 2"},
            {"step": "deposit", "label": "BDI fill (conformal SiN)", "mode": "conformal",
             "material": p["bdi_material"], "name": "bdi", "thickness": "t_bdi/2 + 1"},
            # stop on the top of any other solid: a column through the stack
            # keeps the cavity fill under it
            {"step": "etch", "label": "BDI etch back", "mode": "anisotropic", "regions": ["bdi"],
             "stop_on": ["Si", sg, "Poly", "SiO2"]},
        ]
    process += [
        {"step": "deposit", "label": "outer spacer deposition", "mode": "conformal", "material": "SiN",
         "name": "spacer", "thickness": "spacer"},
        {"step": "pattern", "label": "gate + spacer footprint", "name": "gate_sp",
         "mask": {"rect": ["xg0 - spacer", "xg1 + spacer", -1, "Lz + 1"]}},
        {"step": "etch", "label": "outer spacer etch (stack side walls cleared)", "mode": "anisotropic",
         "regions": ["spacer"], "mask": "!gate_sp"},
        {"step": "etch", "label": "outer spacer etch (gate top)", "mode": "anisotropic", "regions": ["spacer"],
         "depth": "1.15*spacer"},
        {"step": "etch", "label": "S/D recess (through the stack)", "mode": "anisotropic", "materials": ["Si", sg],
         "mask": {"not": {"union": ["gate_sp", "!stack"]}}, "depth": "H_stack + 3"},
        {"step": "etch", "label": "SiGe lateral recess", "mode": "isotropic", "regions": ["sige"],
         "thickness": "inner_spacer"},
        {"step": "deposit", "label": "inner spacer deposition", "mode": "conformal", "material": "SiN",
         "name": "inner_spacer", "thickness": "t_sige/2 + 1"},
        # etch back to the sheet ends: directional, everything outside the
        # gate + spacer footprint (an isotropic etch back deep enough to
        # clear the sheet ends also hollows the pockets: the deposit's
        # surface dips at every gap mouth)
        {"step": "etch", "label": "inner spacer etch back", "mode": "anisotropic", "regions": ["inner_spacer"],
         "mask": "!gate_sp"},
    ]
    _sd_epi_and_contacts(p, process, n, "xg0 - spacer", "xg1 + spacer")
    # replacement metal gate, between the epi/ILD steps and the contacts:
    # ILD CMP opens the dummy gate (planarize to the gate top)
    k_cmp = next(i for i, s in enumerate(process) if s.get("label") == "ILD CMP")
    process[k_cmp] = {"step": "planarize", "label": "ILD CMP (open the dummy gate)", "to_height": "gate_top"}
    rmg = [
        {"step": "etch", "label": "dummy gate pull", "mode": "anisotropic", "regions": ["dummy_gate"]},
        {"step": "etch", "label": "channel release (SiGe removal)", "mode": "isotropic", "regions": ["sige"],
         "thickness": "W_sheet/2 + 2"},
        {"step": "deposit", "label": "gate dielectric (IL + HfO2), wraps every sheet", "mode": "conformal",
         "material": hk, "name": "gate_oxide", "thickness": "tox"},
        {"step": "deposit", "label": "gate metal fill (between the sheets)", "mode": "fill", "material": gm,
         "name": "gate", "to_height": "gate_top"},
        {"step": "planarize", "label": "gate CMP", "to_height": "gate_top"},
        {"step": "deposit", "label": "ILD cap", "mode": "fill", "material": p["ild_material"], "name": "ild",
         "to_height": "ild_top + ild_overfill"},
        {"step": "planarize", "label": "ILD CMP", "to_height": "ild_top"},
    ]
    process[k_cmp + 1:k_cmp + 1] = rmg
    domain = {"x": [0, "Lx"], "z": [0, "Lz"], "height": ["-(H_sl + sub_fin + sub_depth)",
                                                          "ild_top + ild_overfill + top_margin"], "voxel": "voxel"}
    return {"params": p, "domain": domain, "process": process, "device": _device_layer(p)}


TEMPLATES = {"planar": planar, "finfet": finfet, "nanosheet": nanosheet}
