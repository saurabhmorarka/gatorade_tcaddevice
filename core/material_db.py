"""The material-properties registry: a name -> MaterialProperties lookup,
meant as a common catalog future sims beyond diode/MOS/avalanche can also
draw from.

Values below are seeded from this project's existing hardcoded defaults
(core/params.py's Material, mos/mos_params.py's oxide/band constants) plus
standard literature values for entries not yet used anywhere in the solver
(Germanium, Si3N4, the named metals). mos_params.py's own GATE_WORKFUNCTION_*
floats (poly-Si and idealized-midgap work functions, not real metal
identities) are intentionally left untouched by this module - they aren't
catalog entries, and nothing about them changes here.
"""
import dataclasses

from core.materials import AlloyMaterial, MaterialProperties

MATERIALS: dict = {
    "Silicon": MaterialProperties(
        name="Silicon", category="semiconductor",
        eps_r=11.7, chi_eV=4.05, Eg_eV_300K=1.12,
        Nc_300K=2.8e19, Nv_300K=1.04e19,
        mu_n=1350.0, mu_p=480.0, tau_n=1.0e-9, tau_p=1.0e-9,
        varshni_alpha_eV_per_K=4.73e-4, varshni_beta_K=636.0,
    ),
    "Germanium": MaterialProperties(
        name="Germanium", category="semiconductor",
        eps_r=16.2, chi_eV=4.0, Eg_eV_300K=0.66,
        Nc_300K=1.04e19, Nv_300K=6.0e18,
        mu_n=3900.0, mu_p=1900.0, tau_n=1.0e-9, tau_p=1.0e-9,
        varshni_alpha_eV_per_K=4.77e-4, varshni_beta_K=235.0,
    ),
    "SiO2": MaterialProperties(
        name="SiO2", category="insulator",
        eps_r=3.9, chi_eV=0.9, Eg_eV_300K=9.0,
    ),
    "Si3N4": MaterialProperties(
        name="Si3N4", category="insulator",
        eps_r=7.5, Eg_eV_300K=5.0,
    ),
    "Ti": MaterialProperties(name="Ti", category="metal", workfunction_eV=4.33),
    "TiN": MaterialProperties(name="TiN", category="metal", workfunction_eV=4.6),
    # NiSi: near-midgap on Si (4.6-4.7 reported).
    "NiSi": MaterialProperties(name="NiSi", category="metal", workfunction_eV=4.65),
    "Al": MaterialProperties(name="Al", category="metal", workfunction_eV=4.28),
    "W": MaterialProperties(name="W", category="metal", workfunction_eV=4.55),
}


def get(name: str) -> MaterialProperties:
    """Look up a material by exact name; raises KeyError listing every
    known name on a typo (mirrors this project's existing
    ValueError-with-choices convention for math_model/gate_type)."""
    if name not in MATERIALS:
        raise KeyError(f"Unknown material {name!r}; known materials: {sorted(MATERIALS)}")
    return MATERIALS[name]


def derive(name: str, base: str, **overrides) -> MaterialProperties:
    """Register a new named material by copying `base`'s fields and
    overriding only the given ones (dataclasses.replace under the hood),
    e.g. derive("Si1", "Silicon", tau_n=5.0e-9). Promotes this project's
    existing ad hoc copy.copy(dev)+mutate pattern (mos/mos_poly_sweep.py,
    mos/mos_main.py, mos/mos_metal_sweep.py) to a named, registry-aware
    helper - a discrete copy-and-override, distinct from AlloyMaterial's
    parametric composition mixing (materials.py).
    """
    base_props = get(base) if isinstance(base, str) else base
    derived = dataclasses.replace(base_props, name=name, **overrides)
    MATERIALS[name] = derived
    return derived


def derive_alloy(name: str, alloy: AlloyMaterial, x: float,
                  strained_on: str = None, x_substrate_Ge: float = 0.0,
                  **overrides) -> MaterialProperties:
    """Register a new named material by resolving an AlloyMaterial at
    composition x (Vegard-law mixing + Eg bowing, see materials.AlloyMaterial)
    and storing the result under `name`, mirroring derive()'s "register a
    named catalog entry" pattern but sourced from a parametric alloy instead
    of a discrete copy-and-override of one existing entry. **overrides
    apply on top of the resolved composition (dataclasses.replace), e.g. to
    hand-tune a field the Vegard/bowing mix doesn't capture well.

    Calibration note for the relaxed p-SiGe(x_Ge=0.4)/n-Si heterojunction
    example this was added for: AlloyMaterial("SixGe1-x", "Silicon",
    "Germanium", bowing_eV=0.36).resolve(x_a=0.6) (i.e. Si_0.6 Ge_0.4) lands
    at Eg ~= 0.85 eV, the standard Braunstein/People relaxed-alloy fit for
    that composition - bowing_eV is left at AlloyMaterial's default 0.0
    (no bowing) unless the caller passes one, so an uncalibrated call will
    NOT reproduce that number; 0.36 eV is what the calibrated
    configs/input_diode_sige_pn.yaml example actually passes.

    strained_on: pass a substrate material name (e.g. "Silicon") to resolve
    via alloy.resolve_strained(x, substrate=strained_on, x_substrate_Ge=...)
    instead of the plain relaxed alloy.resolve(x) - the compressively-
    strained Si(1-x)Ge(x)-on-substrate band offsets (People & Bean, see
    materials.strained_sige_on_si_offsets), used by
    configs/input_diode_sige_pn_strained.yaml. None (default) keeps
    today's exact relaxed behavior."""
    if strained_on is not None:
        resolved = alloy.resolve_strained(x, substrate=strained_on, x_substrate_Ge=x_substrate_Ge)
    else:
        resolved = alloy.resolve(x)
    derived = dataclasses.replace(resolved, name=name, **overrides) if overrides else \
        dataclasses.replace(resolved, name=name)
    MATERIALS[name] = derived
    return derived
