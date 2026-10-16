"""Loads input_diode.yaml and builds Material/Device/voltage-sweep/solver-
choice/mesh overrides from it. Every value in the input file overrides the
corresponding default from params.py; the input file itself is optional
(defaults are used for anything missing or if the file doesn't exist)."""
import os

import numpy as np
import yaml

from core.params import Material, Device
from core import doping_profiles as dp

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "configs", "input_diode.yaml")


def load_config(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _resolve_material_block(block: dict) -> Material:
    """Resolve one `material:`-shaped block (name/derive_from/overrides/T_K,
    or the new alloy: {...} option) into a live Material - shared by the
    top-level `material:` block (today's exact behavior) and the new
    `material.p_side`/`material.n_side` sub-blocks (see build_from_config),
    each shaped identically. An empty/missing block returns Material()'s
    defaults, exactly like today's top-level-only path."""
    mat = Material()
    block = block or {}

    if block.get("alloy") is not None:
        from core import material_db
        from core.materials import AlloyMaterial, resolve_material
        name = block.get("name")
        if not name:
            raise ValueError(
                "material block with 'alloy' requires 'name' to register the "
                "resolved composition under (e.g. name: SiGe_x0.4)")
        alloy_cfg = block["alloy"]
        alloy = AlloyMaterial(
            name=name,
            end_member_a=alloy_cfg["end_member_a"],
            end_member_b=alloy_cfg["end_member_b"],
            bowing_eV=float(alloy_cfg.get("bowing_eV", 0.0)),
        )
        # strain: {substrate, x_substrate_Ge} - optional, compressively
        # strained Si(1-x)Ge(x)-on-substrate band offsets (People & Bean,
        # see core.materials.strained_sige_on_si_offsets) instead of the
        # plain relaxed Vegard-mixed alloy. Omitted (as in the original
        # relaxed configs/input_diode_sige_pn.yaml example) keeps today's
        # exact relaxed behavior.
        strain_cfg = alloy_cfg.get("strain")
        if strain_cfg is not None:
            material_db.derive_alloy(
                name, alloy, float(alloy_cfg["x_a"]),
                strained_on=strain_cfg.get("substrate", "Silicon"),
                x_substrate_Ge=float(strain_cfg.get("x_substrate_Ge", 0.0)))
        else:
            material_db.derive_alloy(name, alloy, float(alloy_cfg["x_a"]))
        mat = resolve_material(material_db.get(name), float(block.get("T_K", 300.0)))
    elif block.get("name") is not None:
        from core import material_db
        from core.materials import resolve_material
        name = block["name"]
        if name not in material_db.MATERIALS:
            derive_from = block.get("derive_from")
            if not derive_from:
                raise ValueError(
                    f"material.name {name!r} is not in the material database; "
                    f"give material.derive_from to derive it from a known "
                    f"material (known materials: {sorted(material_db.MATERIALS)})")
            raw_overrides = block.get("overrides") or {}
            overrides = {}
            for key, value in raw_overrides.items():
                if key.endswith("_ns"):
                    overrides[key[:-3]] = float(value) * 1.0e-9
                else:
                    overrides[key] = float(value)
            material_db.derive(name, derive_from, **overrides)
        mat = resolve_material(material_db.get(name), float(block.get("T_K", 300.0)))
    elif block.get("T_K") is not None:
        raise ValueError(
            "material.T_K requires material.name (no material identity to "
            "apply its temperature-dependent formulas to)")

    if block.get("eps_r") is not None:
        mat.eps_r = float(block["eps_r"])
    if block.get("ni_cm3") is not None:
        mat.ni = float(block["ni_cm3"])
    if block.get("doping_mobility") is not None:          # masetti (default) | caughey_thomas
        mat.doping_mobility = str(block["doping_mobility"])
    apply_mobility_block(mat, block.get("mobility"))
    return mat


# Fields a material block's `mobility:` sub-block may set (core/params.py
# Material). Every material - an alloy region included - starts from
# silicon's values; the 2D/3D solver reads each node's own material.
MOBILITY_BLOCK_FIELDS = ("doping_mobility", "masetti_n", "masetti_p", "mu_n_max", "mu_n_min", "N_ref_n",
                         "alpha_n", "mu_p_max", "mu_p_min", "N_ref_p", "alpha_p", "vsat_n", "vsat_p",
                         "beta_n", "beta_p")


def apply_mobility_block(mat: Material, mob) -> Material:
    """mobility: {field: value} on `mat` (in place). masetti_n/p are the
    9 Masetti coefficients (mu_min1, mu_min2, mu1, Pc, Cr, Cs, alpha,
    beta, mu_max)."""
    if not mob:
        return mat
    unknown = set(mob) - set(MOBILITY_BLOCK_FIELDS)
    if unknown:
        raise ValueError(f"material mobility: unknown field(s) {sorted(unknown)}; "
                         f"allowed: {list(MOBILITY_BLOCK_FIELDS)}")
    for k, v in mob.items():
        if k in ("masetti_n", "masetti_p"):
            v = tuple(float(x) for x in v)
            if len(v) != 9:
                raise ValueError(f"material mobility: {k} needs 9 coefficients, got {len(v)}")
        elif k != "doping_mobility":
            v = float(v)
        setattr(mat, k, v)
    return mat


def build_from_config(cfg: dict):
    """Returns (Material, Device, Va_array, math_model, save_bias_points, mesh_opts, structure_file).

    `mesh_opts` includes a `mat_n` key (None for a homojunction, i.e. every
    existing config with no material.p_side/n_side split - see
    core.mesh.build_diode_grid's own mat_n=None default) - this keeps the
    return tuple's ARITY unchanged (every existing call site's fixed
    7-value unpacking stays valid) while still threading a real second
    material through to build_diode_grid(mat, dev, **mesh_opts) for a
    heterojunction config."""
    dev = Device()

    doping = cfg.get("doping") or {}
    dev.p_profile = dp.parse_doping_profile(doping.get("p_side") or {}, dev.Na)
    dev.n_profile = dp.parse_doping_profile(doping.get("n_side") or {}, dev.Nd)
    dev.Na = dev.p_profile.reference_concentration()
    dev.Nd = dev.n_profile.reference_concentration()

    thickness = cfg.get("thickness") or {}
    if thickness.get("Wp_um") is not None:
        dev.Wp = float(thickness["Wp_um"]) * 1e-4
    if thickness.get("Wn_um") is not None:
        dev.Wn = float(thickness["Wn_um"]) * 1e-4

    material = cfg.get("material") or {}
    mat_n = None
    if material.get("p_side") is not None or material.get("n_side") is not None:
        mat = _resolve_material_block(material.get("p_side"))
        mat_n = _resolve_material_block(material.get("n_side"))
    else:
        mat = _resolve_material_block(material)

    mesh_cfg = cfg.get("mesh") or {}
    mesh_opts = dict(
        growth=float(mesh_cfg.get("growth", 1.06)),
        bulk_spacing_debye_factor=float(mesh_cfg.get("bulk_spacing_debye_factor", 5.0)),
        junction_spacing_debye_factor=float(mesh_cfg.get("junction_spacing_debye_factor", 0.05)),
        mat_n=mat_n,
    )

    vs = cfg.get("voltage_sweep", {})
    rev = np.linspace(
        vs.get("reverse_start_V", -2.0),
        vs.get("reverse_stop_V", -0.05),
        int(vs.get("reverse_points", 10)),
    )
    fwd = np.linspace(
        vs.get("forward_start_V", 0.0),
        vs.get("forward_stop_V", 0.65),
        int(vs.get("forward_points", 27)),
    )
    Va_list = np.concatenate([rev, fwd])

    math_model = cfg.get("solver", {}).get("math_model", "gummel")
    if math_model not in ("gummel", "newton", "newton_qf", "newton_avalanche", "newton_tat"):
        raise ValueError(
            "solver.math_model must be 'gummel', 'newton', 'newton_qf', "
            f"'newton_avalanche', or 'newton_tat', got {math_model!r}")

    output_cfg = cfg.get("output", {})
    save_bias_points = output_cfg.get("save_bias_points", "last")
    structure_file = output_cfg.get("structure_file", "diode_structure.json")

    return mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file
