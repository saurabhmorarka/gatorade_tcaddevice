"""Virtual fab command line.

  python3 -m fab build configs/fab_planar.yaml [--out out/fab_planar] [--final-only]
  python3 -m fab expand configs/fab_planar.yaml > flow.yaml   # layer 1 -> editable layer 2/3
  python3 -m fab simulate configs/fab_planar.yaml [--out out/fab_planar_sim] [--mesher octree]   # 3D Id-Vg

Then view the build step by step:  python3 -m tcadview out/fab_planar/build.pvd
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mobility_options(args):
    """The --mobility / --orientation / --mobility-set options as a dict
    (None: off)."""
    if args.mobility == "off":
        return None
    o = {"mode": args.mobility, "orientation": args.orientation}
    for kv in args.mobility_set:
        k, v = kv.split("=", 1)
        try:
            v = float(v)
        except ValueError:
            pass
        o[k] = v
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fab", description="tcad1d virtual fab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="run a process flow and write VTK snapshots")
    b.add_argument("config")
    b.add_argument("--out", help="output folder (default out/<config name>)")
    b.add_argument("--final-only", action="store_true", help="write only the final structure")
    e = sub.add_parser("expand", help="print the explicit flow (layer 2/3) a template generates")
    e.add_argument("config")
    m = sub.add_parser("simulate", help="3D drift-diffusion Id-Vg of the built device")
    m.add_argument("config")
    m.add_argument("--out", help="output folder (default out/<config name>_sim)")
    m.add_argument("--vds", type=float, nargs="+", default=[0.05, 1.0])
    m.add_argument("--vgs-max", type=float, default=1.2)
    m.add_argument("--vgs-step", type=float, default=0.1)
    m.add_argument("--full-width", action="store_true", help="solve the full width (default: half + symmetry)")
    m.add_argument("--quantum", action="store_true",
                   help="Schrodinger-Poisson channel charge (quantum box under the gate; plans/quantum_plan.md)")
    m.add_argument("--quantum-spacing", type=float, default=None, metavar="NM",
                   help="one Schrodinger line per NM x NM lattice cell (default: every interface column)")
    m.add_argument("--quantum-margin", type=float, default=None, metavar="NM",
                   help="extend the quantum box NM past the gate along the same wall (under the spacers)")
    m.add_argument("--vgs-min", type=float, default=0.0,
                   help="also walk the gate from 0 down to this (negative: GIDL), same step")
    m.add_argument("--tunneling", choices=("local",), default=None,
                   help="band-to-band + trap-assisted tunneling (btbt/local3d.py): GIDL, junction leakage")
    m.add_argument("--vsat", choices=("fast", "exact"), default="fast",
                   help="velocity-saturation driving force: fast (hybrid, default) or exact (modified "
                        "quasi-Fermi gradient, Sentaurus-style; 2.5-3x slower in 3D; docs/vsat_modes.md)")
    m.add_argument("--mesher", choices=("cartesian", "octree"), default="cartesian",
                   help="device mesh: graded rectilinear (default) or octree (exact Voronoi dual)")
    m.add_argument("--stress", nargs="?", const="solve", default=None, metavar="NPZ",
                   help="process stress -> mobility (piezoresistance, docs/piezo_mobility.md): alone, run "
                        "mechanics.fab_stress first and save <out>/stress_field.npz; or a saved field to reuse")
    m.add_argument("--piezo", choices=("saturating", "linear"), default="saturating",
                   help="piezoresistance model with --stress: saturating (default) or linear (Smith)")
    m.add_argument("--piezo-doping", choices=("on", "off"), default="on",
                   help="doping factor of the piezo coefficients (Kanda n / Richter p; default on)")
    m.add_argument("--stress-set", nargs="*", default=[], metavar="K=V",
                   help="mechanics.fab_stress stressor overrides with --stress, e.g. gate_GPa=0")
    m.add_argument("--mobility", choices=("fast", "exact", "off"), default="fast",
                   help="inversion-layer mobility (docs/mobility.md): fast (default; Lombardi phonon + surface "
                        "roughness, Masetti bulk on the total doping, interface charge 3e11 cm^-2 + its "
                        "Coulomb scattering on the gate-dielectric interfaces), exact (Darwish + Klaassen, carrier screening), off (bulk only)")
    m.add_argument("--orientation", choices=("auto", "100"), default="auto",
                   help="surface-orientation parameter sets: auto (per interface: (100) tops, (110) fin "
                        "sidewalls) or 100 (one set)")
    m.add_argument("--mobility-set", nargs="*", default=[], metavar="K=V",
                   help="mobility options, e.g. interface_charge=0 (no fixed charge / Coulomb; default 3e11) "
                        "interface_charge_where=all (also STI / spacers / BOX; default gate) highk=1 "
                        "surface_scale_n=1.1 "
                        "(solver2d/surface_mobility.MobilityModel)")
    m.add_argument("--ballistic", nargs="?", const="on", choices=("on", "off"), default="off",
                   help="ballistic mobility 1/mu = 1/mu_low + 1/mu_bal in the gated channel (docs/ballistic.md; "
                        "default off)")
    m.add_argument("--ballistic-k", type=float, default=None, metavar="K",
                   help="with --ballistic: mu_bal/L_ch in cm2/(V s nm) for both carriers (default: from the "
                        "masses, 23.9 n / 16.4 p; Sentaurus 20; Monte Carlo 8.4)")
    m.add_argument("--ballistic-placement", choices=("lowfield", "total"), default=None,
                   help="with --ballistic: lowfield (Matthiessen into the low-field mobility, before velocity "
                        "saturation) or total (on the field-dependent mobility, Sentaurus Mobility(BalMob))")
    args = ap.parse_args(argv)

    import yaml
    from fab import flow
    if args.cmd == "simulate":
        from fab.simulate import simulate
        out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.config))[0] + "_sim")
        quantum = None
        if args.quantum or args.quantum_spacing or args.quantum_margin:
            quantum = dict(spacing_nm=args.quantum_spacing, margin_nm=args.quantum_margin)
        simulate(args.config, out, vds=args.vds, vgs_max=args.vgs_max, vgs_step=args.vgs_step,
                 half_z=not args.full_width, mesher=args.mesher, quantum=quantum, vgs_min=args.vgs_min,
                 tunneling=args.tunneling, vsat=True if args.vsat == "fast" else "mqf", stress=args.stress,
                 piezo=args.piezo, piezo_doping=args.piezo_doping == "on",
                 stress_set={k: float(v) for k, v in (kv.split("=") for kv in args.stress_set)},
                 mobility=_mobility_options(args),
                 ballistic=None if args.ballistic == "off" else dict(
                     {"enabled": True}, **({"k": args.ballistic_k} if args.ballistic_k is not None else {}),
                     **({"placement": args.ballistic_placement} if args.ballistic_placement else {})))
        return 0
    cfg = flow.load(args.config)
    if args.cmd == "expand":
        yaml.safe_dump(flow.expand(cfg), sys.stdout, sort_keys=False, default_flow_style=None, width=110)
        return 0
    from fab.export import SnapshotWriter
    out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.config))[0])
    w = SnapshotWriter(out, every=not args.final_only)
    t0 = time.perf_counter()
    _, times = flow.build(cfg, on_step=w)
    t_build = sum(t for _, t in times)
    path = w.finish()
    print(f"build {t_build:.1f} s (+ snapshots {time.perf_counter() - t0 - t_build:.1f} s) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
