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
                 tunneling=args.tunneling, vsat=True if args.vsat == "fast" else "mqf")
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
