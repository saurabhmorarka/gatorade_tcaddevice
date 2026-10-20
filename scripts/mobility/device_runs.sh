#!/bin/bash
# Mobility-stack device matrix (DEVELOPMENT_LOG 83, docs/mobility.md): every
# fab device I-V plus the 2D planar NMOS/PMOS, one after another (never two
# timing runs at once).
#
#   scripts/mobility/device_runs.sh <code dir> <tag> [extra fab simulate args...]
#
# <code dir>: the checkout to run (a copy of the baseline for "before");
# results go to $OUTROOT (default: the main checkout's out/mob_runs)/<tag>/<device>. The extra
# arguments go to every fab simulate call (e.g. --mobility exact); the 2D
# runs take MOB2D=<config suffix> from the environment (see below).
set -u
CODE=$1; TAG=$2; shift 2
PY=/Users/saurabhmorarka/miniconda3/envs/tcad/bin/python
OUT=${OUTROOT:-/Users/saurabhmorarka/Desktop/Gatorade/tcad1d/out/mob_runs}/$TAG
mkdir -p "$OUT"
cd "$CODE" || exit 1
ONLY=${ONLY:-}
fab() {   # name config vds... -- extra
    local name=$1; shift
    if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then return; fi
    echo "=== $name $(date +%H:%M:%S)"
    /usr/bin/time -p $PY -m fab simulate "$@" --out "$OUT/$name" > "$OUT/$name.log" 2>&1
    tail -6 "$OUT/$name.log"
}
N="--vds 0.05 0.7 --vgs-max 0.8"
P="--vds -0.05 -0.7 --vgs-min -0.8 --vgs-max 0"
fab planar      configs/fab_planar.yaml --mesher octree "$@"
fab finfet      configs/fab_finfet.yaml --mesher octree $N --vgs-step 0.05 "$@"
fab finfet_pmos configs/fab_finfet_pmos.yaml --mesher octree $P --vgs-step 0.05 "$@"
fab finfet_real configs/fab_finfet_realistic.yaml --mesher octree $N --vgs-step 0.05 "$@"
fab ns          configs/fab_nanosheet.yaml --mesher octree $N --vgs-step 0.1 "$@"
fab ns_pmos     configs/fab_nanosheet_pmos.yaml --mesher octree $P --vgs-step 0.1 "$@"
fab ns_bdi      configs/fab_nanosheet_bdi.yaml --mesher octree $N --vgs-step 0.1 "$@"
fab ns_bdi_pmos configs/fab_nanosheet_bdi_pmos.yaml --mesher octree $P --vgs-step 0.1 "$@"
two() {   # name config extra...
    local name=$1; shift
    if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then return; fi
    echo "=== $name $(date +%H:%M:%S)"
    /usr/bin/time -p $PY main2d_mosfet_sweep.py "$@" --out-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
    grep -E "SS|Vt,cc|Ion|real" "$OUT/$name.log" | tail -5
}
two 2d_nmos configs/input_mosfet_2d.yaml ${MOB2D_ARGS:-}
two 2d_pmos configs/input_mosfet_2d_stress_pmos.yaml --no-stress ${MOB2D_ARGS:-}
echo "ALLDONE $(date +%H:%M:%S)"
