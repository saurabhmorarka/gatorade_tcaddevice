cd /Users/saurabhmorarka/Desktop/Gatorade/tcad1d
run() { name=$1; shift; echo "=== $name"; TCAD_LINSOLVE=petsc_ilu ~/miniconda3/envs/tcad/bin/python -W ignore -m fab simulate configs/fab_planar.yaml --mesher octree --out out/qstudy_$name "$@" 2>&1 | grep -E "quantum box|Vt,cc|Ion|total|Error|Traceback"; }
run classical
run all --quantum
run margin10 --quantum --quantum-margin 10
run s2 --quantum-spacing 2
run s4 --quantum-spacing 4
run center --quantum-spacing 1000
