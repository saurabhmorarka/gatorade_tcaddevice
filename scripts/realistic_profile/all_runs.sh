cd /Users/saurabhmorarka/Desktop/Gatorade/tcad1d/.claude/worktrees/realistic-profile
echo "=== seed sweep"
for t in 0 5 7 10; do for c in fab_finfet_pmos fab_nanosheet_pmos; do
  /usr/bin/time -p /Users/saurabhmorarka/miniconda3/envs/tcad/bin/python -m mechanics.fab_stress configs/$c.yaml --fast --fab epi_seed_thickness=$t --tag _seed$t --out out/stress3d_seed
done; done
echo "=== stress realistic"
for c in fab_finfet_pmos_realistic fab_nanosheet_pmos_realistic; do
  /usr/bin/time -p /Users/saurabhmorarka/miniconda3/envs/tcad/bin/python -m mechanics.fab_stress configs/${c}.yaml --fast --out out/stress3d_realistic
done
echo "=== IV"
run() { echo "--- $1"; /usr/bin/time -p /Users/saurabhmorarka/miniconda3/envs/tcad/bin/python -m fab simulate $1 --mesher octree --vds 0.05 0.7 --vgs-max 0.8 --vgs-step $2 --out $3; }
run configs/fab_finfet.yaml 0.05 out/iv_cmp/finfet_ideal
run configs/fab_finfet_realistic.yaml 0.05 out/iv_cmp/finfet_real
run configs/fab_nanosheet.yaml 0.1 out/iv_cmp/nanosheet_ideal
run configs/fab_nanosheet_realistic.yaml 0.1 out/iv_cmp/nanosheet_real
echo ALLDONE
