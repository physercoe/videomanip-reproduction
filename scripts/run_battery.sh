#!/usr/bin/env bash
# Full evaluation battery: retargeted + DRO-predicted grasps for all objects.
# Usage: bash scripts/run_battery.sh [epoch]
set -u
cd "$(dirname "$0")/.."
EPOCH=${1:-200}
OBJECTS="spraybottle bottle can bulb hat jengabox"

echo "=== 1. DRO inference (epoch $EPOCH) ==="
source .venv-recon/bin/activate
python scripts/run_dro_inference.py --epoch "$EPOCH" --n_samples 100 --filter

echo "=== 2. IsaacLab eval ==="
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
for obj in $OBJECTS; do
  echo "--- $obj: retargeted grasps ---"
  python scripts/run_grasp_eval2x.py --object "$obj" \
      --grasps "outputs/$obj/eval/retarget_grasps.npz" --num_envs 20 \
      --out "outputs/$obj/eval/retarget_results.json" || true
  echo "--- $obj: DRO predicted (filtered) ---"
  if [ -f "outputs/$obj/dro/predicted_grasps_filtered.npz" ]; then
    n=$(python3 -c "import numpy as np; print(len(np.load('outputs/$obj/dro/predicted_grasps_filtered.npz')['q']))" 2>/dev/null || echo 0)
  else
    n=0
  fi
  if [ "${n:-0}" -ge 1 ]; then
    grasps_file="outputs/$obj/dro/predicted_grasps_filtered.npz"
  else
    grasps_file="outputs/$obj/dro/predicted_grasps.npz"
  fi
  echo "--- $obj: DRO predicted ($grasps_file) ---"
  python scripts/run_grasp_eval2x.py --object "$obj" \
      --grasps "$grasps_file" --num_envs 100 \
      --out "outputs/$obj/eval/dro_results.json" || true
done

echo "=== 3. Aggregate report ==="
python scripts/make_report.py
