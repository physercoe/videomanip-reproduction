#!/usr/bin/env bash
# Mirror DRO training checkpoints from the GPU server and sim-eval each new one.
# Usage: bash scripts/mirror_eval_handflow.sh   (loops until both runs hit epoch 20)
set -u
cd "$(dirname "$0")/.."
LOCK=/tmp/mirror_eval_handflow.lock
exec 9>"$LOCK"
flock -n 9 || { echo "already running"; exit 0; }

SERVER=${VIDEOMANIP_SERVER:?set VIDEOMANIP_SERVER=user@host}
RUNS="videomanip_mixed3x_handflow videomanip_mixed3x_union"
OBJECTS="spraybottle bottle can bulb hat jengabox"
CURVE=outputs/eval/handflow_sim_curve.log
DONE=outputs/eval/.handflow_eval_done
touch "$DONE"

mirror() {
  rsync -az --timeout=60 "$SERVER:/tmp/vm/drograsp/output/" third_party/drograsp/output/ 2>/dev/null
}

eval_ckpt() {
  local run="$1" ep="$2"
  echo "=== $run epoch $ep $(date +%H:%M:%S) ===" | tee -a "$CURVE"
  source .venv-recon/bin/activate
  python scripts/run_dro_inference.py --run "$run" --epoch "$ep" --n_samples 100 \
      > /dev/null 2>&1 || { echo "  inference failed" | tee -a "$CURVE"; return 1; }
  deactivate
  source .venv/bin/activate
  export OMNI_KIT_ACCEPT_EULA=YES
  local mean=0
  for obj in $OBJECTS; do
    python scripts/run_grasp_eval2x.py --object "$obj" \
        --grasps "outputs/$obj/dro/predicted_grasps.npz" --num_envs 50 \
        --out "outputs/$obj/eval/dro_${run}_e${ep}_results.json" > /dev/null 2>&1
    r=$(python3 -c "import json;print(round(json.load(open('outputs/$obj/eval/dro_${run}_e${ep}_results.json'))['success_rate'],3))" 2>/dev/null || echo NA)
    echo "  $obj: $r" | tee -a "$CURVE"
  done
  deactivate
  echo "${run} ${ep}" >> "$DONE"
}

while true; do
  mirror
  all_done=1
  for run in $RUNS; do
    d="third_party/drograsp/output/$run/state_dict"
    [ -d "$d" ] || continue
    for f in "$d"/epoch_*.pth; do
      [ -f "$f" ] || continue
      ep=$(basename "$f" .pth | sed 's/epoch_//')
      [ "$ep" = "20" ] || true
      if ! grep -qx "$run $ep" "$DONE"; then
        eval_ckpt "$run" "$ep"
        all_done=0
      fi
    done
    # run considered finished when epoch_20.pth exists and is evaluated
    if [ -f "$d/epoch_20.pth" ] && grep -qx "$run 20" "$DONE"; then :; else all_done=0; fi
  done
  [ "$all_done" = "1" ] && { echo "[done] all checkpoints evaluated"; break; }
  sleep 300
done
