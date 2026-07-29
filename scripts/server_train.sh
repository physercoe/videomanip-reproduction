#!/usr/bin/env bash
# Launch a DRO training run on a GPU server (detached; survives ssh disconnect).
# Usage: bash scripts/server_train.sh <run_name> <dataset_dir> <gpus> [epochs] [batch_per_gpu] [server]
#   e.g.: bash scripts/server_train.sh videomanip_mixed3x_handflow CMapDataset_mixed3x_handflow 0,1 20 16
# NOTE: cd into the repo INSIDE the remote command — a leading `cd X && setsid ... &`
# puts the whole chain in background and the next command runs from $HOME (lesson 2026-07-24).
set -euo pipefail
NAME=$1; DATASET=$2; GPUS=$3; EPOCHS=${4:-20}; BATCH=${5:-16}; SERVER=${6:-${VIDEOMANIP_SERVER:?pass user@host as arg 6 or set VIDEOMANIP_SERVER}}
RDIR=${RDIR:-/tmp/vm/drograsp}
GPU_LIST="[${GPUS}]"

# shellcheck disable=SC2029
ssh "$SERVER" "cd $RDIR && setsid nohup env DRO_DATASET_DIR=data/$DATASET .venv/bin/python train.py \
  name=$NAME \
  'dataset.robot_names=[barrett,allegro,shadowhand,ezgripper,robotiq_3finger,inspire]' \
  dataset.batch_size=$BATCH dataset.num_workers=$BATCH \
  'gpu=$GPU_LIST' training.max_epochs=$EPOCHS training.save_every_n_epoch=1 \
  model.pretrain=pretrain_inspire.pth \
  > /tmp/vm/train_$NAME.log 2>&1 & echo launched pid \$!"
echo "[ok] $NAME on $SERVER gpu $GPUS, data=$DATASET, epochs=$EPOCHS — log: /tmp/vm/train_$NAME.log"
