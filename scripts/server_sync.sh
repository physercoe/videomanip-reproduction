#!/usr/bin/env bash
# Sync DRO-Grasp code + data + pretrain ckpts to a GPU server (~1.2 GB).
# Code comes from third_party/drograsp (patched: trackio, DRO_DATASET_DIR,
# DRO_INIT_SD); data+ckpt live in /app/models/drograsp (symlinked).
# Usage: bash scripts/server_sync.sh [user@host] [remote_dir] [dataset_dir ...]
#   extra args = dataset dirs under data/ to sync (default: all CMapDataset_* + videomanip objects)
set -euo pipefail
SERVER=${1:-${VIDEOMANIP_SERVER:?pass user@host as arg 1 or set VIDEOMANIP_SERVER}}
RDIR=${2:-/tmp/vm/drograsp}
shift 2 || true
cd "$(dirname "$0")/.."
DRO=third_party/drograsp
MODELS=/app/models/drograsp

echo "== code =="
rsync -az --delete \
  --exclude '__pycache__' --exclude 'data' --exclude 'ckpt' --exclude 'output' \
  --exclude '.git' --exclude 'log' --exclude '.venv' \
  "$DRO/" "$SERVER:$RDIR/"

echo "== data (from /app/models/drograsp) =="
if [ "$#" -gt 0 ]; then
  # sync only the named dataset dirs (relative to $MODELS/data)
  (cd "$MODELS/data" && rsync -az -R "$@" "$SERVER:$RDIR/data/")
else
  rsync -az "$MODELS/data/" "$SERVER:$RDIR/data/"
fi

echo "== pretrain ckpts =="
rsync -az "$MODELS/ckpt/pretrain/" "$SERVER:$RDIR/ckpt/pretrain/"

echo "[done] synced -> $SERVER:$RDIR"
