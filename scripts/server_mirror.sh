#!/usr/bin/env bash
# Pull training artifacts back from a GPU server (server is ephemeral — mirror often!).
# Usage: bash scripts/server_mirror.sh [user@host] [remote_dir]
set -euo pipefail
SERVER=${1:-${VIDEOMANIP_SERVER:?pass user@host as arg 1 or set VIDEOMANIP_SERVER}}
RDIR=${2:-/tmp/vm/drograsp}
cd "$(dirname "$0")/.."

echo "== state_dicts + logs =="
rsync -az --timeout=60 --include='*/' --include='state_dict/***' --include='*.pth' \
  --exclude='*' "$SERVER:$RDIR/output/" third_party/drograsp/output/ || true
rsync -az --timeout=60 "$SERVER:/tmp/vm/train_*.log" outputs/eval/server_logs/ 2>/dev/null || true

echo "== trackio db =="
mkdir -p outputs/eval
rsync -az --timeout=60 "$SERVER:.cache/huggingface/trackio/DROGrasp.db" outputs/eval/DROGrasp.db || true

echo "[done] mirrored from $SERVER"
