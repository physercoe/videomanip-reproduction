#!/usr/bin/env bash
# Build an offline-reproduction kit for this project on an external drive.
# The drive may be exfat (no symlinks/perms/hardlinks), so everything is packed
# as uncompressed tar archives (data is already-compressed wheels/ckpts/mp4).
# Usage: bash scripts/make_offline_kit.sh [dst_dir]
#   default dst: /media/wb/T7/stardata/videomanip
set -euo pipefail
DST=${1:-/media/wb/T7/stardata/videomanip}
SRC=/app/project/videomanip
mkdir -p "$DST"
cd "$SRC"

echo "[1] git bundle (full history, single file)"
git bundle create "$DST/videomanip-reproduction.bundle" --all

echo "[2] project.tar (working tree + .git, without venvs)"
tar -cf "$DST/project.tar" -C /app/project \
  --exclude='videomanip/.venv' --exclude='videomanip/.venv-*' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='videomanip/.coverage' \
  videomanip

echo "[3] models.tar (shared weights/datasets -> extract at /app)"
tar -cf "$DST/models.tar" -C /app \
  models/drograsp models/hamer models/handflow models/mano models/sam2

echo "[4] huggingface-cache.tar (MoGe-2, GroundingDINO, VLM, ... -> extract at \$HOME)"
tar -cf "$DST/huggingface-cache.tar" -C "$HOME" .cache/huggingface

for pair in .venv:venv_isaaclab232 .venv-recon:venv_recon .venv-handflow:venv_handflow \
            .venv-contactopt:venv_contactopt .venv-triposr:venv_triposr .venv-lab3:venv_lab3_parked; do
  v=${pair%%:*}; n=${pair##*:}
  [ -d "$SRC/$v" ] || { echo "  skip $v (missing)"; continue; }
  echo "[5] $n.tar <- $v"
  tar -cf "$DST/$n.tar" -C "$SRC" "$v"
done

echo "[6] uv-kit.tar (offline wheel cache + python 3.11/3.12 dists + uv binary)"
tar -cf "$DST/uv-kit.tar" -C "$HOME" \
  .cache/uv .local/share/uv .local/bin/uv .local/bin/uvx 2>/dev/null || \
tar -cf "$DST/uv-kit.tar" -C "$HOME" .cache/uv .local/share/uv .local/bin/uv

echo "[7] SHA256SUMS"
(cd "$DST" && sha256sum *.tar *.bundle > SHA256SUMS)

echo "[done] offline kit at $DST"
ls -lh "$DST"
