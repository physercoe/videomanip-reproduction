#!/usr/bin/env bash
# Bootstrap a fresh GPU server for DRO-Grasp training (A800/sm_80 recipe; also
# works on other CUDA GPUs by changing the torch wheel tag).
# Usage: bash scripts/server_bootstrap.sh [user@host] [remote_dir]
#   defaults: $VIDEOMANIP_SERVER /tmp/vm/drograsp
# Safe to re-run (idempotent).
set -euo pipefail
SERVER=${1:-${VIDEOMANIP_SERVER:?pass user@host as arg 1 or set VIDEOMANIP_SERVER}}
RDIR=${2:-/tmp/vm/drograsp}
TORCH_TAG=${TORCH_TAG:-cu126}          # cu126 for A800; cu128 also OK on driver >=560
cd "$(dirname "$0")/.."

echo "== 1. venv (py3.11) at $SERVER:$RDIR/.venv =="
ssh "$SERVER" "export PATH=\"\$HOME/.local/bin:\$PATH\"; \
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; \
  mkdir -p $RDIR && cd $RDIR && \
  if [ ! -x .venv/bin/python ]; then uv venv --python 3.11 --clear .venv; fi"

echo "== 2. torch 2.7.0+$TORCH_TAG =="
# shellcheck disable=SC2029
ssh "$SERVER" "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd $RDIR && \
  uv pip install --python .venv/bin/python torch==2.7.0+$TORCH_TAG torchvision==0.22.0+$TORCH_TAG \
    --index-url https://download.pytorch.org/whl/$TORCH_TAG"

echo "== 3. remaining deps from envs/freeze-server-drograsp.txt =="
grep -vE '^(torch|torchvision|triton|nvidia-|torch-gen|torchaudio)' \
  envs/freeze-server-drograsp.txt > /tmp/server_rest_reqs.txt
scp -q /tmp/server_rest_reqs.txt "$SERVER:/tmp/server_rest_reqs.txt"
# shellcheck disable=SC2029
ssh "$SERVER" "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd $RDIR && \
  uv pip install --python .venv/bin/python -r /tmp/server_rest_reqs.txt"

echo "== 4. sanity check =="
# shellcheck disable=SC2029
ssh "$SERVER" "cd $RDIR && .venv/bin/python -c \
  'import torch, pytorch_lightning, trackio, pytorch_kinematics, cvxpylayers; \
   print(\"OK\", torch.__version__, torch.cuda.is_available())'"

echo "[done] server env ready at $SERVER:$RDIR"
