#!/usr/bin/env bash
# Clone all third_party dependencies at the pinned commits recorded in
# third_party/README.md and apply our local patches from third_party/patches/.
# Idempotent: existing checkouts are left untouched (fetch + checkout to re-pin).
set -euo pipefail
cd "$(dirname "$0")/.."

# name|url|commit|patch
REPOS=(
  "drograsp|https://github.com/zhenyuwei2003/DRO-Grasp.git|07590bd8aeb074671e0d133fb372027f46fbd5f3|drograsp.patch"
  "ContactOpt|https://github.com/facebookresearch/ContactOpt.git|9eeb59a1cdddf4a5e94fec39d77808ddd5ed512c|contactopt.patch"
  "hamer|https://github.com/geopavlakos/hamer.git|3a01849f4148352e9260b69bf28b65d1671a4905|"
  "HandFlow|https://github.com/mxxu00/HandFlow.git|67fa7df536db233408fe6270ca5d2de28d5959c3|"
  "MoGe|https://github.com/microsoft/MoGe.git|925b8ed835a7a9cdb7578ba15c658a0afc969030|"
  "sam2|https://github.com/facebookresearch/sam2.git|2b90b9f5ceec907a1c18123530e92e794ad901a4|"
  "GeoCalib|https://github.com/cvg/GeoCalib.git|97b8968e7798a66bf04fcf791fb535624241bda7|"
  "TripoSR|https://github.com/VAST-AI-Research/TripoSR|107cefdc244c39106fa830359024f6a2f1c78871|triposr.patch"
  "IsaacLab|https://github.com/isaac-sim/IsaacLab.git|37ddf626871758333d6ed89cf64ad702aef127d0|"
  "IsaacLab3|https://github.com/isaac-sim/IsaacLab.git|ffff603eafc6b74264a5261cc0183d6a65390d78|"
  "xr_teleoperate|https://github.com/unitreerobotics/xr_teleoperate.git|7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6|"
)

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url commit patch <<< "$entry"
  if [ ! -d "third_party/$name/.git" ]; then
    echo "== clone $name =="
    git clone "$url" "third_party/$name"
  fi
  echo "== pin $name @ ${commit:0:9} =="
  git -C "third_party/$name" checkout -q "$commit"
  if [ -n "$patch" ]; then
    if git -C "third_party/$name" apply --check "third_party/patches/$patch" 2>/dev/null; then
      git -C "third_party/$name" apply "third_party/patches/$patch"
      echo "   applied $patch"
    else
      echo "   $patch already applied or not applicable — skipping"
    fi
  fi
done
echo "[done] third_party ready"
