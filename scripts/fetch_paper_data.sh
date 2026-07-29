#!/usr/bin/env bash
# Fetch the paper's released data (input videos + predicted-grasp GLBs) from the
# official project-site repo https://github.com/videomanip/videomanip.github.io
# (files live under static/data/). See docs/DATA.md for the inventory and notes.
# Usage: bash scripts/fetch_paper_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="https://raw.githubusercontent.com/videomanip/videomanip.github.io/main/static/data"

dl() { # dl <remote_path> <local_path>
  mkdir -p "$(dirname "$2")"
  [ -s "$2" ] && { echo "skip $2 (exists)"; return; }
  echo "fetch $2"
  curl -fSL --retry 3 -o "$2" "$BASE/$1"
}

# --- input RGB clips (data/videos/<object>/rgb.mp4) ---
dl "grasp/human_grasp_video.mp4"        "data/videos/spraybottle/rgb.mp4"   # NOTE: only frames 0-297 are full-res
dl "manip/in_scene/pourtea/rgb.mp4"     "data/videos/bottle/rgb.mp4"
dl "manip/in_scene/placebottle/rgb.mp4" "data/videos/can/rgb.mp4"
dl "manip/in_the_wild/bulb/rgb.mp4"     "data/videos/bulb/rgb.mp4"
dl "manip/in_the_wild/hang/rgb.mp4"     "data/videos/hat/rgb.mp4"
dl "manip/in_the_wild/jenga_move/rgb.mp4" "data/videos/jengabox/rgb.mp4"
dl "manip/in_scene/closedrawer/rgb.mp4" "data/videos/drawer/rgb.mp4"        # optional

# --- paper's predicted grasps (data/reference/glb/<object>.glb), 20 objects ---
for obj in apple bottle bowl case cloth_hanger cup hand_bag hat ladle mug pan pot \
           powerdrill scissors soap_dispenser spraybottle sunglass toothbrush umbrella wineglass; do
  dl "grasp/inspire/$obj/0.glb" "data/reference/glb/$obj.glb"
done

echo "[done] paper data fetched. data/reference/meshes/ (hand/object ply per object)"
echo "       is derived from these GLBs with trimesh; a copy is also on our HF repo."
