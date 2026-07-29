#!/usr/bin/env python3
"""Prepare per-object frame sequences from the raw input videos.

- Trims the spraybottle compilation clip to its full-res segment (frames 0-296).
- Extracts PNG frames at native fps for each object into outputs/<obj>/frames/.
Usage: python scripts/prepare_frames.py [object ...]  (default: all)
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "data" / "videos"
OUT = ROOT / "outputs"

OBJECTS = ["spraybottle", "bottle", "can", "bulb", "hat", "jengabox"]
# frame ranges [start, end) to trim; None = whole video
TRIM = {"spraybottle": (0, 114)}


def main() -> None:
    objects = sys.argv[1:] or OBJECTS
    for obj in objects:
        src = VIDEOS / obj / "rgb.mp4"
        if not src.exists():
            print(f"[skip] {obj}: {src} missing")
            continue
        dst = OUT / obj / "frames"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        # keep frames start<=n<end and reset timestamps
        if obj in TRIM:
            vf = f"select='gte(n\\,{TRIM[obj][0]})*lt(n\\,{TRIM[obj][1]})',setpts=N/FRAME_RATE/TB"
        else:
            vf = "setpts=N/FRAME_RATE/TB"
        cmd = ["ffmpeg", "-v", "error", "-i", str(src), "-vf", vf, "-start_number", "0",
               "-q:v", "2", str(dst / "%05d.jpg"), "-y"]
        subprocess.run(cmd, check=True)
        n = len(list(dst.glob("*.jpg")))
        print(f"[ok] {obj}: {n} frames -> {dst}")


if __name__ == "__main__":
    main()
