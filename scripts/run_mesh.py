#!/usr/bin/env python3
"""Stage 2: object mesh from the best pre-grasp frame (image-to-3D).

Selects the frame with the largest object mask among the first 30% of the clip
(object still unoccluded by the hand), crops with margin, and reconstructs a mesh.

Backends:
  meshy   — MeshyAI API (paper's choice). Needs MESHY_API_KEY in env
            (run with `source ~/.bashrc` first; user keeps the key there).
  hunyuan — Hunyuan3D-2 local (open-source comparison). TODO: install.

Reads  outputs/<obj>/frames/, outputs/<obj>/masks/
Writes outputs/<obj>/mesh/input.png, mesh/meshy_task.json, mesh/object.glb (meshy)
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

MESHY_API = "https://api.meshy.ai/openapi/v1/image-to-3d"


def pick_best_frame(obj: str, frac: float = 0.3) -> tuple[np.ndarray, np.ndarray, int]:
    frames = sorted((OUT / obj / "frames").glob("*.jpg"))
    n = max(1, int(len(frames) * frac))
    best, best_area = None, -1
    for i in range(n):
        m = cv2.imread(str(OUT / obj / "masks" / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        area = int((m > 0).sum())
        if area > best_area:
            best, best_area = i, area
    assert best is not None, f"no masks found for {obj}"
    img = cv2.imread(str(frames[best]))
    mask = cv2.imread(str(OUT / obj / "masks" / f"{best:05d}.png"), cv2.IMREAD_GRAYSCALE)
    return img, mask, best


def crop_on_white(img: np.ndarray, mask: np.ndarray, margin: float = 0.25) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(img.shape[1], x2 + mx), min(img.shape[0], y2 + my)
    crop_img, crop_mask = img[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    white = np.full_like(crop_img, 255)
    out = np.where(crop_mask[..., None] > 0, crop_img, white)
    # square-pad (Meshy works best with square images)
    h, w = out.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 255, np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = out
    return sq


def meshy_image_to_mesh(image_path: Path, work_dir: Path, polycount: int = 100000) -> dict:
    import requests
    key = os.environ.get("MESHY_API_KEY")
    assert key, "MESHY_API_KEY not in env (run `source ~/.bashrc` first)"
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "image_url": f"data:image/png;base64,{b64}",
        "ai_model": "latest",
        "topology": "triangle",
        "target_polycount": polycount,
        "enable_pbr": False,
        "should_remesh": True,
    }
    for attempt in range(4):
        try:
            r = requests.post(MESHY_API, headers=headers, json=payload, timeout=(30, 300))
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[meshy] POST attempt {attempt + 1} failed: {type(e).__name__}")
            time.sleep(5)
    else:
        raise RuntimeError("Meshy POST failed after retries")
    task_id = r.json()["result"]
    print(f"[meshy] task {task_id} created")
    while True:
        time.sleep(10)
        r = requests.get(f"{MESHY_API}/{task_id}", headers=headers, timeout=60)
        r.raise_for_status()
        info = r.json()
        status = info.get("status")
        print(f"[meshy] {status} {info.get('progress', 0)}%")
        if status in ("SUCCEEDED", "FAILED", "CANCELED"):
            (work_dir / "meshy_task.json").write_text(json.dumps(info, indent=1))
            if status != "SUCCEEDED":
                raise RuntimeError(f"Meshy task {status}: {info.get('task_error')}")
            return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--backend", choices=["meshy", "hunyuan"], default="meshy")
    args = ap.parse_args()

    for obj in args.objects:
        mesh_dir = OUT / obj / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        img, mask, fidx = pick_best_frame(obj)
        crop = crop_on_white(img, mask)
        cv2.imwrite(str(mesh_dir / "input.png"), crop)
        print(f"[{obj}] input from frame {fidx} -> {mesh_dir/'input.png'}")
        if args.backend == "meshy":
            info = meshy_image_to_mesh(mesh_dir / "input.png", mesh_dir)
            import requests
            url = info["model_urls"]["glb"]
            content = requests.get(url, timeout=300).content
            (mesh_dir / "object.glb").write_bytes(content)
            print(f"[ok] {obj}: mesh -> {mesh_dir/'object.glb'} ({len(content)} bytes)")
        else:
            raise NotImplementedError("hunyuan backend TODO")


if __name__ == "__main__":
    main()
