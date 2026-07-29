#!/usr/bin/env python3
"""Upload this project's unique artifacts to HF dataset repo physer/videomanip-reproduction.
Run: env -u all_proxy -u ALL_PROXY -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
       HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=... python scripts/upload_hf.py
"""
import os
import sys
import time

from huggingface_hub import HfApi

REPO = "physer/videomanip-reproduction"
ROOT = "/app/project/videomanip"
DRO_DATA = "/app/models/drograsp/data"
OBJECTS = "spraybottle bottle can bulb hat jengabox".split()

JOBS = []  # (local_dir, path_in_repo)
for run, short in [
    ("videomanip_mixed3x", "mixed3x"),
    ("videomanip_mixed3x_handflow", "mixed3x_handflow"),
    ("videomanip_mixed3x_union", "mixed3x_union"),
]:
    JOBS.append((f"{ROOT}/third_party/drograsp/output/{run}/state_dict", f"checkpoints/{short}"))
for name in [
    "CMapDataset_videomanip", "CMapDataset_handflow", "CMapDataset_union",
    "CMapDataset_mixed", "CMapDataset_mixed3x", "CMapDataset_mixed3x_handflow",
    "CMapDataset_mixed3x_union", "CMapDataset_selfdistill",
]:
    JOBS.append((f"{DRO_DATA}/{name}", f"datasets/{name}"))
JOBS.append((f"{DRO_DATA}/PointCloud/object/videomanip", "datasets/PointCloud_videomanip"))
for obj in OBJECTS:
    for stage in ("hand", "handflow"):
        d = f"{ROOT}/outputs/{obj}/{stage}"
        if os.path.isdir(d):
            JOBS.append((d, f"hand_records/{obj}/{stage}"))
for d in sorted(os.listdir(f"{ROOT}/outputs")):
    for sub, dst in (("dro", "predictions"), ("eval", "eval_results")):
        p = f"{ROOT}/outputs/{d}/{sub}"
        if os.path.isdir(p):
            JOBS.append((p, f"{dst}/{d}"))
JOBS.append((f"{ROOT}/outputs/eval", "eval_results/_global"))
JOBS.append((f"{ROOT}/data/reference/meshes", "derived_meshes"))

api = HfApi()
api.create_repo(REPO, repo_type="dataset", exist_ok=True)
print(f"repo ready: {REPO}; {len(JOBS)} upload jobs", flush=True)

CARD = "/tmp/hf_card.md" if os.path.exists("/tmp/hf_card.md") else f"{ROOT}/docs/HF_DATASET_CARD.md"
api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md",
                repo_id=REPO, repo_type="dataset")
print("card uploaded", flush=True)

failed = []
for i, (src, dst) in enumerate(JOBS, 1):
    if not os.path.isdir(src):
        print(f"[{i}/{len(JOBS)}] MISSING {src} — skipped", flush=True)
        failed.append(src)
        continue
    t0 = time.time()
    try:
        api.upload_folder(folder_path=src, path_in_repo=dst,
                          repo_id=REPO, repo_type="dataset")
        print(f"[{i}/{len(JOBS)}] OK {dst} <- {src} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:  # noqa: BLE001 - log and continue with remaining jobs
        print(f"[{i}/{len(JOBS)}] FAIL {dst}: {type(e).__name__}: {e}", flush=True)
        failed.append(src)

print("FAILED:" if failed else "ALL DONE", *failed, flush=True)
sys.exit(1 if failed else 0)
