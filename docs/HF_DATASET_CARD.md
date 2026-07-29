---
license: mit
tags:
  - robotics
  - grasping
  - dexterous-manipulation
  - videomanip
  - dro-grasp
  - isaaclab
pretty_name: VideoManip Reproduction (IsaacLab) — artifacts
---

# VideoManip Reproduction (IsaacLab) — data & checkpoints

Unique artifacts produced by the **unofficial sim-only reproduction of VideoManip**
([arXiv:2602.09013](https://arxiv.org/abs/2602.09013)) in IsaacLab 2.3.2 / Isaac Sim 5.1.
Code, docs, protocol and full result tables:
**https://github.com/physercoe/videomanip-reproduction**

## Layout

```
checkpoints/
  mixed3x/           epoch_{5..40}.pth   DRO run on HaMeR+ContactOpt data  (6-obj sim mean 47.0% @ e5)
  mixed3x_handflow/  epoch_{1..20}.pth   DRO run on HandFlow+ContactOpt data (50.0% @ e16; 65.0% on paper-20)
  mixed3x_union/     epoch_{1..20}.pth   DRO run on the union dataset      (69.0% @ e2)
datasets/
  CMapDataset_{videomanip,handflow,union,mixed,mixed3x,mixed3x_handflow,mixed3x_union,selfdistill}/
  PointCloud_videomanip/   512x6 (xyz+normal) object point clouds, 6 objects
hand_records/<obj>/{hand,handflow}/*.npy   per-frame hand records (HaMeR / HandFlow, depth-corrected)
predictions/<obj>/predicted_grasps*.npz    DRO predictions + wrench-refined variants (object-at-origin q19)
derived_meshes/<obj>/{hand,object}.ply     extracted from the paper's predicted-grasp GLBs
eval_results/<obj>/*.json                  every sim-eval result (100-trial disturbance protocol)
eval_results/_global/                      loss/oscillation curves, trackio db, eval logs
```

6 own objects: spraybottle, bottle, can, bulb, hat, jengabox. `pref_*` = the paper's
20 objects evaluated with our models/harness. q19 convention = (dummy, x,y,z,
roll,pitch,yaw, 12 finger joints) — see repo `AGENTS.md`.

## Checkpoints & headline numbers

| checkpoint | 6 own objects | paper's 20 objects |
|---|---|---|
| mixed3x/epoch_5.pth | 47.0% | 56.65% |
| mixed3x_handflow/epoch_16.pth | 50.0% | **65.0%** (paper reports 63.75%) |
| mixed3x_union/epoch_2.pth | **69.0%** | 62.05% |

Note the checkpoint-oscillation finding (see repo `docs/REPORT.md`): later epochs
are NOT better — always select checkpoints by sim evaluation, not by loss.

## Usage

```python
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="physer/videomanip-reproduction", repo_type="dataset")
# or selectively: allow_patterns=["checkpoints/mixed3x_union/epoch_2.pth", "datasets/*"]
```

Datasets load with the patched DRO-Grasp in the GitHub repo
(`DRO_DATASET_DIR=data/<name>`); inference: `scripts/run_dro_inference.py`.

## Provenance & license

- `CMapDataset_mixed*` contain samples derived from the DRO-Grasp authors'
  released grasp data (https://github.com/zhenyuwei2003/DRO-Grasp) mixed with
  grasps reconstructed by this project — credit both.
- `derived_meshes/` come from the VideoManip authors' predicted-grasp GLBs
  (https://github.com/videomanip/videomanip.github.io) — credit the paper's authors.
- Everything else was produced by this reproduction and is released under MIT.
- If you use these artifacts, cite the original VideoManip paper (bibtex in the GitHub README).
