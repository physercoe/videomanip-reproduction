# VideoManip Reproduction (IsaacLab)

[![arXiv](https://img.shields.io/badge/arXiv-2602.09013-b31b1b.svg)](https://arxiv.org/abs/2602.09013)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data & checkpoints](https://img.shields.io/badge/%F0%9F%A4%97-datasets%20%26%20checkpoints-blue)](https://huggingface.co/datasets/physer/videomanip-reproduction)

An **unofficial, sim-only reproduction** of **VideoManip**
([Chen et al., arXiv:2602.09013](https://arxiv.org/abs/2602.09013),
[project page](https://videomanip.github.io)) — learning dexterous multi-finger
grasping directly from monocular RGB human videos — ported from IsaacGym to
**IsaacLab 2.3.2 / Isaac Sim 5.1**, plus a set of method-level explorations
(HandFlow hand-stage upgrade, dataset-union training, checkpoint-oscillation
analysis, wrench-aware grasp refinement) that go beyond the paper and are
written up in [`docs/PAPER.md`](docs/PAPER.md).

> The original authors are not involved in this repository. No real-robot part
> (no LEAP Hand / xArm); everything is evaluated in simulation.

## Headline results

Grasp disturbance evaluation, paper's protocol (300-step hold, forces of
0.5× object mass from ±x/±y/±z sequentially, success = displacement < 3 cm,
100 trials/object, all DRO predictions evaluated — no filtering):

**Our 6 objects** (reconstructed from the paper's own released videos):

| method | spraybottle | bottle | can | bulb | hat | jengabox | mean |
|---|---|---|---|---|---|---|---|
| retarget (direct) | 15 | 15 | 25 | 0 | 85 | 20 | 26.7 |
| DRO mixed3x-e5 (HaMeR+ContactOpt data) | 19 | 27 | 34 | 51 | 97 | 54 | 47.0 |
| DRO handflow-e16 (HandFlow+ContactOpt data) | 36 | 54 | 31 | 12 | 91 | 76 | 50.0 |
| **DRO union-e2 (HaMeR ∪ HandFlow data)** | 30 | 49 | 67 | 97 | 78 | 93 | **69.0** |
| DRO mixed3x-e5 + wrench refinement | 42 | 89 | 90 | 100 | 95 | 98 | **85.7** |
| paper headline (single-video, their 20 objects) | — | — | — | — | — | — | 63.75 |

**The paper's own 20 objects** (their released predicted-grasp GLB meshes,
our harness, our models): **handflow-e16 = 65.0% — above the paper's reported
63.75%**; union-e2 = 62.05%; mixed3x-e5 = 56.65%; per-object union of the two
models = 73.3%. Full per-object tables: [`docs/REPORT.md`](docs/REPORT.md).

## Contributions beyond the reproduction

1. **Hand-stage upgrade (HaMeR → [HandFlow](https://github.com/mxxu00/HandFlow))**:
   ~100% detection, 1.5–3× smoother wrist/fingertip accelerations; a
   metric-translation bias (up to ~16 cm on 3/6 objects) diagnosed and fixed by
   MoGe depth-anchoring + median filtering (`scripts/fix_handflow_depth.py`).
   Trained on HandFlow-derived data, the grasp model beats the paper's headline
   on the paper's own objects. *Raw* HandFlow+ContactOpt grasps, however, hold
   only 8% in sim — the value is in the training distribution, not the raw grasps.
2. **Dataset-union training** (HaMeR-derived ∪ HandFlow-derived grasps):
   47.0/50.0% → **69.0%** on our 6 objects.
3. **Checkpoint oscillation**: sim success is strongly non-monotonic in training
   epochs (best epoch 70.7% → next epoch 8.7%; SWA is net-negative).
   *Checkpoint selection by sim evaluation is part of the method.*
4. **Wrench-aware grasp refinement** and a set of negative results (geometric
   de-penetration refinement is net-negative in a kinematic-hold harness —
   penetration *is* the grip-force mechanism; contact-quality filters do not
   predict sim success). Prior-art positioning (DFC, FRoGGeR, GWB) in
   [`docs/REPORT.md`](docs/REPORT.md) §Research findings.

## Pipeline

```
RGB video (paper's released clips)
  └─ MoGe-2 metric depth + intrinsics ─ SAM2 masks ─ GeoCalib (in-the-wild only)
  └─ image→mesh (MeshyAI API, or open TripoSR) + two-stage scale estimation (VLM prior + render-error refine)
  └─ FoundationPose 6D object track
  └─ hand recovery: HaMeR  |  HandFlow (upgrade)   → metric-depth correction
  └─ AnyTeleop/DexPilot-style retargeting → Inspire hand q19
  └─ ContactOpt (DeepContact) grasp optimization
        → D(R,O) grasp model (DRO-Grasp, patched: inspire hand, trackio) — trained on 4×A800
        → IsaacLab disturbance harness (scripts/run_grasp_eval2x.py)
```

## Repository layout

```
data/                 size_priors.json (committed); videos/ + reference/ fetched by scripts/fetch_paper_data.sh
docs/                 PLAN / PROGRESS (lab log) / DATA / REPORT / PAPER / SOTA / SERVER_SETUP / REQUIREMENTS / TRELLIS
src/videomanip/       pipeline package (reconstruct/, grasp/, sim/)
scripts/              one runnable entry point per stage (see table below)
third_party/          manifest + patches only — clone upstreams via scripts/setup_third_party.sh
envs/freeze-*.txt     exact uv pip freeze manifests for every environment
outputs/*/eval|dro/   committed evidence: all sim-eval result jsons, curves, predicted grasps
```

## Installation

Requirements: Linux, NVIDIA GPU (developed on RTX 5080 / trained on 4×A800),
driver ≥ 580 (CUDA 12.8 stack), [uv](https://docs.astral.sh/uv/) for all Python
packaging. Every environment has an exact freeze manifest under `envs/`.

```bash
git clone https://github.com/physercoe/videomanip-reproduction.git && cd videomanip-reproduction
bash scripts/setup_third_party.sh        # clone pinned upstreams + apply our patches
bash scripts/fetch_paper_data.sh         # paper's videos + predicted-grasp GLBs

# environments (one venv per stage family; details + network-mirror notes in AGENTS.md)
uv venv .venv --python 3.11 && uv pip install --python .venv -r envs/freeze-isaaclab232.txt   # sim stack: Isaac Sim 5.1 + IsaacLab 2.3.2
uv venv .venv-recon --python 3.11 && uv pip install --python .venv-recon -r envs/freeze-recon.txt
uv venv .venv-handflow --python 3.11 && uv pip install --python .venv-handflow -r envs/freeze-handflow.txt
uv venv .venv-contactopt --python 3.11 && uv pip install --python .venv-contactopt -r envs/freeze-contactopt.txt
```

**External weights & data** (not redistributable / third-party — download from sources):

| Asset | Source |
|---|---|
| MANO (`mano_v1_2`) | https://mano.is.tue.mpg.de (license acceptance required) |
| HaMeR `_DATA` | https://github.com/geopavlakos/hamer release |
| HandFlow `handflow_denoiser.pt`, WiLoR `detector.pt`, `normalization_stats.npz` | https://github.com/mxxu00/HandFlow release |
| SAM2 checkpoints | https://github.com/facebookresearch/sam2 release |
| MoGe-2, GroundingDINO | Hugging Face hubs of the respective projects |
| DRO-Grasp release data + pretrained ckpts | https://github.com/zhenyuwei2003/DRO-Grasp |
| MeshyAI key (optional; paper's mesh stage) | https://www.meshy.ai — `MESHY_API_KEY` env var |

**Our artifacts** (unique data produced by this project) are on Hugging Face:
[`physer/videomanip-reproduction`](https://huggingface.co/datasets/physer/videomanip-reproduction) —
trained DRO checkpoints (full epoch series of the mixed3x / handflow / union
runs), the derived grasp datasets (`CMapDataset_{videomanip,handflow,union,...}`),
HandFlow/HaMeR hand records, predicted grasps, meshes derived from the paper's
GLBs, and all sim-eval result jsons + curves.

## Running the pipeline (per stage)

Each stage reads `data/` or the previous stage's `outputs/<object>/` and writes
its own `outputs/<object>/<stage>/` (see `docs/DATA.md`). One RGB clip per
object at `data/videos/<object>/rgb.mp4`.

| Stage | Entry point |
|---|---|
| clip prep / frames | `scripts/prepare_frames.py` |
| depth + intrinsics (MoGe-2) | `scripts/run_depth.py` |
| masks (SAM2) | `scripts/run_masks.py` |
| object mesh (Meshy / TripoSR) | `scripts/run_mesh.py` |
| object scale prior (VLM) | `scripts/vlm_size_priors.py` |
| 6D object track (FoundationPose) | `scripts/run_pose.py` |
| hand recovery (HaMeR) | `scripts/run_hamer.py` |
| hand recovery (HandFlow upgrade) | `scripts/run_handflow.py` (+ `scripts/fix_handflow_depth.py`) |
| retargeting → Inspire q19 | `scripts/run_retarget.py` |
| contact optimization | `scripts/run_contactopt.py` |
| DRO dataset build | `scripts/build_dro_dataset.py` |
| DRO training (GPU server) | `scripts/server_{bootstrap,sync,train,mirror}.sh` (guide: `docs/SERVER_SETUP.md`) |
| DRO inference | `scripts/run_dro_inference.py` |
| wrench refinement | `scripts/run_refine_wrench.py` |
| sim evaluation | `scripts/run_grasp_eval2x.py` (run with `OMNI_KIT_ACCEPT_EULA=YES`) |

Sim eval example:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/run_grasp_eval2x.py \
  --object bottle --grasps outputs/bottle/dro/predicted_grasps.npz \
  --num_envs 100 --out outputs/bottle/eval/dro_results.json
```

## Documentation map

- [`docs/REPORT.md`](docs/REPORT.md) — full result tables (incl. paper-20 detail), bugs found, negative results
- [`docs/PAPER.md`](docs/PAPER.md) — research-paper draft of the beyond-reproduction findings
- [`docs/PROGRESS.md`](docs/PROGRESS.md) — chronological lab notebook (every command + outcome)
- [`docs/DATA.md`](docs/DATA.md) — data inventory and provenance
- [`docs/PLAN.md`](docs/PLAN.md), [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) — scope and protocol
- [`docs/SOTA.md`](docs/SOTA.md), [`docs/TRELLIS.md`](docs/TRELLIS.md) — component choices (SOTA survey, open mesh models)
- [`docs/SERVER_SETUP.md`](docs/SERVER_SETUP.md) — migrate the training stack to a new GPU server in 3 commands
- [`AGENTS.md`](AGENTS.md) — conventions, IsaacLab caveats, network/mirror notes

## Citation

Please cite the original paper (this reproduction is unofficial):

```bibtex
@article{chen2026videomanip,
  title   = {Dexterous Manipulation Policies from RGB Human Videos via 3D Hand-Object Trajectory Reconstruction},
  author  = {Chen, Hongyi and Dong, Tony and Wu, Tiancheng and Wang, Liquan and Jangir, Yash and Niu, Yaru and Ye, Yufei and Bharadhwaj, Homanga and Erickson, Zackory and Ichnowski, Jeffrey},
  journal = {arXiv preprint arXiv:2602.09013},
  year    = {2026}
}
```

## License & acknowledgements

This repo's own code and docs: [MIT](LICENSE). Upstream components under
`third_party/` keep their own licenses (DRO-Grasp, ContactOpt, HaMeR, HandFlow,
MoGe, SAM2, GeoCalib, TripoSR, IsaacLab — see `third_party/README.md`).
MANO requires its own license. The input videos and reference GLBs are the
VideoManip authors' data, fetched from their project-site repo. Built with
Isaac Sim / IsaacLab (NVIDIA), DRO-Grasp (NUS LinS Lab), and many open-source
vision models — thanks to all the authors.
