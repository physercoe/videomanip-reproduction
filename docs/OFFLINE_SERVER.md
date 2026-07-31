# VideoManip Reproduction — Offline Kit (for air-gapped internal server)

This directory (`stardata/videomanip/`) contains **everything needed to reproduce the
VideoManip reproduction** (code, environments, weights, data, results) on a server with
**no internet access**. Upstream reference (not reachable offline, for provenance only):
code+docs — https://github.com/physercoe/videomanip-reproduction ;
large artifacts — https://huggingface.co/datasets/physer/videomanip-reproduction ;
paper — https://arxiv.org/abs/2602.09013 .

## Requirements for the internal server

- Linux x86_64, NVIDIA GPU (developed on RTX 5080 16 GB; trained on 4× A800 80 GB).
- **NVIDIA driver ≥ 580** (R580 verified for the Isaac Sim 5.1 sim stack; torch cu128
  stacks work on ≥ 560). CUDA toolkit not required (wheels ship CUDA).
- ~250 GB free disk for full restore (project ~19 GB, venvs ~76 GB, models ~10 GB,
  HF cache ~17 GB, uv cache ~80 GB). Minimal restore (skip uv-kit + venv_lab3) ≈ 130 GB.
- No root strictly required **if** you may write `/app`; otherwise see "different paths"
  in step 5.

## Kit layout

| File | Extract at | Contents |
|---|---|---|
| `uv-kit.tar` | `$HOME` | `~/.cache/uv` (all wheels, offline installs), `~/.local/share/uv` (cpython 3.11/3.12), `~/.local/bin/uv` |
| `models.tar` | `/app` | `/app/models/{drograsp,hamer,handflow,mano,sam2}` — DRO datasets+pretrain, HaMeR+MANO, HandFlow, SAM2 ckpts |
| `project.tar` | `/app/project` | the full repo working tree incl. `.git`, `third_party/` clones, `data/`, all `outputs/` artifacts |
| `venv_isaaclab232.tar` | `/app/project/videomanip` | `.venv` — ACTIVE sim stack: Isaac Sim 5.1 + IsaacLab 2.3.2 |
| `venv_recon.tar` | same | `.venv-recon` — reconstruction/retarget/DRO inference (torch 2.7+cu128) |
| `venv_handflow.tar` | same | `.venv-handflow` — HandFlow hand-stage upgrade |
| `venv_contactopt.tar` | same | `.venv-contactopt` — ContactOpt (CPU pytorch3d) |
| `venv_triposr.tar` | same | `.venv-triposr` — open image→mesh baseline |
| `venv_lab3_parked.tar` | same | `.venv-lab3` — PARKED IsaacLab 3.0 beta stack (optional, skip to save 27 GB) |
| `huggingface-cache.tar` | `$HOME` | `~/.cache/huggingface` — MoGe-2, GroundingDINO, VLM for size priors |
| `videomanip-reproduction.bundle` | — | git bundle (full history); alternative to `project.tar`'s `.git` |
| `SHA256SUMS` | — | integrity: `sha256sum -c SHA256SUMS` after copying |

## Restore (target layout identical to the dev machine — recommended)

```bash
# 0. verify integrity
sha256sum -c SHA256SUMS        # ignore the lab3 line if you skipped that file

# 1. python/uv + offline wheel cache
tar -xf uv-kit.tar -C "$HOME"
export PATH="$HOME/.local/bin:$PATH"

# 2. shared weights/datasets (needs write access to /app; else see step 5)
sudo mkdir -p /app && sudo chown "$USER" /app     # once
tar -xf models.tar -C /app

# 3. project tree (includes .git, third_party clones, data, outputs)
mkdir -p /app/project
tar -xf project.tar -C /app/project
cd /app/project/videomanip

# 4. virtual environments (pick what you need; tar preserves hardlinks/symlinks)
tar -xf venv_isaaclab232.tar -C /app/project/videomanip
tar -xf venv_recon.tar       -C /app/project/videomanip
tar -xf venv_handflow.tar    -C /app/project/videomanip
tar -xf venv_contactopt.tar  -C /app/project/videomanip
tar -xf venv_triposr.tar     -C /app/project/videomanip   # optional
# tar -xf venv_lab3_parked.tar -C /app/project/videomanip # optional, parked stack

# 5. HF model cache
tar -xf huggingface-cache.tar -C "$HOME"
```

**5-alt. different paths** (no `/app` write access): extract wherever you like, then

```bash
ln -sfn <models>/drograsp/data   /path/to/videomanip/third_party/drograsp/data
ln -sfn <models>/drograsp/ckpt   /path/to/videomanip/third_party/drograsp/ckpt
ln -sfn <models>/hamer/_DATA     /path/to/videomanip/third_party/hamer/_DATA
ln -sfn <models>/sam2            /path/to/videomanip/third_party/sam2/checkpoints
# HandFlow scripts read these env vars (defaults point to /app/models):
export DETECTOR_CKPT=<models>/handflow/detector.pt
export MANO_ROOT=<models>/mano/mano_v1_2/models
```

Venvs moved away from `/app/project/videomanip` break (absolute shebangs) — rebuild
offline instead: `uv venv .venv --python 3.11 && uv pip install --offline --python
.venv -r envs/freeze-isaaclab232.txt` (works because uv-kit provides the full cache;
freeze manifests for every venv are in `envs/`).

## Offline environment settings

```bash
export HF_HUB_OFFLINE=1            # never try to reach huggingface.co
export OMNI_KIT_ACCEPT_EULA=YES    # Isaac Sim headless runs
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

- MeshyAI (the paper's mesh stage) is a paid API and **unusable offline** — alternatives:
  `scripts/run_mesh.py --backend hunyuan` (open models, see `docs/TRELLIS.md`; TripoSR
  checkout + `.venv-triposr` also shipped), or simply reuse the shipped meshes in
  `outputs/<obj>/mesh/` (already generated for all 6 objects).
- Logging is trackio (local sqlite) — no wandb anywhere.
- MANO is included (`/app/models/mano`) — it is license-restricted; keep internal.

## Smoke test (10 min)

```bash
cd /app/project/videomanip
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1

# 1. sim stack + eval harness against a shipped result (expect ~49/100 for bottle union-e2):
.venv/bin/python scripts/run_grasp_eval2x.py --object bottle \
  --grasps outputs/bottle/dro/predicted_grasps_union_e2.npz \
  --num_envs 100 --out /tmp/smoke_eval.json

# 2. recon stack: MoGe-2 depth on shipped frames (expect outputs/smoke? writes outputs/bottle/depth)
.venv-recon/bin/python scripts/run_depth.py bottle

# 3. DRO inference with a shipped checkpoint (expect predicted_grasps.npz):
.venv-recon/bin/python scripts/run_dro_inference.py \
  --run videomanip_mixed3x_union --epoch 2 --n_samples 100 --objects bottle
```

Reference numbers to compare against (100-trial disturbance eval, protocol in
`docs/REPORT.md`): union-e2 = 69.0% mean over our 6 objects (spraybottle 30, bottle 49,
can 67, bulb 97, hat 78, jengabox 93); handflow-e16 = 65.0% on the paper's 20 objects.

## What you can run, in order

All inputs and intermediate artifacts are shipped under `outputs/<obj>/`, so every
stage can be re-run independently or skipped. Stage → entry point table:
`docs/../README.md` ("Running the pipeline"), conventions in `AGENTS.md`.

- **Reproduce evals only** (fastest): use `outputs/*/dro/*.npz` +
  `scripts/run_grasp_eval2x.py` (`.venv`).
- **Re-train DRO** (GPU-heavy): datasets are at `third_party/drograsp/data/CMapDataset_*`
  (symlink → `/app/models/drograsp/data`). Build a training venv from
  `envs/freeze-server-drograsp.txt` (`uv pip install --offline`), then e.g.:
  ```bash
  cd third_party/drograsp
  DRO_DATASET_DIR=data/CMapDataset_mixed3x_union ../../.venv-recon/bin/python train.py \
    name=videomanip_mixed3x_union 'dataset.robot_names=[barrett,allegro,shadowhand,ezgripper,robotiq_3finger,inspire]' \
    training.epochs=20   # see scripts/server_train.sh for the full hydra arg list
  ```
  (the `server_*.sh` scripts do the same over ssh for a remote box; on a local GPU
  server just run `train.py` directly, possibly with torchrun/DDP per `docs/SERVER_SETUP.md`).
- **Full pipeline from video**: `data/videos/<obj>/rgb.mp4` is included; follow the
  stage table. GPU-heavy stages: depth/masks/pose (`.venv-recon`), hand recovery
  (`.venv-handflow` or HaMeR in `.venv-recon`), ContactOpt (`.venv-contactopt`).

## Docs map (inside project.tar)

`docs/REPORT.md` (all results), `docs/PAPER.md` (research findings), `docs/PROGRESS.md`
(lab notebook with every command), `docs/DATA.md` (data inventory), `docs/SERVER_SETUP.md`
(training-stack migration), `AGENTS.md` (IsaacLab caveats: q19 convention, PhysX API
workarounds, URDF sacrificial joint).
