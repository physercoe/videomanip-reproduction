# third_party — external dependencies (pinned)

External repos are **not vendored** into this repository. Clone them at the pinned
commits below and apply our patches:

```bash
bash scripts/setup_third_party.sh   # clones everything below + applies third_party/patches/*.patch
```

| Dir | Upstream | Pinned commit | Patch | Used for |
|---|---|---|---|---|
| `drograsp/` | https://github.com/zhenyuwei2003/DRO-Grasp.git | `07590bd8aeb074671e0d133fb372027f46fbd5f3` | `patches/drograsp.patch` | D(R,O) grasp model training/inference |
| `ContactOpt/` | https://github.com/facebookresearch/ContactOpt.git | `9eeb59a1cdddf4a5e94fec39d77808ddd5ed512c` | `patches/contactopt.patch` | contact optimization of hand poses |
| `hamer/` | https://github.com/geopavlakos/hamer.git | `3a01849f4148352e9260b69bf28b65d1671a4905` | — | HaMeR hand recovery |
| `HandFlow/` | https://github.com/mxxu00/HandFlow.git | `67fa7df536db233408fe6270ca5d2de28d5959c3` | — | HandFlow hand-stage upgrade |
| `MoGe/` | https://github.com/microsoft/MoGe.git | `925b8ed835a7a9cdb7578ba15c658a0afc969030` | — | metric depth + intrinsics |
| `sam2/` | https://github.com/facebookresearch/sam2.git | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | — | object masks |
| `GeoCalib/` | https://github.com/cvg/GeoCalib.git | `97b8968e7798a66bf04fcf791fb535624241bda7` | — | gravity alignment (in-the-wild clips) |
| `TripoSR/` | https://github.com/VAST-AI-Research/TripoSR | `107cefdc244c39106fa830359024f6a2f1c78871` | `patches/triposr.patch` | open image→mesh baseline (vs MeshyAI) |
| `IsaacLab/` | https://github.com/isaac-sim/IsaacLab.git | `37ddf626871758333d6ed89cf64ad702aef127d0` | — | IsaacLab 2.3.2 source (sim stack) |
| `IsaacLab3/` | https://github.com/isaac-sim/IsaacLab.git | `ffff603eafc6b74264a5261cc0183d6a65390d78` | — | PARKED: IsaacLab 3.0.0-beta2.patch1 experiment |
| `xr_teleoperate/` | https://github.com/unitreerobotics/xr_teleoperate.git | `7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6` | — | reference only (teleop retargeting) |

## What our patches do

- **`drograsp.patch`** — ports DRO-Grasp to this project: `DRO_DATASET_DIR` env var to
  select the dataset directory, `DRO_INIT_SD` to warm-start from a chosen checkpoint,
  Inspire-hand link handling (`removed_links.json`), and `utils/trackio_logger.py`
  (trackio replaces wandb).
- **`contactopt.patch`** — `contactopt/pointnet.py`: load-flexibility fix for our
  DeepContact checkpoint.
- **`triposr.patch`** — `tsr/models/isosurface.py`: compatibility fix with the pinned
  torch stack.

## Not covered by patches (weights & data symlinks)

- `hamer/_DATA` is a symlink to HaMeR release weights + MANO (see `docs/DATA.md`; MANO
  requires accepting the license at https://mano.is.tue.mpg.de).
- `drograsp/{data,ckpt}` are symlinks into the shared assets dir (`/app/models/drograsp`
  on the author's machine); download our datasets/checkpoints from the Hugging Face
  repo linked in the top-level README, and the original DRO-Grasp release data from
  https://github.com/zhenyuwei2003/DRO-Grasp.
- `drograsp/{pretrain_inspire,videomanip_inspire,videomanip_inspire_aug}/0/checkpoints/`
  are untracked early-training leftovers, safe to delete (superseded by `drograsp/output/`).
