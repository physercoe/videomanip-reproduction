# AGENTS.md — VideoManip Reproduction

Guidance for agents working in this repo. Keep this file and `docs/` current when you
change structure, commands, or conventions.

## What this is

Sim-only reproduction of VideoManip (arXiv:2602.09013): learn dexterous grasping from RGB
human videos, evaluated in **IsaacLab** (not IsaacGym). Full context: `README.md`.
Current state: `docs/PROGRESS.md` (read it first). Data inventory: `docs/DATA.md`.

**Published (2026-07-27)**: code+docs → github.com/physercoe/videomanip-reproduction;
unique large data → huggingface.co/datasets/physer/videomanip-reproduction. third_party is
NOT vendored there (pinned manifest `third_party/README.md` + `patches/` +
`scripts/setup_third_party.sh`); server scripts take `$VIDEOMANIP_SERVER` or `user@host`.

## Environment

- `.venv/` — Python 3.11, Isaac Sim 5.1 + IsaacLab 2.3.2 — the ACTIVE sim stack since the
  driver downgrade to R580 (2026-07-23). Sim scripts run with `OMNI_KIT_ACCEPT_EULA=YES`.
- `.venv-lab3/` — PARKED: Python 3.12, torch 2.10+cu128, Isaac Sim 6.0.1 + IsaacLab
  3.0.0-beta2.patch1 (was needed on driver R595; superseded after the R580 downgrade).
- `.venv-recon/` — Python 3.11, torch 2.7+cu128: MoGe, SAM2, GeoCalib, HaMeR (+smplx; chumpy
  copied in manually), drograsp training, pytorch_kinematics, trackio
- `.venv-handflow/` — Python 3.11, torch 2.7.0+cu128 (same stack as recon): HandFlow
  (third_party/HandFlow, MIT) hand-stage upgrade. Weights in `/app/models/handflow/`,
  MANO in `/app/models/mano/mano_v1_2/models`. Entry: `scripts/run_handflow.py`.
- `.venv-contactopt/` — Python 3.11, torch 2.7, pytorch3d 0.7.8 (CPU-only), torch_geometric,
  manopth, open3d; ContactOpt runs with cwd=`third_party/ContactOpt`
- **Migrating to a GPU server**: `docs/SERVER_SETUP.md` (bootstrap/sync/train/mirror
  scripts + env manifests in `envs/freeze-*.txt`). Server is ephemeral — mirror outputs
  back immediately; datasets/weights canonical under `/app/models`.
- Activate: `source <venv>/bin/activate`; run scripts from repo root.

## IsaacLab 3.0 beta caveats (read before touching sim code)

- `sim.step()`/`*.data` are stale in this build — step physics via
  `omni.physx.get_physx_simulation_interface().simulate()` and read/write via own
  `omni.physics.tensors` views (see scripts/run_grasp_eval.py).
- Isaac Sim 6 URDF importer consumes the chain's first joint — inspire URDF carries a
  sacrificial `virtual_joint_dummy`; **q19 = (dummy, x,y,z,roll,pitch,yaw, 12 fingers)**
  everywhere; drop index 0 for the sim's 18-dof articulation.
- ArticulationRootAPI is at `Hand/Geometry/world/virtual_link_dummy`.
- Set actuator stiffness/damping via `ImplicitActuatorCfg` (URDF-converter joint_drive is ignored).

## Network reachability (verified 2026-07-22; proxy is 127.0.0.1:7897)

| Host | Route |
|---|---|
| pypi.org, files.pythonhosted.org | PROXY (direct is ~70KB/s) |
| download.pytorch.org | PROXY |
| github.com, raw.githubusercontent.com | PROXY |
| pypi.nvidia.com | DIRECT (`no_proxy`), proxy times out |
| pypi.tuna.tsinghua.edu.cn | DIRECT (`no_proxy`) |
| api.meshy.ai | DIRECT, all proxies unset |
| hf-mirror.com | DIRECT + all proxies unset (`env -u all_proxy -u ALL_PROXY -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY`) |
| huggingface.co | broken both ways — use `HF_ENDPOINT=https://hf-mirror.com` |

`all_proxy=socks://...` breaks `huggingface_hub`; any python process touching HF hub must
run with proxies unset (see hf-mirror row).
- Heavy research repos with conflicting deps get their own venvs (`.venv-<stage>`);
  stages communicate only through files under `outputs/`.

## Hard rules

- **Log your work**: append a dated entry to `docs/PROGRESS.md` after any non-trivial step
  (what, commands, outcome, next steps). Update `docs/DATA.md` when adding data.
- **uv only** for Python packages (`uv venv`, `uv pip install ...`). No conda, no system pip.
- **Latest stable versions**, not betas/rc's, unless a component requires otherwise (record why —
  IsaacLab 3.0 beta is the recorded exception).
- Do not commit secrets. API keys are provided by the user at runtime:
  `MESHY_API_KEY` lives as an export in `~/.bashrc` — but bashrc returns early for
  non-interactive shells, so load it with
  `eval "$(grep '^export MESHY_API_KEY=' ~/.bashrc)"` (verified). Never print key values.
- Artifacts stay inside this directory (`data/`, `outputs/`, `third_party/`, `.venv*/`),
  except shared assets under `/app/models`, `/app/datasets` (symlink into the project).
- Do not run `git commit/push` unless the user explicitly asks.

## Layout

```
data/videos/<object>/rgb.mp4   raw input clips (see docs/DATA.md)
data/reference/glb/            paper's predicted-grasp GLBs (reference only)
docs/                          PLAN/PROGRESS/DATA docs
src/videomanip/                pipeline package (reconstruct/, grasp/, sim/)
third_party/                   cloned external repos (do not edit in place; fork-patch if needed)
outputs/<object>/              per-object reconstruction artifacts
outputs/eval/                  IsaacLab eval logs/results
scripts/                       entry points, one per pipeline stage
```

## Pipeline stage conventions

- Each stage reads from `data/` or the previous stage's `outputs/<object>/` and writes its own
  `outputs/<object>/<stage>/`. No cross-stage hidden state; everything via files on disk.
- Per-object dirs use the names in `data/videos/`: spraybottle, bottle, can, bulb, hat, jengabox.
- **Frames**: per-frame `retarget*/%05d.npy` q19 grasps live in the reconstruction frame
  (object NOT at origin). Sim eval and refinement consume the object-at-origin npz form
  (`outputs/<obj>/eval/retarget_grasps.npz`, `outputs/<obj>/dro/predicted_grasps*.npz`).
  q19 = (dummy, x,y,z, roll,pitch,yaw, 12 fingers) everywhere.
- Visualization QA: after any reconstruction/mesh step, render a contact sheet or screenshot and
  view it before moving on.
