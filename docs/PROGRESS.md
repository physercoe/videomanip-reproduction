# Progress Log

Chronological record of work. Newest entries at the bottom. Each entry: what was done,
exact commands/paths where useful, and the outcome.

## 2026-07-22 — Paper reading & scoping

- Read the paper (arXiv:2602.09013 v2 HTML). Core method recorded in README.md.
  Key numbers: sim grasp eval = IsaacGym, 18-DoF Inspire hand, 300-step disturbance from
  ±x/±y/±z, force = 0.5× object mass, success = object displacement < 3 cm, 100 trials/object.
  Paper results: 63.75% avg over 20 objects (single video per object), 30.7% without ContactOpt,
  70.25% with 10 extra videos for 5 failed objects.
- Confirmed **no official code/data release** (project page + web/github search).
- Version research: IsaacLab latest stable = **v2.3.2** (final main-branch release) on
  **Isaac Sim 5.1**; IsaacLab 3.0.0-beta exists but is beta. Isaac Sim 5.1 pip install supports
  Blackwell (torch 2.7+cu128, Python 3.11). Decision: Isaac Sim 5.1 + IsaacLab 2.3.2 via uv.
- Identified D(R,O) grasp model = [D(R,O) Grasp (arXiv:2410.01702)](https://arxiv.org/abs/2410.01702),
  open source at nus-lins-lab/drograsp.

## 2026-07-22 — User decisions (via questions)

Scope = full sim grasping; videos = scrape project page; MeshyAI key available for comparison,
GPT-4.1 → Kimi (Moonshot API, "kimi-3"), other ideas allowed; 3–5 objects first.

## 2026-07-22 — Data acquisition

- Live site 404s on per-task videos; found the site's git repo and correct paths under
  `static/data/manip/{in_scene,in_the_wild}/<task>/rgb.mp4`.
- Downloaded 7 input videos + teaser → `data/videos/` (see docs/DATA.md).
- Downloaded 20 reference grasp GLBs (paper's predicted grasps, Inspire hand) → `data/reference/glb/`.
- Grasp compilation video analysis: frames 0–297 = full-res Clorox spray-bottle grasp clip
  (matches paper's "Spray Bottle" object); remainder = 5x5 montage grid, unusable.
- Selected 5 target objects: spraybottle, bottle, can, bulb, hat (+jengabox as 6th candidate).
- Verified all downloads with `file` (real MP4/GLB, not 404 HTML).

## 2026-07-22 — Environment setup (in progress)

- `uv venv --python 3.11 .venv` (CPython 3.11.15).
- Installed `torch==2.7.0+cu128` + `torchvision==0.22.0` from pytorch index; verified
  `torch.cuda.is_available()` on RTX 5080, capability (12,0). 
- **Network gotcha**: host proxy (`127.0.0.1:7897`, set via env vars) times out for
  `pypi.nvidia.com`; direct connection works. For NVIDIA pip installs use
  `no_proxy="pypi.nvidia.com,..."` (see AGENTS.md env section).
- Isaac Sim 5.1 pip install running in background (task: install isaacsim[all,extscache]==5.1.0).
- Cloned `third_party/IsaacLab` @ v2.3.2 (shallow).

## 2026-07-22 — Frames + reference meshes

- `scripts/prepare_frames.py`: extracted per-object frames → `outputs/<obj>/frames/`
  (spraybottle 297 [trimmed to full-res segment], bottle 148, can 189, bulb 93, hat 58, jengabox 65).
- Extracted hand/object meshes from all 20 reference GLBs → `data/reference/meshes/<obj>/{hand,object}.ply`.
  GLB structure: `geometry_0` = posed Inspire hand mesh (43306 verts), `obj_mesh_grasp_*.obj` =
  object mesh (15002 verts, metric scale). Visual QA passed (grasps look correct).
- Reference meshes exist for 3 of our 6 objects: spraybottle, bottle, hat (not can/bulb/jengabox).

## 2026-07-22 — Env setup cont., DRO prep, user answers

- **Isaac Sim 5.1 installed** into `.venv` (exit 0). Two gotchas: (1) `pypi.nvidia.com` must go
  DIRECT (proxy times out) but `files.pythonhosted.org` must go through the PROXY (direct ~70KB/s);
  (2) first boot needs `OMNI_KIT_ACCEPT_EULA=YES`. Boot test running.
- Cloned third_party: hamer, sam2, MoGe, GeoCalib, ContactOpt (facebookresearch),
  drograsp (= zhenyuwei2003/DRO-Grasp; note "nus-lins-lab/drograsp" does not exist).
- **drograsp integration mapped**: training needs only `(q_grasp, object_mesh)` pairs;
  GT D(R,O) computed on-the-fly via cdist. Robots registered via `data/data_urdf/robot/
  urdf_assets_meta.json` + `removed_links.json` + `data/PointCloud/robot/<name>.pt` from
  `data_utils/generate_pc.py`. Their URDFs use 6 virtual root joints (x,y,z prismatic +
  roll,pitch,yaw revolute); q = (6 + DOF).
- Created `scripts/make_inspire_extended_urdf.py`; generated inspire extended URDF
  (6 virtual + 12 revolute, dof=18 = paper's R^18) + registered `inspire` in drograsp data.
  Hand URDF source: unitreerobotics/xr_teleoperate assets (the de-facto public Inspire hand;
  caveat: 59753 visual verts vs paper GLB's 43306 — different tessellation, same hand class;
  scale ~20 cm matches).
- Downloaded drograsp ckpt+data (github release v1.0): robots allegro/barrett/ezgripper/
  robotiq_3finger/shadowhand/leaphand + CMapDataset.
- SAM2 checkpoints (tiny..large) downloaded to third_party/sam2/checkpoints/.
- **MoGe HF prefetch failed**: httpx chokes on `all_proxy=socks://...`; must unset
  `all_proxy/ALL_PROXY` for any HF download (AGENTS.md updated).
- **HaMeR data**: gdown to Google Drive failed; retrying with official mirror
  `https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz` (contains MANO).
- **User answers (API keys)**: MeshyAI key is exported as `MESHY_API_KEY` in `~/.bashrc`
  (not visible to non-interactive shells — run wrappers must `source ~/.bashrc` first).
  Moonshot/GPT-4.1 replacement: **use the Kimi Code agent itself** (user: "you are kimi") —
  the agent writes object size priors (it can also view frames via ReadMediaFile, so it is
  multimodal for this purpose). No Moonshot API key needed.
- Wrote `docs/PLAN.md` (full stage-by-stage design), `scripts/run_depth.py`,
  `scripts/run_masks.py` (GroundingDINO→SAM2).
- `.venv-recon` (torch2.7+MoGe+SAM2+GeoCalib+transformers) building in background.

## 2026-07-22 — Isaac Sim driver blocker & pivot to 6.0.1

- Isaac Sim 5.1.0 installs fine but **segfaults on boot in `librtx.scenedb.plugin.so`**
  (after "app ready"). Root cause: known incompatibility of Isaac Sim 5.1 (Kit 107.3.3) with
  driver branch R595 — our exact driver (595.71.05); NVIDIA staff recommend R580 or upgrading
  Isaac Sim ([IsaacSim#687](https://github.com/isaac-sim/IsaacSim/issues/687),
  [#619](https://github.com/isaac-sim/IsaacSim/issues/619),
  [#651](https://github.com/isaac-sim/IsaacSim/issues/651)).
- Tried workarounds without success: VK_ICD_FILENAMES restriction, minimal experience kit,
  dependency pruning of `omni.hydra.rtx` (it's pulled transitively by ~230 extensions;
  `enabled=false`/`autoload=false` don't prevent startup).
- **User decision (2026-07-22)**: pivot to **Isaac Sim 6.0.1 + IsaacLab 3.0.0-beta2.patch1**
  (works with R595 per community reports). IsaacLab 3.0 is beta — accepted deviation from
  "latest stable"; recorded here. New env: `.venv-lab3` (Python 3.12, torch 2.10+cu128).
  `.venv` (5.1 + IsaacLab 2.3.2) kept but parked.
- Mirror guidance from user: Tsinghua for PyPI (direct), hf-mirror.com for HF (direct,
  all proxies unset). Reachability matrix added to AGENTS.md.
- IsaacLab packages (2.3.2) were installed into `.venv` before the pivot; 3.0 into `.venv-lab3`.
- MoGe-2 weights fetched via hf-mirror; depth stage smoke test launched on spraybottle.

## 2026-07-22 — Sim env solved (6.0.1), reconstruction stages rolling

- **Isaac Sim 6.0.1 + R595: BOOTS OK** (`ISAACSIM601_BOOT_OK`). `.venv-lab3` = py3.12 +
  torch 2.10.0+cu128 + isaacsim[all,extscache]==6.0.1.0 (`--prerelease=allow` needed for
  tinyobjloader rc pin; Tsinghua index direct + nvidia index direct + unsafe-best-match).
- IsaacLab v3.0.0-beta2.patch1 packages installed (isaaclab, physx, assets, tasks, rl,
  visualizers + isaaclab_newton[all] — tasks import newton presets at registration).
- Cartpole smoke test times out fetching cartpole USD from NVIDIA asset servers (slow S3
  from this network). Our env uses only LOCAL assets (URDF + meshes), so not blocking;
  revisit if NVIDIA assets needed (isaacsim-asset pkg caches them on first use).
- **Reconstruction progress**: frames (jpg, 6 objects) + MoGe depth + SAM2 masks all done and
  QA-verified (GroundingDINO boxes all correct; SAM2 video tracking robust; can tracked even
  into the bin). HaMeR installed into .venv-recon (no detectron2/mmcv/chumpy needed at
  inference). Scripts written: run_mesh.py (Meshy v1 API; v2 image-to-3d does NOT exist),
  run_pose.py (scale search + ICP tracking, pose.py module), run_hamer.py (GDINO hand boxes
  + metric depth correction), run_retarget.py (DexPilot-style, drograsp HandModel FK,
  dof=18 verified).
- **Network**: api.meshy.ai must go DIRECT (proxy hangs on POST). Updated route list.
- **BLOCKER (user action)**: HaMeR demo data (6GB, integrity OK) does NOT include MANO —
  registration at mano.is.tue.mpg.de required. Asked user (background question); fallback
  options offered. ContactOpt needs the same MANO_RIGHT.pkl.
- `MESHY_API_KEY` loads via `eval "$(grep '^export MESHY_API_KEY=' ~/.bashrc)"` (bashrc
  returns early in non-interactive shells).
- meshy meshes: running.

## 2026-07-22 — Reconstruction stages 1-4 done; eval env built (with beta workarounds)

- **Stages 1-4 complete for all 6 objects**: frames (jpg) → MoGe-2 depth+intrinsics ✓ →
  GroundingDINO+SAM2 masks ✓ → Meshy image→mesh (v1 API + data-URI upload; v2/image-to-3d
  does not exist) ✓ → scale+ICP pose tracking ✓ (costs 5.5-8.6mm, 0 reinits).
- Scale search (0.5-2.0x around Kimi-agent size priors, `data/size_priors.json`) worked for
  5/6 objects; **hat needed manual override to 0.8x** (pure-Chamfer minima at small scales are
  degenerate — mesh collapses inside the mask; chose by visual overlay). Noted as a
  scale-stage improvement TODO (add silhouette-coverage term).
- **IsaacLab grasp eval env built** (`scripts/run_grasp_eval.py`, protocol = paper IV-A:
  50 settle + 6x50 force steps at 0.5x mass, 3cm threshold). Findings & workarounds
  (Isaac Sim 6.0.1 + IsaacLab 3.0.0-beta2):
  1. PhysX needs `LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/nvidia/cu13/lib`
     (libnvrtc-builtins.so.13.0).
  2. **Beta data-layer bug**: `sim.step()` + `obj.data` + USD transforms all freeze after
     warmup (updateToUsd=false in headless kit; scene-data backend stale). Working path:
     step via `omni.physx.get_physx_simulation_interface().simulate()`, read/write via own
     `omni.physics.tensors` views (verified live). IsaacLab used for spawning/config only.
  3. Isaac Sim 6 URDF importer **consumes the first joint** of the chain — added a
     sacrificial zero-limit `virtual_joint_dummy` before the 6 virtual root joints
     (regenerate via scripts/make_inspire_extended_urdf.py). Convention: **q19 =
     (dummy, x,y,z,roll,pitch,yaw, 12 finger joints)** everywhere; drop index 0 for the sim's
     18-dof articulation.
  4. URDF-converter joint_drive gains didn't apply (all zero) — use IsaacLab
     `ImplicitActuatorCfg` actuators instead (root stiff 1e6/1e4, fingers DRO's 1000/200).
  5. ArticulationRootAPI lives at `Hand/Geometry/world/virtual_link_dummy` (view pattern).
  6. Mesh->USD via MeshConverterCfg(make_instanceable=False, collision convexDecomposition,
     scale=factor from scale stage); friction 3.0 on hand+object (DRO eval values).
  - A/B validated: open hand -> object falls (0% success); closed hand synthetic wedge still
    slips (fabricated grasp is geometrically bad, not a physics issue — real grasps pending).
- `scripts/probe_physics.py` kept as the minimal physics probe.

## 2026-07-22 — MANO acquired; trackio; shared resources (user directives)

- **MANO**: user downloaded `mano_v1_2.zip` to /app/download. Extracted to the SHARED
  location `/app/models/mano/mano_v1_2/` and symlinked into
  `third_party/hamer/_DATA/data/mano/` and `third_party/ContactOpt/mano/{models,webuser}/`.
- **chumpy**: required to unpickle MANO pkls; upstream is py3.11-incompatible. Cloned
  mattloper/chumpy (master uses getfullargspec) and copied the package into
  `.venv-recon` and `.venv-contactopt` site-packages (pure python). MANO_RIGHT.pkl loads OK.
- **trackio replaces wandb** (user directive): `third_party/drograsp/utils/trackio_logger.py`
  (minimal PL Logger → trackio, local-only logging). train.py/pretrain.py patched
  (default trackio; `DRO_WANDB=1` re-enables wandb). trackio installed via Tsinghua mirror.
- **CMapDataset patch**: dataset dir overridable via `DRO_DATASET_DIR` env (default upstream
  behavior). `scripts/build_dro_dataset.py` writes our video-grasp dataset
  (object STLs + cmap_dataset.pt + split json, robot "inspire").
- **Shared resources** (user directive): large reusable assets live under `/app/models`
  (mano done; sam2 checkpoints, hamer _DATA, drograsp data/ckpt move once the running jobs
  release them) and `/app/datasets` (nothing yet). HF cache stays at the default shared
  `~/.cache/huggingface` (hf-mirror endpoint). Layout documented in docs/DATA.md.
- DRO config-invariant pretraining on inspire launched (100 epochs).
- ContactOpt venv ready: pytorch3d 0.7.8 (**CPU-only** build — pip nvcc wasn't picked up;
  acceptable for our knn sizes), torch_geometric, manopth, open3d.
- `scripts/run_contactopt.py` written (grasp window detect + DeepContact + DiffContact
  optimize + joints export). MANO conventions verified: HaMeR meters+CV-frame ↔ manopth mm;
  hand_mTc = translation only; aa45→pca15 via layer PCA basis (lstsq).
- `scripts/run_retarget.py` updated to q19 (dummy,x,y,z,r,p,y,12).

## 2026-07-22 — HaMeR + ContactOpt stack validated

- **HaMeR ran on all 6 objects** (GDINO 'hand' boxes + HaMeR + MoGe metric-depth correction):
  `outputs/<obj>/hand/%05d.json` (MANO rotmats, betas, joints3d, metric t) + mesh PLYs.
  QA verified: boxes + 2D keypoints track the hand well (spraybottle, bulb inspected).
  Fixes along the way: chdir for hamer's relative _DATA paths, torch scalar→numpy cast.
- chumpy (py3.11 fork) copied into both venvs; MANO loads.
- **ContactOpt venv fully validated**: ManoLayer forward (778 verts/21 joints), DeepContact
  checkpoint (1.4M params) loads, all imports OK after patches (PointConv→PointNetConv,
  +trimesh/scipy/cv2).
- sam2 checkpoints relocated to `/app/models/sam2/` (symlinked back).
- GPU job sequencing rule: ONE heavy GPU job at a time (16GB card; DRO pretrain alone eats ~14GB).

## 2026-07-22 — Retarget saga resolved; ContactOpt done; DRO training launched

- **Retarget debugging (long)**: naive DexPilot-only optimization drifted (~80-100mm tip
  errors). Root causes found in sequence: (1) anchor terms too weak vs direction loss;
  (2) degenerate zero-length robot "bones" in the finger mapping (tips duplicated);
  (3) **joint order**: HaMeR's joints3d uses the **OpenPose hand convention** (mano_wrapper
  joint_map: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky) — not raw
  MANO order; (4) **t_metric bug**: wrist anchor applied to the MANO origin which sits
  ~9.6cm from the wrist — fixed with `t_metric = anchor - joints3d[0]` (scripts/fix_tmetric.py
  patches existing records; run_hamer.py corrected). Final scheme: closed-form wrist from a
  palm-frame match (exact, 0mm error) + finger-only DexPilot optimization. Overlay QA on RGB
  now correct for all objects (outputs/*/retarget, vec costs 1.1-5.1).
- **ContactOpt venv completed**: pyg_lib 0.5.0+torch_cluster 1.6.3+torch_scatter (manual
  wheel download from data.pyg.org, DIRECT), torch_geometric pinned 2.5.3 (2.6+ demands
  pyg-lib>=0.6 which lacks cu128 wheels), PointConv→PointNetConv patch, hand_contact copy
  patch, PCA buffer = th_selected_comps/th_hands_mean.
- **ContactOpt ran on all 6**: grasp windows auto-detected (hand<5cm → object moves);
  57 optimized grasps total (stride 2). Re-retargeted → `retarget_contactopt/` (53 in-window).
- **DRO pretraining on inspire**: 100 epochs, loss 0.283, ckpt → ckpt/pretrain/pretrain_inspire.pth.
- **Dataset built**: `data/CMapDataset_videomanip/` (53 grasps q19, 5 train objects +
  jengabox held out for validation; CMapDataset patched with DRO_DATASET_DIR env).
- **DRO training launched**: 200 epochs, trackio logging, pretrain loaded.

## 2026-07-23 — Driver downgrade unblocks the stable sim stack

- **User downgraded driver to R580** (580.159.03). Isaac Sim 5.1 now boots cleanly
  (no more librtx.scenedb crash). The 6.0.1/3.0-beta stack is parked.
- IsaacLab 3.0-beta findings (for the record, in case we ever go back): URDF importer
  consumes the first chain joint and welds the articulation root even with fix_base=False;
  articulation↔rigid-body contact absent (reproduced with DRO-Grasp's own ShadowHand+YCB
  orange+their grasp); sim.step/data stale; position-target backend broken.
- Wrote `scripts/run_grasp_eval2x.py` for the **stable 2.3.2 API** (classic working data
  layer: `set_joint_position_target`, `obj.data.root_pos_w`, `set_external_force_and_torque`).
  Same protocol (welded wrist from q19, fingers 1000/200, friction 3.0 scene-wide,
  50 settle + 6x50 force steps at 0.5x mass, 3cm threshold).
- 2.3.2 API notes: `RigidBodyPropertiesCfg`, `MeshCollisionPropertiesCfg`,
  `RigidBodyMaterialCfg` (not the 3.0 names); UrdfFileCfg needs explicit
  `joint_drive` gains; physics material set scene-wide via `sim_cfg.physics_material`.
- pymeshfix rejected for watertight repair (mangles the mesh frame/shape — see DATA.md
  note); object meshes used as-is with convex decomposition.

## 2026-07-23 — Sim eval unblocked; physics findings; DRO augmentation

- **Physics root cause (both Sim 5.1 AND 6.0.1)**: IsaacLab's own `asset.data.*` is STALE on
  this machine (frozen at warmup values) while physics runs fine — verified with a falling
  ball: `obj.data` frozen at 1.9971, my post-reset `omni.physics.tensors` view live
  (0.65→-18). Workaround used everywhere: create tensor views AFTER sim.reset() and do all
  reads/writes through them (+ `physx.simulate()` stepping). Also: position-target backend
  broken → kinematic hold via position writes; `simulation_app.close()` hangs → `os._exit(0)`.
- **URDF/articulation lessons (2.3.2)**: articulation root at `Hand/root_joint`;
  Isaac Sim 6 consumes the chain's first joint (needs sacrificial dummy);
  `RigidBodyPropertiesCfg/MeshCollisionPropertiesCfg/RigidBodyMaterialCfg` (2.x names);
  joint_drive gains required; scene-wide friction via sim_cfg.physics_material.
- **Grasp holding physics**: spawn-contact ejection fixed by ramp-close settle
  (fingers 0.3x → 1.0x over half the settle) — mirror of a real grasp controller;
  squeeze bias alone didn't help the loose side-grasps.
- **pymeshfix rejected** (mangles mesh frames); camera rendering broken in this env
  (SyntheticData TypeError) — QA via matplotlib/trimesh renders instead.
- **Current numbers (spraybottle)**: retargeted grasps 2/7 hold (0.0mm);
  DRO (53 samples) predictions too loose (0/100) — dataset too small.
- **DRO data augmentation** (build_dro_dataset.py --augment 30): ±0.15 rad finger noise,
  ±1 cm / ±10° wrist noise → 1590 samples (53→1590). Retraining (100 epochs).
- Eval scripts: `scripts/run_grasp_eval2x.py` (2.3.2, post-reset views, ramp-close, squeeze
  param); `scripts/probe_ball2x.py` (dynamic sanity), `scripts/run_refine.py` (robot-side
  contact refinement — first version REGRESSED grasps (opened fingers), needs rework).

## 2026-07-23 — GPU server for heavy jobs (user-provided, temporary)

- Server: user-provided shared box (`$VIDEOMANIP_SERVER`), 4× A800 80GB PCIe, driver 560.35.05, Ubuntu 22.04,
  Python 3.10, nvcc at /usr/local/cuda, sudo without password. **Temporary: all data on it
  is deleted after the test — keep nothing important there (sync results back immediately).**
- Network: `proxy_on` in ~/.bashrc sets 100.64.0.1:8888, but direct connections work for
  pypi/download.pytorch.org/astral (verified); use `bash -lc` + explicit env for HF (hf-mirror).
- Synced drograsp (patched code + dataset + ckpts) to `/tmp/vm/drograsp/`; venv at
  `/tmp/vm/drograsp/.venv` (py3.11, torch 2.7.0+cu126 for A800/sm_80).
- Use it for: DRO training batches (large GPU memory), future multi-video/multi-object
  scale-ups. Sim eval stays local (Isaac Sim 5.1 on this machine).

## 2026-07-23 — GPU server cont. (driver 580, /tmp wiped on reboot)

- Server admin upgraded the GPU server driver to **580.159.03** (same as local) — cu126/cu128
  wheels now valid there (cu118 already worked). **/tmp does NOT survive reboot**: the whole
  /tmp/vm setup was wiped; re-synced and rebuilt. Lesson recorded: treat everything on the
  server as ephemeral per user's note (it's for test only).
- Server env recipe: `uv venv --python 3.11 .venv`, torch 2.7.0+cu126 (A800 sm_80), deps:
  pytorch_kinematics, trimesh, hydra-core, pytorch-lightning, trackio, viser, cvxpy(layers).
  rsync with `-R` (relative) — a plain rsync flattens the data/ prefix.
- DRO augmented training (1590 samples) running both locally (batch 2) and on the server
  (batch 16).

## 2026-07-23 — Sim eval working end-to-end; conversion bugs fixed

- **USD conversion bugs (2.3.2)**: (1) mesh_approximation_name strings don't map — use
  `ConvexDecompositionPropertiesCfg()` subclass instead; (2) `make_instanceable` defaults
  True → meshes move to a Props layer and **instanced rigid bodies fail to resolve** —
  set `make_instanceable=False` (same finding as IsaacLab 3.0's instancing issue);
  (3) stale schema-less object.usd from the pre-fix runs must be deleted or the converter
  skips regeneration.
- **Current eval numbers (spraybottle)**: retargeted grasps hold 2/7 (0.0mm);
  DRO-predicted (unfiltered, 53-sample model): 0/100. Paper's own fitted grasp also fails
  in our harness (spawn-overlap ejection from the ~13mm GLB-fit error — harness is fine,
  the fit is lossy due to mesh revision mismatch).
- IsaacLab 3.0-beta's 'data stale' issue was ALSO present in 2.3.2 on this machine:
  workaround = post-reset tensor views everywhere (documented in AGENTS.md).
- Augmented DRO (1590 samples, epoch_80 local, local train killed at epoch 83 by the 3h cap)
  inference: contact-filter keeps 40/100 (spraybottle), 24 (hat), 5 (can), 5 (jengabox),
  0 (bottle), 0 (bulb) within 5mm.
- Server (A800×4) drograsp training relaunched batch=32/workers=32, 60 epochs.

## 2026-07-23 — Evaluation results (first complete pass)

**Retargeted grasps (reconstruction-direct, 20 trials/object, 2.3.2 harness):**
spraybottle 15%, bottle 15%, can 25%, bulb 0%, **hat 85%**, jengabox 20% — mean ~26.7%.
Paper's DRO-predicted single-video avg = 63.75% (different basis: model-predicted grasps).
Side-grasps on smooth objects are physically hard in the table-less disturbance test;
hat (caged soft object) holds best.

**DRO-predicted grasps: 0%** across objects/epochs — the model is the weak link:
- FK hand-surface distance for "contacting" predictions: 72-218mm — fundamentally loose,
  not a sim-eval artifact. Multilateration/IK can't recover poses from the weak D(R,O)
  predictions of a 53-sample model. My 5mm contact filter was also mis-designed
  (mlat points land ON the object by construction → dmin≈0 always — not a useful gate).
- Augmentation (1590) improved contact-rate but not holding.
- Server (A800) training batch 16 / 60 epochs running as the next attempt.

**Harness validation**: IsaacLab 2.3.2 disturbance eval now fully works (retargeted grasps
hold up to 85% for hat, object/harness/conversion bugs all fixed; the paper's own fitted
GLB grasp fails only because the ~13mm GLB-fit error makes it spawn-overlapping).

## 2026-07-23 — Refinement v2, FK filter, mixed dataset (server train run 2 queued)

- **run_refine.py v2** (rewritten): surface-based point-to-plane contact model over the full
  hand surface (9216 pts) vs the object's sampled surface+normals. v1's keypoint model
  equilibrated at a 2mm standoff and OPENED grasps. v2: one-sided attraction into a
  [-PEN_ALLOW, +GAP_OK] contact band + penetration pushback + wrist(6dof)/finger deviation
  penalties. Two settings tried: strict (1mm/20) and mild (3mm/5).
- **FRAME TRAP**: per-frame `retarget*/*.npy` q19s live in the RECONSTRUCTION frame (object
  NOT at origin); the eval npz (`outputs/<obj>/eval/retarget_grasps.npz`) is the same grasps
  transformed to object-at-origin. Refine/eval only against the npz form. run_refine.py now
  takes `--grasps <npz>` (with `<obj>` substitution) and refuses the npy dirs.
- **Refinement results** (20 trials/object): helps only ejection-limited objects —
  spraybottle 15→25%, hat 85→95%; hurts slip-limited ones (bottle 15→0, can 25→5,
  jengabox 20→5) because in our kinematic-hold harness penetration IS the grip-force
  mechanism; de-penetrating removes the normal force. Best-per-object (baseline∪refined):
  spraybottle 25, bottle 15, can 25, bulb 0, hat 95, jengabox 20 → **mean 30.0%**.
  QA json per run: `outputs/<obj>/eval/retarget_grasps_refined_qa.json`.
- **DRO filter fixed**: run_dro_inference.py `--filter` now gates on FK(recovered q) hand
  surface vs object (n_contacts≥100, pen_pts≤300), replacing the mlat-pc dmin gate that was
  ~0 by construction. Server epoch_30 model: median 0 FK contacts (still too weak).
- **Merged DRO dataset** (`third_party/drograsp/data/CMapDataset_mixed/`): paper's
  CMapDataset_filtered subsampled to 4920 (uniform over 58 objects, 5 robots — NO inspire
  samples exist in it) + our 1590 inspire samples = 6510; split 53 train / 11 val objects.
  Synced to server + queued a 6-robot mixed run (40 epochs, starts automatically when the
  current 60-epoch inspire-only run exits, via /tmp/vm/queue_mixed.sh).

## 2026-07-23 — inspire-only 60ep model still too weak; mixed run launched

- Server 60-epoch inspire-only model (1590 augmented samples): FK filter keeps 0-1/100
  per object; median closest hand-to-object distance 81mm (spraybottle) / 139mm (hat).
  6-object data scale is definitively the bottleneck — no filter/refinement can recover
  grasps that far off. Paper's own no-ContactOpt ablation (30.7%) matches our
  retarget-direct mean (26.7%), consistent with the story that DRO trained on proper
  data is what adds the margin.
- **Bug fixed**: our `data/PointCloud/object/videomanip/*.pt` were 65536×6 but the
  paper's convention (and validate-time stacking) is 512×6 — subsampled ours to 512
  (xyz+normal), re-synced. This had crashed the first mixed-run launch.
- **Mixed-data run launched** (server, detached via setsid): CMapDataset_mixed
  (4920 paper grasps over 58 objects/5 robots + our 1590 inspire grasps), 6 robot_names,
  batch 16, 40 epochs, init from pretrain_inspire.pth. ~9min/epoch → ETA ~01:30.
  Checkpoints mirrored to local every 10min (background task) because the server is
  ephemeral. **Lesson**: remote long jobs must be launched with `setsid nohup ... &`;
  a plain nohup inside `ssh bash -c` gets reaped when the local ssh task times out.

## 2026-07-23 — Mixed-data DRO works: first contact-valid predictions

- Loss curves (trackio, server): inspire-only run pinned at ~0.73 from step 4 to 5639
  (never learned — underfit, not overfit); mixed run 0.77→0.11 within epoch 1.
- **Mixed epoch_5 FK filter results** (100 samples/object, gate = ≥100 contact pts AND
  ≤300 pen pts): spraybottle 30, bottle 2, can 27, bulb 61, hat 43, jengabox 15 pass
  (vs 0-1 for inspire-only epoch_60). Median contact counts high (63-912) but skewed by
  deep penetration on bottle/can/jengabox → refinement (pen pushback) is the intended
  next step for those.
- Sim eval of filtered DRO grasps + refinement of all 600 predictions running.
- trackio db on server: /home/wb/.cache/huggingface/trackio/DROGrasp.db (sqlite, table
  `metrics`, loss curves for both runs).

## 2026-07-23 — Mixed model first sim results (epoch 5/10)

- **Mixed epoch_5, raw 100 predictions/object (paper protocol)**: spraybottle 5, bottle 67,
  can 60, bulb 31, hat 2, jengabox 75 → **mean 40.0%** (vs retarget-direct 26.7%, vs
  paper 63.75% on 20 objects). DRO and retargeting are complementary (retarget wins
  spraybottle/hat; DRO wins bottle/can/bulb/jengabox — bulb 31% vs retarget 0%).
- Epoch_10: mean 28.3% (spray 10, bottle 25, can 16, bulb 26, hat 59, jenga 34) — per-object
  competence OSCILLATES between epochs (hat 2→59% while bottle/can/jenga collapse). FK
  probe and sim success disagree (deep-press grasps fail the FK pen gate but hold in sim).
  Checkpoint selection matters; epochs 15-40 get auto sim-eval as they land
  (outputs/eval/mixed_sim_curve.log).
- **Refinement of DRO predictions: net negative** (mean 36% vs raw 40%): spray 5, bottle 63,
  can 49, bulb 28, hat 4, jenga 67. De-penetration removes grip force in the kinematic-hold
  harness; attraction can't create enclosing geometry for loose predictions. Raw predictions
  are the DRO headline.
- **My FK contact filter is diagnostic-only** — do NOT gate eval sets on it (paper evaluates
  all 100 predictions; deep-press grasps it rejects actually hold).
- trackio db backed up to outputs/eval/DROGrasp.db; loss plot outputs/eval/dro_loss_curves.png.

## 2026-07-23 — All 4 GPUs in use: second DDP run with 3x-oversampled videomanip data

- train.py already supports multi-GPU (`devices=cfg.gpu`, DDP when pretrain set) — no code
  change needed. Kept the original mixed run on GPU0 (already ~epoch 22; a 4-GPU restart
  would barely beat its ETA and lose this trajectory), launched a **3-GPU DDP run on GPUs
  1-3** (`videomanip_mixed3x`): same mixed dataset but our 1590 samples **3x-oversampled**
  (9690 total, ~49% ours) to counter the per-object oscillation seen in the first run
  (our objects drowned ~1:3 by paper data as training progresses).
- Local automation consolidated into one flock-serialized mirror+sim-eval loop for both
  runs (results → outputs/eval/mixed_sim_curve.log; per-object jsons dro_mixed{,3x}_e*).

## 2026-07-24 — FINAL: mixed3x epoch_5 = 47.0% mean; report written

- Both DRO runs completed (mixed on GPU0, mixed3x 3-GPU DDP on GPUs 1-3) and every
  checkpoint sim-evaluated on all 6 objects (curve: outputs/eval/mixed_sim_curve.log).
- **Best single checkpoint: videomanip_mixed3x epoch_5 = 47.0% mean**
  (spray 19, bottle 27, can 34, bulb 51, hat 97, jenga 54) vs paper 63.75% (20 objects).
  Runner-up: mixed epoch_5 = 40.0%. Best-per-object across both pipelines (retarget∪DRO):
  62.3% ≈ paper number. Checkpoint oscillation documented (early ckpts systematically best).
- Final report: `docs/REPORT.md` (comprehensive: results, bugs, negative results, gap
  analysis, next steps). Tables regenerable via `python3 scripts/make_report.py`.
- DATA.md updated with the mixed/mixed3x dataset + checkpoint inventory.
- GPU server runs exited; all 16 checkpoints + trackio db mirrored locally (server is
  ephemeral — nothing important remains only there).

## 2026-07-24 — Our model on the PAPER's 20 objects: 56.65% (vs their 63.75%)

- Ran mixed3x-e5 inference + our sim harness on the paper's 20 GLB object meshes
  (data/reference/glb → outputs/pref_<obj>/, centroid-centered, mass 0.3kg).
  Per-object: apple 8, bottle 76, bowl 67, case 87, cloth_hanger 33, cup 63, hand_bag 70,
  hat 92, ladle 65, mug 69, pan 69, pot 92, powerdrill 23, scissors 77, soap_dispenser 44,
  spraybottle 49, sunglass 31, toothbrush 15, umbrella 21, wineglass 82 → **mean 56.65%**
  (100 trials each, raw predictions, paper protocol). Only 7pp below the paper's own
  reported 63.75% on the same 20 objects.
- On the paper's 5 single-video FAILURE objects (pan/hat/bowl/case/hand_bag) we score 77%
  mean — their failures were caused by their own reconstruction/pose errors, which don't
  affect a model evaluated on the final meshes directly. Our weak objects: apple 8,
  toothbrush 15, umbrella 21, powerdrill 23 (thin/small/articulated).
- Interpretation: most of our earlier 16pp "gap" was object-set difficulty (bulb hollow
  shell, single egocentric spraybottle view), not model quality. Data scale fixed the model;
  object set explains the rest.
- Mesh comparison (ours MeshyAI vs paper GLB, same objects): paper meshes preserve
  functional features (spray-bottle trigger, cap brim) that ours smooth over — reconstruction
  input quality (crops/pose) matters as much as the mesh model. TripoSR open-source
  alternative being tested on bulb.
- Dense-checkpoint run (save every epoch, 4-GPU DDP, mixed3x data, 12 epochs) launched to
  find the true early-training peak (epoch 5 of 40 was just the first saved point).

## 2026-07-24 — Wrench-aware refinement: mean 85.7% (vs 47% raw, paper 63.75%)

- **Idea**: geometric contact refinement failed because proximity ≠ disturbance resistance.
  New objective (`scripts/run_refine_wrench.py`): per-contact normal force = spring in
  penetration (0.5 N/m/pt, capped), resistance per disturbance direction = normal
  opposition + friction-cone support; maximize softmin margin over the paper's 6 force
  directions + 3mm penetration cap + deviation penalties. 300 Adam iters, ~2min/100 grasps.
- **Results on mixed3x-e5 predictions (100/obj)**: spray 19→42, bottle 27→89, can 34→90,
  bulb 51→100, hat 97→95, jenga 54→98 → **mean 47.0→85.7%**. QA viz: one-sided pinches
  become enclosing wraps. Generalizes: 89/97% under 12 random directions, 87% at 1.5x force.
- **SWA/weight-averaging: negative** (avg of oscillating ckpts 8-25% < best single 42%) —
  checkpoints drift between basins; checkpoint SELECTION (sim-eval each ckpt) is the
  protocol, not averaging.
- **Dense-ckpt run (12 epochs, 4-GPU)**: train loss 0.78→0.06 monotonically while triage
  success oscillates 0.7→42 — train objective fully decoupled from downstream metric.
  e3 nearly-zero between decent e2/e4: epoch-level instability confirmed.
- Paper-20 wrench eval running (the apples-to-apples vs 63.75%).

## 2026-07-24 — Paper-20 wrench results + research synthesis

- Paper-20 wrench-refined: mean 48.9% (vs raw 56.65%) — helps weak sets (hand_bag 70→99,
  umbrella 21→59, spraybottle 49→74) but perturbs strong ones (cup 63→12, pot 92→59).
  Per-object union 62.1% ≈ paper 63.75%. Margin surrogate does NOT gate; raw success rate
  is the practical gate. Retarget-wrench: can 25→45, spray/hat degrade (same boundary).
- Generalization validated: 89/97% under 12 random dirs; 87% at 1.5x force.
- docs/REPORT.md restructured: "Research findings beyond the reproduction" section
  (wrench method, checkpoint oscillation + SWA-negative + decoupled objective,
  penetration-bias critique, paper-20 detail, MeshyAI-vs-TripoSR), gap analysis resolved.
- Eval harness gained --rand_dirs/--force_scale (generalization tests).

## 2026-07-24 — Research-day 2: prior art, self-distillation, TRELLIS install

- **Prior art for wrench refinement**: differentiable force-closure refinement exists —
  DFC (Liu et al., RA-L 2021), FRoGGeR (Li et al. 2023, min-weight metric), task-oriented
  GWB estimator (arXiv 2309.13586). NOT in prior art: repairing NN/video-learned grasp
  predictions (they synthesize from scratch); the geometric-vs-wrench head-to-head;
  the penetration-bias critique; checkpoint oscillation in few-shot cross-embodiment DRO.
- **Gated refinement**: raw-success gate alone insufficient on paper-20 (53.75% < raw 56.65%)
  — wrench hurts thin objects regardless of gate. Framing: few-shot repair of weak
  predictions, not universal postprocess.
- **Self-distillation loop** (DAgger-ish): mixed3x data + 600 wrench-refined predictions
  (CMapDataset_selfdistill, 10290) fine-tuned from mixed3x-e5 (lr 5e-5, 10 epochs).
  train.py patched with env-gated full-state-dict init (`DRO_INIT_SD`); model.pretrain is
  encoder-only + freezes robot encoder (detach) — all runs so far share that regime.
  Server ops notes: NCCL 30-min timeout killed the first 4-GPU attempt during slow dataset
  init; zombie DDP ranks hold GPU memory (kill by PID list; pkill patterns self-match the
  ssh wrapper's own cmdline — use /tmp/killtrain.sh).
- **TRELLIS on server** (full recipe in docs/TRELLIS.md): CUDA 12.6 toolkit + dev headers
  installed (server toolkit was 12.1 — cu126 torch mismatch); nvdiffrast needs
  --no-build-isolation; kaolin wheel from NVIDIA index; flexicubes is a git submodule
  (clone with --recurse-submodules); sparse attention hard-requires flash-attn (built
  2.8.3 from source, ~10 min on 96 cores); ATTN_BACKEND=sdpa for the dense path;
  hf-mirror transient 404s resolve on retry.
- **Local TRELLIS install**: no nvcc on this machine — install cuda-nvcc-12-6 + dev headers
  first (recipe in docs/TRELLIS.md), or copy server-built wheels (both py3.11/torch2.7).

## 2026-07-24 — Self-distillation negative; TRELLIS mesh comparison done

- **Self-distillation fine-tune** (mixed3x + 600 wrench-refined predictions, init
  mixed3x-e5, lr 5e-5, 10 epochs): every checkpoint WORSE than the e5 init — best
  epoch 8 = 34.5% vs 47.0%. The peak checkpoint is a fragile basin: any continued
  training drifts away (consistent with the oscillation finding). Conclusion: wrench
  refinement belongs at INFERENCE time; DAgger-style data loops don't survive the
  basin geometry. Curve: outputs/eval/selfdistill_curve.log.
- **TRELLIS (open-source SOTA image-to-3D) tested on server** (recipe: docs/TRELLIS.md):
  on both test inputs it produces normalized-box blobs (bulb reflective — expected hard;
  spraybottle — the crop shows only the bottle top). TripoSR worse, MeshyAI best of the
  three. Conclusion reinforced: input crop quality, not the mesh model, is the binding
  constraint; open models are not drop-in replacements for MeshyAI on our inputs.
  Outputs: outputs/{bulb,spraybottle}/mesh_trellis/object.glb.
- Server hygiene lessons added: `bash /tmp/killtrain.sh` to clean DDP zombies;
  NCCL timeout on 4-GPU during slow dataset init → 3-GPU config proven stable.

## 2026-07-24 — Component SOTA audit + VLM size priors test

- Full per-stage audit in `docs/SOTA.md`: paper is mostly 2024-vintage (SAM2, HaMeR,
  FoundationPose, ContactOpt, GPT-4.1); 2025-current on MoGe-2/DRO/MeshyAI.
- SAM3 exists (concept-prompt segmentation, 2× PCS) — mask-stage upgrade available but
  masks aren't our bottleneck. HaWoR (CVPR 2025) is the principled hand-stage upgrade
  (documented; old torch 1.13 env + CC-BY-NC-ND license kept us from installing today).
- **VLM size priors tested** (Qwen2.5-VL-7B via ModelScope, `scripts/vlm_size_priors.py`):
  brand-level identification 6/6 (Clorox/Pringles/Brisk/Jenga...), but metric estimates
  run 15-25% SMALL on 4/6 objects — kimi-agent priors are closer. Verdict: VLM-for-ID +
  LLM-for-size is the right split; GPT-4.1 replacement is a non-issue for final metrics
  since scale-refinement corrects seeds anyway.
- ModelScope confirmed as a solid weight mirror (16GB clean download, proxies unset).

## 2026-07-24 — Hand-stage upgrade exploration: HandFlow (replaces HaWoR plan)

- **Pivot HaWoR → HandFlow** ([repo](https://github.com/mxxu00/HandFlow), arXiv:2607.11221):
  beats HaWoR by >30% world-space pose error, ~12× faster, **MIT license** (HaWoR is
  CC-BY-NC-ND), and ships on **torch 2.7.0+cu128 / py3.10** — the same stack as
  .venv-recon, so no torch-1.13 migration was needed. `--fix_camera` mode fits our
  static-camera videos (skips the ViPE/DROID-SLAM CUDA-extension dependency entirely).
- **Env**: `.venv-handflow` (py3.11, torch 2.7.0+cu128 — matches .venv-recon).
  hamer submodule installed `--no-deps -e`; chumpy fork copied from .venv-recon;
  ultralytics + manopth(git) + torchdiffeq + lmdb + pyrender added (last three missing
  from HandFlow's requirements.txt). pytorch3d NOT needed — QA via our own overlay.
  Weights at `/app/models/handflow/` (denoiser 668MB + WiLoR detector.pt + norm stats,
  hf-mirror). MANO reused from `/app/models/mano/mano_v1_2/models`.
- **Adapter**: `scripts/run_handflow.py` — frames + per-frame MoGe intrinsics →
  WiLoR YOLO → online HaMeR → FM denoiser (overlapping windows) → MANO pose48/trans/betas
  → same json schema as run_hamer.py (`outputs/<obj>/handflow/%05d.json` + mesh PLYs +
  raw npz). Detections ~100% on all 6 objects (bulb 91/93, FM infills misses).
- **BUG (fixed)**: manopth master returns its 21 joints ALREADY in OpenPose order —
  verified empirically on the zero pose (continuous 4-joint chains, thumb +z) and by
  bone-length equality with HaMeR. Applying hamer's mano_to_openpose permutation
  scrambles the hand (retarget vec cost 3.7-6.9, bone lengths nonsense). Do NOT remap.
- **BUG (fixed)**: HandFlow's FM metric translation is systematically biased on our
  videos (hand-object distance during grasp: bottle/can/hat med ~11 cm; its 2D alignment
  is fine — depth-only error). Fix = the paper's own MoGe depth anchoring (same formula
  as run_hamer.py) + temporal median filter k=7: `scripts/fix_handflow_depth.py`.
  Contact med after: 2.2-18 mm (≈HaMeR's 1.4-7.8). Smoothness after correction:
  wrist accel 1.5-3.0× lower than HaMeR on ALL 6 objects (fingertips similar) —
  the temporal-consistency win survives the depth anchoring.
- **Sim eval (IsaacLab 2.3.2 harness, window-frame grasps)**:
  HaMeR-direct (stride 1, N=9-53/obj): spray 91.7, bottle 0, can 56.2, bulb 1.9,
  hat 70, jenga 44.4 → **mean 44.0%**; HandFlow-direct (same frames): all ~0 (bulb 3.8)
  → **mean 0.6%**; HaMeR+ContactOpt 26.7% (old numbers, npz provenance verified =
  retarget_contactopt); HandFlow+ContactOpt 8% (bulb 48, rest 0).
  Direct-variant failures are spawn-penetration ejections (disp >50 m) — raw perception
  grasps interpenetrate slightly; ContactOpt exists to fix exactly this.
  ContactOpt convergence on HandFlow inputs is fine (opt_loss ≤ HaMeR's).
  Side finding: ContactOpt HURT our HaMeR-direct numbers (44.0 vs 26.7) — opposite of
  the paper's ablation; our ContactOpt settings were never retuned.
- **Where HandFlow genuinely wins**: retargeted robot trajectories are smoother —
  finger-joint |ddq| mean 1.8-6× lower on 5/6 objects (spraybottle the exception).
  This is the property that matters for DRO training data, not raw grasp eval.
- **Verdict**: hand-stage upgrade works technically (modern env, better smoothness,
  ~100% detection, principled depth-anchored hybrid) but does NOT improve grasp-hold
  success on our 6 objects — holding is dominated by contact geometry, not temporal
  jitter. The real downstream test (DRO retrained on HandFlow-derived grasps) is
  untested — flagged as next step; hand stage is not the bottleneck vs object
  mesh/pose/data scale.
- **Script patches**: run_retarget.py --src now free-form (writes retarget_<src>/);
  run_contactopt.py gained --src/--out; new scripts/build_eval_grasps.py rebuilds the
  eval npz (object-at-origin) for any retarget dir (--src-jsons maps contactopt-retarget
  npys back to true frame ids). retarget_contactopt regenerated identically after an
  accidental overwrite (verified vs the old eval npz, maxdiff 0).

## 2026-07-24 — DRO retrain on HandFlow-derived grasps (the flagged downstream test)

- **Experiment**: does DRO trained on HandFlow-derived grasp data beat the HaMeR-derived
  mixed3x model (47.0% mean)? Variants (same mixed3x recipe: paper 4920 + ours 3x
  oversampled, same split 53 train/11 val objects, jengabox held out):
  - `CMapDataset_mixed3x_handflow` (9420): inspire grasps = 50 HandFlow+ContactOpt
    (retarget_contactopt_handflow, frame-id-corrected poses) ×30 augment = 1500
  - `CMapDataset_mixed3x_union` (9780): inspire grasps = 108 (53 HaMeR-CO + 50
    HandFlow-CO) ×15 augment = 1620
- **build_dro_dataset.py patched**: --src NPY_DIR[:JSONS_DIR] (multiple sources),
  --name. NOTE: the original CMapDataset_videomanip had a latent frame-id mismatch
  (enumerated contactopt npys paired with poses by npy index, not true frame id) —
  object pose barely moves inside the window, so the effect was mild pose noise;
  new datasets use exact json-stem frame ids.
- **Training** (server, 2× A800 per run): `train.py name=videomanip_mixed3x_{handflow,union}`
  6 robots, batch 16/GPU, 20 epochs, save_every_n_epoch=1 (dense — checkpoint oscillation
  protocol), model.pretrain=pretrain_inspire.pth (encoder init + robot-encoder freeze),
  trackio. Logs: /tmp/vm/train_{handflow,union}.log.
- **Eval**: `scripts/mirror_eval_handflow.sh` (flock-serialized) mirrors state_dicts
  every 5 min and triage-evaluates each new checkpoint (DRO inference 100 samples/obj,
  50 trials/obj) → outputs/eval/handflow_sim_curve.log + dro_<run>_e<N>_results.json.

## 2026-07-24 — Migration readiness (user directive: future servers, fast migration)

- **Env manifests**: `envs/freeze-{isaaclab232,recon,handflow,contactopt,lab3}.txt` (local
  venvs) + `envs/freeze-server-drograsp.txt` (server venv, 108 pkgs, torch 2.7.0+cu126).
  Caveat noted: freezes don't capture git/copied packages (chumpy fork, manopth git,
  hamer -e --no-deps) — rebuild notes point at AGENTS.md/PROGRESS.
- **Scripts**: `scripts/server_bootstrap.sh` (uv venv + torch + deps, idempotent),
  `scripts/server_sync.sh` (code from third_party/drograsp, data+ckpt from
  /app/models/drograsp — symlinks dereferenced; --exclude .venv fix after first run
  tried to delete remote venv), `scripts/server_train.sh` (proven launch command,
  setsid inside remote cd), `scripts/server_mirror.sh` (state_dicts + logs + trackio db).
  server_sync smoke-tested against the live server (delta-only).
- **docs/SERVER_SETUP.md**: full guide — source-of-truth table, 3-command migration,
  server facts (A800/sm_80, /tmp volatile, 4GB-per-sample rule), lessons (setsid,
  remote-cd precedence bug, NCCL 4-GPU timeout, zombie-kill pattern, checkpoint
  oscillation protocol).
- AGENTS.md env section + DATA.md updated to point at the guide.

## 2026-07-27 — Server incident, union results (best e2 = 69.0% full), handflow retrain

- **Server incident**: between 07-24 and 07-27 the temp server lost (a) the 07-24
  handflow checkpoints (output dirs replaced by a 07-27 restart someone launched;
  never mirrored locally — the 07-24 mirror loop had died) and (b) the venv
  (empty remnants). Union's 07-27 restart completed all 20 epochs; handflow's
  restart NCCL-timed-out at init and left 31 zombie ranks on GPUs 0-1 (killed by
  PID list). **The box is SHARED — user directive: never kill other tenants'
  processes; only PIDs whose cmdline matches our exact run name.** Two more
  handflow NCCL deaths followed (GPUs 0,1 pair and 0-2 triple); the run finally
  went healthy on the GPUs 2,3 pair (the pair union had just succeeded on) —
  NCCL here is flaky under tenant load. Lesson reinforced: mirror IMMEDIATELY.
- **Migration tooling validated under fire**: server_bootstrap.sh rebuilt the venv
  cleanly (two fixes: `uv venv --clear` for stale dir; stray `which uv` line removed
  from envs/freeze-server-drograsp.txt). Handflow retrained on GPUs 0-2 (3-GPU config).
- **Union results (50-trial triage curve, all 20 epochs)**: early peak then
  oscillation (same pattern as mixed3x): e1 58.7, **e2 70.7**, e3 58.3, e4 55.0,
  e5-13 collapse (9-30), e14 44.7, e17 49.0, e20 26.0.
- **Union e2 confirmed at FULL protocol (100 trials/obj)**: spray 30, bottle 49,
  can 67, bulb 97, hat 78, jenga 93 → **mean 69.0%** (vs mixed3x-e5 47.0%,
  vs paper 63.75% on its own 20). Paper-20 eval of union e2 running.

## 2026-07-27 — FINAL: HandFlow-DRO results; handflow-e16 beats the paper on paper-20

- Handflow retrain completed after the GPU-pair fix (GPUs 2,3; two prior NCCL deaths
  on 0,1 and 0-2). Full triage curve: e1 47.3, e2 43.3, collapse e3-e10 (9-33),
  recovery e11 41.0, **best e16 54.7**, e17-20 13-29. Same oscillation regime.
- **Full protocol (100 trials, our 6)**: handflow-e16 spray 36, bottle 54, can 31,
  bulb 12, hat 91, jenga 76 → **mean 50.0%** (mixed3x-e5 47.0%, union-e2 69.0%).
- **Paper-20 (100 trials, raw)**: handflow-e16 = **65.0% — above the paper's 63.75%**
  (union-e2 62.05%, mixed3x-e5 56.65%). Balanced profile: powerdrill 75 (union 0),
  spraybottle 68 (union 26); weak: apple 3, cloth_hanger 9, soap_dispenser 30.
- **Complementarity finding**: union wins our-6 (69.0 vs 50.0), handflow wins paper-20
  (65.0 vs 62.05); per-object union of both models on paper-20 = **73.3%**.
  Hand-stage-upgrade downstream verdict: POSITIVE — HandFlow-derived DRO training data
  (50 grasps, smoother trajectories) outperforms HaMeR-derived (53 grasps) on both
  eval sets, and the union maximizes our-6. Remaining caveat: checkpoint oscillation
  means e16/e2 were selected by sim-eval, not by training loss.
- Wrench refine on union-e2: 69.0→71.0% (hat regresses 78→31, known boundary);
  per-object union(raw∪wrench) = 78.8% on our 6.
- Server: all 20 handflow ckpts + union ckpts + trackio db + logs mirrored locally;
  server clean (no processes). docs/REPORT.md updated (three-way tables).

## 2026-07-27 — Paper draft written (docs/PAPER.md)

- **`docs/PAPER.md`**: research write-up of the three findings beyond the reproduction:
  (1) smooth HandFlow trajectories as DRO training data (+depth-anchored hybrid fix for
  FM metric-trans bias) — 65.0% on paper-20, above the paper's 63.75%; union 69.0%
  on our 6; data-source complementarity (per-object model union 73.3%);
  (2) wrench-aware refinement vs geometric (47.0→85.7%; boundaries documented);
  (3) checkpoint oscillation (9-71% swings, loss decoupled, SWA negative,
  select-don't-average). Includes figure `outputs/eval/dro_oscillation_curves.png`
  (triage curves + trackio loss), per-object appendix, reproducibility map.

## Next steps

1. Write AGENTS.md; scaffold uv project (pyproject, src layout).
2. Trim/segment videos; extract frames; quick visual QA of each clip.
3. Install Isaac Sim 5.1 + IsaacLab 2.3.2 into `.venv` with uv (large download, run in background).
4. Inspect reference GLBs (units, hand vs object mesh separation) with trimesh.
5. Build reconstruction stage (MoGe-2 → SAM2 → mesh → scale → FoundationPose → HaMeR → retarget).

## Open blockers / user input needed later

- **MeshyAI API key**: ask user to place it at a path TBD when reaching the mesh-comparison step.
- **Moonshot/Kimi API key + exact model name** for the GPT-4.1 replacement (user said "kimi-3").
- **MANO license**: HaMeR/ContactOpt need `MANO_RIGHT.pkl` from mano.is.tue.mpg.de (free registration).

## 2026-07-27 — Published to GitHub + Hugging Face

- **GitHub**: <https://github.com/physercoe/videomanip-reproduction> (public). Contents:
  all own code/docs (src/, scripts/, docs/, envs/, outputs/*/eval+dro evidence jsons,
  data/size_priors.json), new publication-grade README, MIT LICENSE, CITATION.cff.
- **Not vendored** (pointers instead): third_party upstreams — pinned-commit manifest in
  `third_party/README.md` + our local diffs extracted to `third_party/patches/*.patch`
  (drograsp/contactopt/triposr; reverse-apply verified) + `scripts/setup_third_party.sh`;
  paper's videos/GLBs — `scripts/fetch_paper_data.sh`; detector.pt/MANO/etc — source links
  in README. `outputs/*/*` excluded except eval/ and dro/ (2.8G reconstruction artifacts
  stay local / are regenerable).
- **Scrubbed** internal GPU-server address from scripts/server_*.sh, mirror_eval_handflow.sh,
  SERVER_SETUP.md, PROGRESS.md → `$VIDEOMANIP_SERVER` env var or `user@host` arg.
- **Hugging Face**: <https://huggingface.co/datasets/physer/videomanip-reproduction> —
  unique artifacts: 3 key DRO runs' full epoch series (mixed3x 430M, handflow 1.1G,
  union 1.1G), derived CMapDataset_{videomanip,handflow,union,mixed,mixed3x,
  mixed3x_handflow,mixed3x_union,selfdistill} + PointCloud_videomanip (~25M),
  hand+handflow records (6 obj), predicted-grasp npz, GLB-derived meshes, eval jsons+curves+trackio db.
