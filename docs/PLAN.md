# Pipeline Design

Stage-by-stage design of the reproduction, following the paper (Sec. III) with the agreed
substitutions. Each stage: inputs → outputs (all under `outputs/<obj>/<stage>/`), tool, key
parameters, deviations from paper.

Paper reference: RGB video → MoGe-2 (metric depth + intrinsics) → SAM2 (masks) → MeshyAI
(image→mesh) → GPT-4.1 + FoundationPose-based refinement (scale) → FoundationPose (6D pose)
→ HaMeR (hand mesh) → DexPilot-style retargeting → [GeoCalib gravity align, in-the-wild only]
→ ContactOpt (grasp-stage contact optimization) → D(R,O) training → sim eval.

## Objects

spraybottle (in-scene), bottle (in-scene), can (in-scene), bulb (wild), hat (wild),
jengabox (wild). In-scene = static third-person camera; wild = egocentric, needs GeoCalib.

## Stage 1a — depth+intrinsics (`scripts/run_depth.py`, .venv-recon) ✅ written

MoGe-2 (`Ruicheng/moge-2-vitl-normal`) per frame → `depth/%05d.npy` (float32 m),
`depth/intrinsics.json`. QA sheet `depth/qa.png`. Metric scale comes from MoGe-2 directly
(paper does the same), used for hand-depth correction and object scale refinement.

## Stage 1b — masks (`scripts/run_masks.py`, .venv-recon) ✅ written

GroundingDINO (`IDEA-Research/grounding-dino-tiny`, text prompt per object) on frame 0 →
box prompt; SAM2.1-hiera-large video predictor → `masks/%05d.png`. Deviation: paper does not
specify the prompt mechanism; GroundingDINO automates it.

## Stage 2 — object mesh (image→mesh) — TODO

Best single frame per object (max mask area, sharp) → crop+segment → mesh.
- Primary (paper): **MeshyAI API** (needs `secrets/meshy.env:MESHY_API_KEY`).
- Open-source comparison: **Hunyuan3D-2** (fallback TRELLIS).
Output: `mesh/object.glb` (+ `mesh/qa.png` renders).
Acceptance: visually correct shape; compare against `data/reference/meshes/<obj>/object.ply`
where available (spraybottle/bottle/hat) via Chamfer distance after scale/align.

## Stage 3 — object scale — TODO

Two-stage per paper: (1) coarse physical size prior from **Kimi/Moonshot** (replaces GPT-4.1;
needs `secrets/moonshot.env:MOONSHOT_API_KEY` + model id), e.g. "a Clorox spray bottle is
~27 cm tall"; (2) refine scale over candidates 0.5×–2× by min render-vs-mask error against
SAM2 mask + MoGe metric depth. Output: `scale/scale.json`.

## Stage 4 — object 6D pose track — TODO

Paper: FoundationPose. Install of FoundationPose proper is fragile (nvdiffrast/kaolin builds,
no sm_120 prebuilts); planned fallback: registration-based tracking — frame-0 init by aligning
scaled mesh to masked MoGe depth cloud (centroid/PCA + ICP), then per-frame point-to-plane ICP.
Outputs `pose/%05d.json` (4x4 T_cam_obj). QA: reprojection overlays.
Decision gate: if tracking visibly fails, invest in real FoundationPose (needs pip CUDA toolkit).

## Stage 5 — hand pose (HaMeR) — TODO

.venv-hamer (torch 2.7, smplx, pytorch-lightning; **no detectron2** — deviation: hand boxes
from GroundingDINO "hand" prompt instead of ViTDet/detectron2; model itself unchanged).
Demo data incl. MANO: `gdown 1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT` (hamer_demo_data.tar.gz).
Depth correction per paper: weak-perspective tz replaced by mean MoGe metric depth at 2D kpts.
Outputs `hand/%05d.json` (MANO θ,β, global orient, corrected translation) + meshes.

## Stage 6 — retarget (human→Inspire) — TODO

DexPilot-style optimization: minimize distance between Inspire link keypoints (via URDF FK,
`third_party/xr_teleoperate/.../inspire_hand_right.urdf`, 12 revolute joints) and human hand
joints (MANO), with wrist SE(3) + 12 joint angles free → q_t = (6 wrist + 12 fingers) = paper's
R^18. Vector-distance formulation (DexPilot) to handle size mismatch. Torch/Adam.
Outputs `retarget/%05d.npy` (18,) + QA overlays.

## Stage 7 — gravity align (wild only) — TODO

GeoCalib (`geocalib` pkg) on frame 0 → gravity dir in cam frame → rotation applied to all
trajectories/meshes. Objects: bulb, hat, jengabox.

## Stage 8 — grasp window + ContactOpt — TODO

Grasp window [t1,t2]: hand within 5 cm of object (t1) to stable grasp (t2, object starts moving).
ContactOpt (facebookresearch/ContactOpt, own venv if legacy deps fight): DeepContact
(checkpoint bundled in repo) predicts target contact maps; DiffContact optimizes MANO pose.
Legacy risk: pytorch3d/torch_geometric on torch 2.7 — fallback is porting DiffContact+DeepContact
inference (small, bounded). MANO_RIGHT.pkl reused from HaMeR demo data.
Then re-retarget optimized human poses → q_opt_t. Outputs `contactopt/q_opt.json`.

## Stage 9 — D(R,O) training — TODO

In main `.venv` (torch 2.7; drograsp has no CUDA ext, transformer-based). Add `inspire` robot:
- URDF+meshes → `third_party/drograsp/data/data_urdf/robot/inspire*` (+ floating 6-DoF root
  to match their q=(6+DOF) convention; check their URDF base when data.zip lands)
- `urdf_assets_meta.json` + `removed_links.json` entries
- `data_utils/generate_pc.py --robot_name inspire` → `data/PointCloud/robot/inspire.pt`
Build `cmap_dataset.pt` from our q_opt grasps + object meshes
(`data/data_urdf/object/videomanip/<obj>/<obj>.stl`), plus train/validate split json.
Pretrain (configuration-invariant) on inspire, then train on our grasps per paper's setup.
Inference: predicted distance matrix → multilateration → q_grasp (their validate.py flow,
minus IsaacGym).

## Stage 10 — IsaacLab eval — TODO

Env (main `.venv`, IsaacLab 2.3.2): floating Inspire hand + object (mesh from stage 2,
physical props: mass from scale prior, friction defaults from paper's IsaacGym setup where known).
Protocol (paper IV-A): init hand at q_grasp & object at reconstructed pose; 300 steps;
forces ±x/±y/±z sequentially, magnitude 0.5× object mass; success = object displacement < 3 cm.
100 trials/object (randomize like paper where specified). Compare: paper 63.75% avg (single-video).
Also evaluate the paper's own reference grasps (from GLBs) as an upper-reference sanity check
(requires fitting joint angles to GLB hand meshes — optional).

## Cross-cutting

- Every stage writes a QA image/json; review before proceeding.
- Deviations logged here + in PROGRESS.md: GroundingDINO prompts, no detectron2,
  FoundationPose→ICP fallback, IsaacLab instead of IsaacGym (user requirement),
  Hunyuan3D-2 vs MeshyAI comparison.
