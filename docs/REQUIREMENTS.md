# Pipeline requirements, input QA, compute cost, portability

Answers (2026-07-24) for: what does an input video need, how to check its quality,
what the pipeline costs (FLOPs/memory/time), and what it takes to move off NVIDIA.

## 1. Input video criteria

Hard requirements (derived from our 6-object runs; violations = known failure modes):

| Criterion | Why | Our data |
|---|---|---|
| RGB video, hand interacting with **one rigid object** | pipeline reconstructs one object + one hand | all 6 |
| **Fixed camera** (static scene) | object pose = ICP tracking in a static frame; GeoCalib gravity align assumes it. HandFlow supports moving cameras (ViPE), our object stage does not | all 6 |
| **Right hand** visible (left must be mirrored) | HandFlow FM model is right-hand trained | all 6 |
| Object fully visible & roughly static for the first ~10 frames | scale search + pose bootstrap need a clean reference frame | all 6 |
| Grasp segment where the hand encloses the object | grasp window detection = hand<5cm then object moves | all 6 |
| No heavy/reflective object surfaces | MeshyAI image-to-3d fails on reflections (bulb) & partial crops | bulb = weak case |
| ≥ 720p, ~30 fps, 2–15 s, ~50–300 frames | tested range | 58–297 frames, 1280×720/960 |
| Non-cluttered background helps SAM2 tracking | can survived being dropped into a bin, but it's the edge | can |

Nice-to-have: size prior for the object class (we use the kimi agent's priors in
`data/size_priors.json`; the 0.5–2.0× scale-refinement search corrects bad seeds anyway).

## 2. Input quality checks (automated, in pipeline order)

Every stage writes QA artifacts; check these numbers BEFORE trusting the grasps:

1. **Masks** (`outputs/<obj>/masks/`): SAM2 track alive on the last frame; GroundingDINO
   box on the right object in frame 0 (`qa.png`).
2. **Depth/intrinsics** (`depth/intrinsics.json`): per-frame fx stable (±5%); wild
   jumps = MoGe struggling (usually motion blur).
3. **Mesh** (`mesh/`): render mesh overlay on a mid frame (`probe_grasp_visual.py` /
   3D QA png) — functional features (trigger, brim) must survive; blob = bad mesh.
4. **Scale/pose** (`scale/scale.json`, `pose/`): chosen factor within 0.5–2.0;
   ICP cost < 10 mm, 0 reinitializations (our good objects: 5.5–8.6 mm).
   hat needed manual override — pure-Chamfer can pick degenerate small scales.
5. **Hand** (`hand*/qa.png`): detection rate ≈ 100% (ours: 114/114 … 91/93);
   2D joint overlay lands on the hand; wrist accel < ~25 mm/frame² (HandFlow: <15).
6. **Contact sanity**: median hand-joint→object distance inside the grasp window
   < ~10 mm (both `hand/` and `handflow/` pass on good objects).
7. **Retarget** (`retarget*/qa.json`): median DexPilot vec cost < ~1.5
   (0.3–1.7 across objects/methods); >3 = joint-convention or scale bug.

## 3. Compute cost (measured on this machine unless noted)

Perception (per object, ~150 frames), single consumer GPU (RTX 5080 16 GB):

| Stage | Model | Params | Per-frame | Peak VRAM | Wall/object |
|---|---|---|---|---|---|
| frames | ffmpeg | — | — | — | seconds |
| depth (MoGe-2 vitl) | ViT-L | ~300 M | ~0.1–0.3 s (~200–300 GF est.) | ~4 GB | ~1 min |
| masks | GDINO-T + SAM2-L | 172 M + 224 M | ~0.2 s | ~6 GB | ~1–2 min |
| mesh | MeshyAI v1 (cloud API) | — | — | — | ~5–10 min (API) |
| scale+pose | Chamfer + ICP | — | CPU | — | ~1 min |
| hand (HandFlow) | WiLoR-YOLO 13 M + HaMeR ViT-H 672 M + FM denoiser 167 M | ~0.3 s | **9.2 GB** | ~1 min |
| retarget | drograsp FK (Adam 200 it) | — | ~0.3 s | ~2 GB | ~1–2 min |
| ContactOpt | DeepContactNet (CPU pytorch3d) | 1.4 M | CPU | — | ~3 min |

Perception total ≈ 10–15 min/object, **fits in 16 GB**; ~100 TFLOPs/object (est.).
Old HaMeR stage alone: ~1.5× the HandFlow time (GDINO + per-frame HaMeR + depth correction).

DRO-Grasp (the trainable model):

| Phase | Data | Hardware | Cost |
|---|---|---|---|
| pretrain (config-invariant) | inspire point clouds | 1× A800 | 100 epochs, ~14 GB |
| train (mixed3x recipe) | 9.4k samples, 6 robots | 2× A800 | **64 GB/GPU @ batch 16** (~4 GB/sample), ~5–7 min/epoch, 20 epochs ≈ 2 h |
| inference (100 grasps/object) | — | 1 GPU | ~6 GB, ~1–2 min |

Sim eval (IsaacLab 2.3.2, disturbance protocol): ~10–60 s per object per checkpoint
(50–100 parallel envs, ~4–8 GB VRAM).

## 4. Non-NVIDIA portability (Ascend / Cambricon MLU / Hygon DCU)

**TL;DR: everything except the IsaacLab sim-eval is portable PyTorch work; IsaacLab
itself is NVIDIA-locked and needs a CPU-MuJoCo fallback or one NVIDIA box.**

| Component | Portability | Notes |
|---|---|---|
| MoGe, SAM2, GroundingDINO (HF transformers) | ✅ portable | standard transformer/conv ops; run on torch_npu / catch / DTK |
| HandFlow/HaMeR | ✅ portable | SDPA fallback already used (no flash-attn installed); HaMeR is vanilla ViT+MLP |
| WiLoR YOLO (ultralytics) | ✅ portable | pure PyTorch |
| DRO training (PL + DDP) | ✅ portable w/ vendor backend | Ascend: torch_npu + HCCL; Cambricon: CNCL; Hygon: RCCL. Batch 16→8 halves the 64 GB peak |
| DRO inference (cvxpylayers IK) | ⚠️ CPU fallback | diffcp uses C/SCS solvers — CPU fine at inference scale |
| ContactOpt (torch_geometric) | ⚠️ CPU OK / NPU needs torch_scatter ports | we already run it CPU-only |
| pytorch3d | ⚠️ CPU build exists | we use CPU-only build locally; vendor builds possible |
| MeshyAI mesh | ✅ none | cloud API, no GPU |
| MANO/manopth/smplx/chumpy | ✅ portable | pure torch/numpy |
| **Isaac Sim 5.1 / IsaacLab 2.3.2 sim eval** | ❌ **NVIDIA-only** | Omniverse Kit + PhysX + Vulkan/RTX; no Ascend/MLU/DCU path |

Sim-eval replacement options (if the target machine has no NVIDIA GPU):

1. **MuJoCo (CPU)** — practical: our eval scene is one Inspire hand (URDF) + one
   object mesh; MJCF conversion is mechanical (URDF→MJCF + convex collision meshes),
   the disturbance protocol (50 settle + 6×50 force steps, 3 cm threshold) maps 1:1,
   and 50–100 trials parallelize over CPU processes. Effort: ~1–2 days, mostly
   actuator/PD retuning (ImplicitActuatorCfg stiffness 1000/200 → MJCF position
   actuators) and contact/friction matching (scene-wide 3.0).
2. Keep one small NVIDIA box just for eval (what we do today: training on A800,
   eval on the local RTX 5080).
3. Vendor sims (e.g. Huawei's) — not evaluated; protocol is simple enough to reimplement.

Vendor-stack notes:
- **Ascend (910B, 64 GB)**: torch_npu 2.x + CANN; most transformer models run
  out-of-the-box; watch for torchvision custom ops and einops version quirks.
  64 GB fits our batch-16 training peak exactly — else batch 8.
- **Cambricon MLU**: PyTorch via catch; same op-coverage caveats.
- **Hygon DCU**: ROCm-derived DTK — PyTorch ROCm builds usually work; closest to CUDA.
- Universal checklist: vendor PyTorch fork ≥ torch 2.x, SDPA support, no
  flash-attn requirement (we already fall back), no x86-only wheels (lmdb, rtree are
  source-buildable), JAX NOT required anywhere (MJX unusable for that reason).
