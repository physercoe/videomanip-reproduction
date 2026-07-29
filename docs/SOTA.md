# Component SOTA audit (2026-07-24)

Question (user): does the paper use SOTA models in each aspect? Answer: mostly 2024-vintage.
Per-stage audit of the VideoManip pipeline with what we verified hands-on.

| stage | paper uses | year | SOTA today | verdict |
|---|---|---|---|---|
| metric depth | MoGe-2 | 2025 | MoGe-2 / VGGT / DepthAnythingV3 | current enough |
| masks | SAM2 | 2024 | **SAM3** (Carion et al. 2025, concept prompts, 2× PCS) | upgrade exists; masks are NOT our bottleneck |
| image-to-mesh | MeshyAI (commercial) | 2025 | TRELLIS / TripoSG / Hunyuan3D-2.x / PartCrafter | **tested TRELLIS + TripoSR — both fail on our inputs** (bulb reflective, partial crops); MeshyAI stays; input crop quality is the real lever |
| 6D pose | FoundationPose | 2024 | SAM-6D, GigaPose | adequate; pose stage not the bottleneck |
| hand mesh | HaMeR | 2024 | **HandFlow** (2026-07, video FM, MIT), HaWoR (CVPR 2025, CC-BY-NC-ND) | **tested HandFlow**: smoothness 1.5-3× better, metric-trans bias needs MoGe anchoring; grasp-hold eval not improved (see below) |
| gravity align | GeoCalib | 2024 | GeoCalib | fine |
| **size priors** | GPT-4.1 text-only | 2025-04 | modern VLMs/LLMs | **tested Qwen2.5-VL-7B (ModelScope) — see below** |
| contact opt | ContactOpt | 2021 | our wrench refinement (robot-side) | superseded for the robot side |
| grasp model | DRO-Grasp | 2025 | DRO-Grasp | core method under reproduction |
| policy | DP3 + DemoGen | 2023/2025 | — | not reproduced (grasp-only scope) |

## VLM size priors (Qwen2.5-VL-7B vs GPT-4.1/kimi priors)

Ran `scripts/vlm_size_priors.py` on the 6 object crops (server, model via ModelScope).

| object | VLM identification | VLM diag_m | current prior | real |
|---|---|---|---|---|
| spraybottle | Clorox All Purpose Cleaner ✓ | 0.29 | 0.32 | ~0.32 |
| bottle | Brisk Iced Tea ✓ | 0.21 | 0.22 | ~0.22 |
| can | Pringles Original ✓ | 0.17 | 0.27 | ~0.26 |
| bulb | LED Edison filament ✓ (bulb only) | 0.105 | 0.22 | 0.12 bulb / 0.20+socket |
| hat | Baseball Cap ✓ | 0.24 | 0.36 | ~0.35 |
| jengabox | Jenga ✓ | 0.224 | 0.29 | ~0.29 |

- **Identification: brand-perfect on 6/6** (GPT-4.1 text-only never saw the images).
- **Metric accuracy: ~15–25% small on 4/6** (can/hat/jengabox) — known VLM weakness in
  absolute metric estimation; kimi-agent priors are closer to reality.
- Recommendation: VLM-for-identification + LLM-for-size (or keep agent priors); the
  scale-refinement stage (0.5–2.0× render-error search) is the real corrector either way.
  GPT-4.1 replacement is a non-issue for the final metric — any modern LLM suffices here.

## Model weight mirrors (user point)

- **ModelScope works well**: Qwen2.5-VL-7B (16GB) downloaded cleanly with proxies unset.
  Also mirrors TRELLIS (AI-ModelScope/*) and most open weights. Good fallback when
  huggingface.co/hf-mirror are unavailable.
- hf-mirror.com remains primary (TRELLIS 4GB, MoGe/SAM2/GroundingDINO all fetched fine).

## Hand-stage upgrade path (EXECUTED 2026-07-24 — see PROGRESS.md)

**HandFlow** (arXiv 2607.11221, 2026-07, [repo](https://github.com/mxxu00/HandFlow)) —
chosen over HaWoR: >30% better world-space pose than the strongest baseline (incl.
HaWoR), 12× faster, MIT license (HaWoR is CC-BY-NC-ND), and already on
torch 2.7+cu128 (no torch-1.13 migration needed). Installed in `.venv-handflow`
(py3.11, torch 2.7.0+cu128 = .venv-recon stack); `--fix_camera` mode fits our
static-camera videos (no SLAM extension needed).

Results on our 6 objects: detection ~100% (WiLoR YOLO, right hand everywhere),
wrist/fingertip acceleration 1.5-3× LOWER than HaMeR per-frame, retargeted robot
finger trajectories 1.8-6× smoother on 5/6 objects (the property that matters for
DRO training data). BUT FM-predicted metric translation is biased up to ~16 cm on
3/6 objects → fixed with the paper's own MoGe depth anchoring + temporal median
filter (contact med 2.2-18 mm ≈ HaMeR level). Sim grasp-hold eval: HandFlow variants
do NOT beat HaMeR ones (direct 0.6% vs 44.0%; +ContactOpt 8% vs 26.7%) — holding is
dominated by contact geometry, not temporal jitter; raw HandFlow grasps spawn-
penetrate and eject. Hand stage is not the bottleneck. Untested: DRO retrained on
HandFlow-derived grasps (the actual paper downstream — its smoother trajectories
are the reason to try it).

HaWoR (CVPR 2025) remains the fallback if HandFlow's V1 (inference-only) release
proves limiting; HandFlow training code is not yet released.

**Downstream DRO test (2026-07-27) — POSITIVE**: DRO retrained on HandFlow-CO grasps
(50, ×30 aug, 3× oversampled, mixed3x recipe) scores 50.0% on our 6 and **65.0% on the
paper's 20 objects — above the paper's own 63.75%** (HaMeR-CO data: 47.0 / 56.65;
union of both sources: 69.0 / 62.05; per-object model-union on paper-20: 73.3).
The two data sources are complementary across object sets; checkpoint oscillation
applies (e16/e2 selected by sim-eval). The smooth-trajectory hand-stage upgrade is
now the default data source recommendation for this pipeline.
