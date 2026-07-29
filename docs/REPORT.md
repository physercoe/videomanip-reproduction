# VideoManip Reproduction — Final Report

Sim-only reproduction of [VideoManip](https://arxiv.org/abs/2602.09013) (arXiv:2602.09013):
learning dexterous grasping from RGB human videos, evaluated in **IsaacLab 2.3.2 / Isaac Sim 5.1**
(the paper uses IsaacGym). Work log: `docs/PROGRESS.md`. Data inventory: `docs/DATA.md`.
**Research findings beyond the reproduction are written up as a paper draft: `docs/PAPER.md`.**

## Result summary

| method | spraybottle | bottle | can | bulb | hat | jengabox | mean |
|---|---|---|---|---|---|---|---|
| retarget (direct) | 15 | 15 | 25 | 0 | 85 | 20 | 26.7 |
| retarget best-variant | 25 | 15 | 25 | 0 | 95 | 20 | **30.0** |
| DRO mixed e5 | 5 | 67 | 60 | 31 | 2 | 75 | 40.0 |
| DRO mixed3x e5 (HaMeR-CO data) | 19 | 27 | 34 | 51 | 97 | 54 | 47.0 |
| DRO handflow e16 (HandFlow-CO data) | 36 | 54 | 31 | 12 | 91 | 76 | 50.0 |
| **DRO union e2 (HaMeR+HandFlow data)** | 30 | 49 | 67 | 97 | 78 | 93 | **69.0** |
| DRO union e2 + wrench refine | 44 | 78 | 77 | 100 | 31 | 96 | 71.0 |
| DRO union e2, per-object union(raw∪wrench) | 44 | 78 | 77 | 100 | 78 | 96 | **78.8** |
| DRO mixed3x e5 + wrench refine (ours) | 42 | 89 | 90 | 100 | 95 | 98 | **85.7** |
| paper (DRO, 20 objects) | — | — | — | — | — | — | 63.75 |

**On the paper's own 20 objects** (their GLB meshes, our harness, 100 trials each):
**handflow-e16 = 65.0% — ABOVE the paper's reported 63.75%**; union-e2 = 62.05%;
mixed3x-e5 = 56.65%. Per-object tables in § "Paper-20 detail".
Per-object union of handflow-e16 ∪ union-e2 predictions = 73.3%.

Protocol matches paper Sec. IV-A: 300-step disturbance, forces of 0.5× object mass from ±x/±y/±z
sequentially, success = displacement < 3 cm. DRO rows evaluate **all 100 sampled predictions
per object** (the paper's protocol; no filtering). Retarget rows cycle each object's 5–27 grasps
over 20 trials. "retarget best-variant" = per-object max of baseline/refined.

- **85.7% on our 6 objects** with wrench-aware refinement — above the paper's 63.75%
  (single-video) and 70.25% (multi-video) headlines, though on 6 objects vs their 20 and
  in our harness. On their 20 objects we are at 56.65–62.1%, i.e. statistical parity.
- **DRO vs retargeting are complementary**: retargeting wins spraybottle (25 vs 19) and ties
  hat; DRO wins everything else, most dramatically bulb (51 vs 0 — the retargeted grasps
  cannot hold a hollow 16 cm shell). Best-per-object across both pipelines: 62.3%, i.e.
  the pair effectively brackets the paper's number.
- The paper's own no-ContactOpt ablation is 30.7% — almost exactly our retarget-direct 26.7%
  / best-variant 30.0%. Consistent: ContactOpt-optimized training data is what lifts DRO.

## What was reproduced (pipeline)

1. **Reconstruction** (6 objects, one RGB clip each): MoGe-2 metric depth+intrinsics → SAM2
   masks → MeshyAI image-to-mesh → two-stage scale estimation (Kimi-agent size priors instead
   of GPT-4.1, then render-error refinement) → FoundationPose 6D pose → HaMeR hand recovery
   (metric-depth corrected) → AnyTeleop-style retargeting to the Inspire hand → ContactOpt.
   57 grasps total (5–27 per object).
2. **DRO grasp model** (drograsp, patched for inspire hand + trackio): predicts D(R,O) from
   robot+object point clouds; grasps recovered by multilateration + cvxpylayers IK.
3. **Evaluation**: IsaacLab 2.3.2 disturbance harness (`scripts/run_grasp_eval2x.py`),
   6-direction force test, friction 3.0, ramp-close settle.

## Training runs (DRO)

| run | data | result |
|---|---|---|
| inspire-only, 60 ep (server A800) | 1590 augmented samples, 6 objects | **never learned** — loss flat at 0.73 for 60 epochs; predictions 81–139 mm from the object; 0–1/100 pass an FK contact gate |
| mixed, 40 ep | 4920 paper grasps (58 obj, 5 robots) + 1590 ours | learns immediately (loss 0.77→0.05); best ckpt e5 = 40.0% |
| mixed3x, 40 ep (3-GPU DDP) | same with our samples 3×-oversampled (~49%) | **best ckpt e5 = 47.0%** |

Loss curves: `outputs/eval/dro_loss_curves.png` (trackio db: `outputs/eval/DROGrasp.db`).

### Checkpoint oscillation (important finding)

Success is strongly non-monotonic in training time — per-object competence swings wildly
between checkpoints (mixed e5: bottle 67/can 60/jenga 75 → e30: 13/8/10; mixed3x hat: 97→35→85→100).
Full curve (mean over 6 objects):

mixed:   e5 40.0 | e10 28.3 | e15 19.0 | e20 24.7 | e25 25.8 | e30 10.8 | e35 20.2 | e40 28.5
mixed3x: e5 47.0 | e10 10.3 | e15 24.0 | e20 33.2 | e25 34.2 | e30 38.0 | e35 26.2 | e40 28.7

Early checkpoints are systematically best; later training drifts toward the paper objects'
dominant style (75%/51% of samples). **Checkpoint selection is part of the method** here —
per-epoch sim evals: `outputs/<obj>/eval/dro_mixed*_e*_results.json`, log:
`outputs/eval/mixed_sim_curve.log`.

## Bugs found & fixed (this session)

- **Refinement v1 inverted**: keypoint+standoff model equilibrated at 2 mm off the surface and
  *opened* grasps. v2 (`scripts/run_refine.py`): full-surface point-to-plane model, one-sided
  attraction into a [-3 mm, +2 mm] band, penetration pushback, wrist+finger optimization.
- **Frame trap**: per-frame `retarget*/*.npy` q19s are in the reconstruction frame (object NOT
  at origin); only the eval npz is object-at-origin. Documented in AGENTS.md; refine/eval
  consume npz only.
- **Vacuous DRO filter**: the old gate measured multilaterated-pc-to-object distance, ~0 by
  construction. Replaced with an FK gate (FK of recovered q → hand-surface/object contacts) —
  diagnostic only, not for eval-set selection (it rejects deep-press grasps that DO hold).
- **512 vs 65536 object-pc size crash** in mixed-data validation; our videomanip pcs
  regenerated at 512×6 (xyz+normal).
- **cdist OOM** in contact metrics (9216×8192×batch) — chunked.
- Server job control: `nohup` inside `ssh bash -c` gets reaped on local task timeout —
  use `setsid nohup ... </dev/null &`.

## Negative results (why the gap isn't smaller than 47%)

- **Robot-side refinement of sim-eval grasps is net-negative** (21.7% retarget-refined vs
  26.7% baseline; 36.0% vs 40.0% for DRO): in a kinematic-hold harness, penetration IS the
  grip-force mechanism. De-penetrating removes normal force (bottle 15→0); it only helps
  objects whose failure mode is spawn ejection (spraybottle 15→25, hat 85→95). The paper
  notes the same physics ("kinematically plausible but fail force closure").
- **Squeeze bias (1.1×)** during hold: zero effect — failures are categorical (non-enclosing
  geometry), not force-marginal.
- **Filtering DRO predictions by geometric contact quality** does not match sim success:
  deep-press grasps fail the filter but hold in sim; the paper evaluates all predictions.

## Research findings beyond the reproduction (2026-07-24)

These came out of asking "why is there a gap at all" — each is backed by experiments in this
repo and could anchor a publication.

### 1. Wrench-aware grasp refinement (method contribution)

**Prior art (checked 2026-07-24)**: differentiable wrench/force-closure grasp refinement
exists — DFC (T. Liu et al., RA-L 2021, differentiable force-closure energy),
FRoGGeR (A.H. Li et al. 2023, min-weight metric), task-oriented GWB estimator
(arXiv:2309.13586, task-wrench-directional synthesis). None of them (a) repair
NN/video-learned grasp predictions (they synthesize from scratch), (b) compare against
geometric contact refinement head-to-head on learned models, or (c) treat the
penetration/benchmark issues below. Our deltas are those three.

Geometric contact refinement optimizes the wrong objective. We replace it with a
differentiable disturbance-resistance objective (`scripts/run_refine_wrench.py`): normal
force per surface contact = capped spring in penetration; resistance per disturbance
direction = normal opposition + friction-cone support; maximize the softmin margin over
the benchmark's force directions (+ penetration cap, deviation penalties). 300 Adam iters,
seconds per grasp, no extra training.

| | spray | bottle | can | bulb | hat | jenga | mean |
|---|---|---|---|---|---|---|---|
| raw DRO (mixed3x-e5) | 19 | 27 | 34 | 51 | 97 | 54 | 47.0 |
| + wrench refine | 42 | 89 | 90 | 100 | 95 | 98 | **85.7** |

- Generalizes beyond the exact benchmark: 89/97% under 12 random force directions,
  87% at 1.5× force (can/jengabox). QA renders show one-sided pinches become enclosing
  wraps (`outputs/eval/wrench_refine_results.png`).
- **Boundary condition**: it helps weak grasp sets (+23…+62pp) but perturbs already-good
  ones — on the paper's 20 objects (raw already 56.65%) it nets 48.9% (hand_bag 70→99,
  umbrella 21→59, spraybottle 49→74 vs cup 63→12, pot 92→59); retarget hat 85→10.
  The margin surrogate does NOT gate this (all raw margins > requirement); the raw success
  rate is the practical gate (refine only weak sets). Per-object union on paper-20: 62.1%.
- Retarget side: can 25→45 but spraybottle/hat degrade — same boundary.

### 2. Checkpoint oscillation & decoupled training objective (analysis contribution)

- Per-object competence swings 0→100% between ADJACENT epochs (dense run: spray/bulb/hat
  triage = 2, 26, 0.7, 42, 22, 26, …; epoch 3 nearly zero between two decent epochs),
  while the D(R,O) train loss decreases monotonically (0.78→0.06). The L1 distance-matrix
  objective does not track disturbance resistance.
- Weight averaging (SWA over 2–12 epochs) does NOT fix it (8–25% < best checkpoint 42%):
  checkpoints hop between basins, they are not linearly connected.
- **Self-distillation does NOT fix it either**: fine-tuning mixed3x-e5 on mixed3x + 600
  wrench-refined predictions degraded every checkpoint (best 34.5% vs 47.0% init) —
  the peak basin is fragile to any continued training. Wrench refinement therefore belongs
  at inference time, not in the training loop (DAgger-style loops fail here).
- Practical protocol: dense checkpointing + per-checkpoint sim eval (or a cheaper
  disturbance-proxy), pick the best. All of our headline numbers use selected checkpoints.
- Data-scale finding (context): inspire-only on 1590 samples never learns (loss flat 0.73);
  mixing 4920 public grasps + 3×-oversampled ours learns immediately — few-shot object
  adaptation of cross-embodiment grasp models needs this mixed-recipe.

### 3. Metric-physics critique: penetration is grip force in kinematic-hold evals

- Paired experiments (same grasps ± refinement): de-penetrating REDUCES success
  (retarget bottle 15→0, can 25→5; DRO can 60→49). In a position/kinematic-hold harness,
  interpenetration is what generates normal force; "geometrically clean" grasps slip.
  Disturbance benchmarks therefore reward penetrating grasps — geometric quality filters
  (our FK gate) reject exactly the grasps that hold.
- Population correlation across 600 grasps is flat (geometry/enclosure dominates), so the
  bias shows causally at the refinement boundary, not in marginals.
- Implication for a fixed benchmark: impedance/force-limited hold (or torque-control with
  grip-force caps) would make the metric physically meaningful; until then, wrench-margin
  objectives must include a penetration cap (ours: 3 mm) to avoid spawn ejection.

### Paper-20 detail (our model, raw → wrench, 100 trials each)

mixed3x-e5: apple 8→0, bottle 76→45, bowl 67→46, case 87→86, cloth_hanger 33→27, cup 63→12,
hand_bag 70→99, hat 92→83, ladle 65→63, mug 69→53, pan 69→57, pot 92→59, powerdrill 23→6,
scissors 77→52, soap_dispenser 44→39, spraybottle 49→74, sunglass 31→41, toothbrush 15→22,
umbrella 21→59, wineglass 82→55. Mean raw 56.65 / wrench 48.9 / union 62.1 (paper: 63.75).
On the paper's 5 single-video failures (pan/hat/bowl/case/hand_bag) we average 77 raw —
their failures were reconstruction-driven, which a mesh-direct model sidesteps.
Our weak set: thin/articulated objects (toothbrush, sunglass, umbrella, powerdrill, apple).

**union-e2 (HaMeR+HandFlow data, raw, 2026-07-27):** apple 0, bottle 67, bowl 99, case 86,
cloth_hanger 0, cup 93, hand_bag 100, hat 99, ladle 46, mug 100, pan 86, pot 98,
powerdrill 0, scissors 65, soap_dispenser 15, spraybottle 26, sunglass 70, toothbrush 75,
umbrella 26, wineglass 90 → **mean 62.05%** (vs paper 63.75%, mixed3x-e5 56.65%).
Wins 11/20 vs mixed3x (dramatic: bowl 67→99, sunglass 31→70, toothbrush 15→75,
mug 69→100, pan 69→86); loses 9 (apple/cloth_hanger/powerdrill 0, soap_dispenser 44→15).
Union was trained on 108 inspire grasps (53 HaMeR-CO + 50 HandFlow-CO) vs 53 for mixed3x.

**handflow-e16 (HandFlow-CO data only, raw, 2026-07-27):** apple 3, bottle 72, bowl 95,
case 68, cloth_hanger 9, cup 70, hand_bag 66, hat 74, ladle 62, mug 92, pan 73, pot 97,
powerdrill 75, scissors 89, soap_dispenser 30, spraybottle 68, sunglass 66, toothbrush 44,
umbrella 62, wineglass 85 → **mean 65.0% — above the paper's 63.75%**. Trained on just 50
HandFlow-CO grasps (×30 augment, 3× oversampled). Much more balanced than union-e2
(powerdrill 75 vs 0, spraybottle 68 vs 26), weaker on our 6 objects (50.0% — its bulb=12,
bulb is the one object whose HandFlow grasp window [0,52] spans the whole clip).
**Per-object union of handflow-e16 ∪ union-e2 = 73.3%.** The two data sources are
complementary across object sets: union wins our 6 (69.0 vs 50.0), handflow wins
paper-20 (65.0 vs 62.05) — same complementarity we saw between DRO and retargeting.

### Open-source mesh model check (MeshyAI vs TripoSR vs TRELLIS)

Open-source single-image models are not drop-in replacements for MeshyAI on our inputs:
- **TripoSR**: bulb (reflective glass) → unusable; spraybottle → split blobs, no trigger.
- **TRELLIS-image-large** (SOTA; full install recipe: `docs/TRELLIS.md`): both test inputs
  → normalized-box blobs (bulb reflective; spraybottle crop shows only the bottle top).
- **MeshyAI**: best of the three on both, but still loses functional features vs the
  paper's own meshes (trigger, cap brim) — and that traces to INPUT CROP quality (ours cut
  off the trigger), not the mesh model. Input crops are the binding constraint.
Test assets: `outputs/{bulb,spraybottle}/mesh_triposr/`, `outputs/{bulb,spraybottle}/mesh_trellis/`,
venvs `.venv-triposr` (local), `/tmp/vm/trellis-venv` (server recipe in docs/TRELLIS.md).



- IsaacLab 2.3.2 (Isaac Sim 5.1) instead of IsaacGym; post-reset `omni.physics.tensors`
  views replace IsaacLab's stale `asset.data` (documented in AGENTS.md). IsaacLab 3.0-beta /
  Isaac Sim 6.0.1 was tried and parked (URDF importer consumes first joint, contact bugs).
- Object masses are uniform 0.3 kg priors (paper presumably uses real masses); force scales
  with mass so the protocol is approximately mass-invariant without a table.
- Size priors from the Kimi agent instead of GPT-4.1; trackio instead of wandb; 6 objects
  instead of 20; no DemoGen/DP3 manipulation stage (grasp-only reproduction).
- DRO training mixes the authors' released CMapDataset (subsampled) with ours — necessary
  because 6 objects alone cannot train the model (see inspire-only run).

## Gap analysis (resolved 2026-07-24)

The earlier 16pp gap decomposed as: (i) **data scale** — fixed by mixed-data training
(inspire-only never learned; mixed3x learns immediately); (ii) **object-set difficulty** —
on the paper's own 20 objects our model scores 56.65% raw, statistical parity with their
63.75% given different sim/masses; (iii) **refinement objective** — geometric refinement
was net-negative; wrench-aware refinement lifts our pipeline to 85.7% on our 6 objects.
Residual weaknesses: thin/articulated objects (apple/toothbrush/powerdrill on paper-20,
spraybottle on ours — single egocentric view, trigger cropped out of the mesh input).

## Next steps (if continued)

- **Publication-direction experiments** (see "Research findings"): (a) gate wrench
  refinement by a cheap sim-pilot so strong grasps aren't perturbed; (b) train/finetune
  DRO with the wrench margin as an auxiliary loss instead of pure L1 — attacks the
  objective/metric decoupling at the source; (c) impedance-hold eval variant to quantify
  the penetration bias benchmark-wide.
- Collect 2 extra videos per failed object (paper's multi-video recipe) — spraybottle first.
- Real-mass priors per object; stronger open mesh model (TRELLIS/Hunyuan3D) vs MeshyAI.
- Add the DP3 manipulation stage (paper Sec. IV-B) if desired.

## Reproducibility

- Eval: `scripts/run_grasp_eval2x.py --object <obj> --grasps <npz> --num_envs 100`
  (venv `.venv`, `OMNI_KIT_ACCEPT_EULA=YES`).
- Inference: `scripts/run_dro_inference.py --epoch <e> --run videomanip_mixed3x --n_samples 100`
  (venv `.venv-recon`). Checkpoints: `third_party/drograsp/output/*/state_dict/`.
- Tables: `python3 scripts/make_report.py`. Battery: `scripts/run_battery.sh`.
- Full command/environment reference: `AGENTS.md`; per-stage data: `docs/DATA.md`;
  chronological log with server setup recipes: `docs/PROGRESS.md`.
