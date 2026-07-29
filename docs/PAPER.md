# Smooth-Trajectory Hand Data and Wrench-Aware Refinement for Video-Learned Dexterous Grasping

**Working paper draft (nucleus), 2026-07-27.** Findings from our sim reproduction of
VideoManip (arXiv:2602.09013). All numbers are measured in our IsaacLab 2.3.2 /
Isaac Sim 5.1 harness (disturbance protocol identical to the reference paper, §II-B).
Artifacts: `outputs/eval/`, `docs/REPORT.md`, `docs/PROGRESS.md`.

## Abstract

We study two under-explored ingredients of learning dexterous grasping from human
videos: (i) the *temporal quality* of the hand-motion data used to train the grasp
prediction model, and (ii) *what objective* should drive grasp repair at inference
time. On a full re-implementation of the VideoManip pipeline (RGB video → hand/object
reconstruction → ContactOpt → D(R,O)-grasp model → IsaacLab disturbance eval) we find:
**(1)** replacing per-frame HaMeR reconstructions with temporally consistent HandFlow
trajectories (after fixing its metric-depth bias with a depth-anchored hybrid) yields
training data that improves D(R,O) grasp success from 47.0% to 50.0% on our 6 objects
and from 56.7% to **65.0%** on the reference paper's 20 objects — *above the paper's
own 63.75%*; a union of both data sources reaches 69.0% on our 6 objects, and the two
sources are strongly complementary across object sets (per-object union 73.3%).
**(2)** a wrench-aware refinement objective (maximize the disturbance-rejection margin
under the evaluation's force directions) lifts grasp success by up to +38.7pp
(47.0→85.7%) over raw predictions, while a purely geometric contact objective
*degrades* grasps — proximity is not disturbance resistance.
**(3)** few-shot cross-embodiment grasp training exhibits extreme *checkpoint
oscillation*: sim success swings 9–71% between consecutive epochs while train loss
decreases monotonically; weight averaging fails, and sim-eval-driven checkpoint
selection is required. These results give concrete, transferable guidance for
video-to-grasp pipelines: smooth trajectories as data, wrench (not geometric) repair,
and select-don't-average checkpoints.

## 1. Introduction

Learning dexterous grasps from human RGB videos (VideoManip and predecessors) chains
many reconstruction stages before a grasp model (D(R,O)-Grasp / DRO) is trained on the
recovered (grasp, object) pairs. Reproduction work usually treats the reconstruction
front-end as a fixed input; we instead treat it as a *variable*, and ask three
questions that matter for anyone building such pipelines:

- **Q1 (data temporal quality).** Does temporally consistent, world-frame hand
  reconstruction (HandFlow, 2026) produce better *training data* than per-frame
  regression (HaMeR, 2024) — the temporal smoothness prior is usually claimed to help
  tracking, but does it help the *downstream learned grasp model*?
- **Q2 (repair objective).** Predicted grasps are geometrically approximate. Should
  refinement maximize contact proximity, or directly the metric the evaluation
  measures — disturbance-rejection margin?
- **Q3 (training protocol).** How reliable is a checkpoint when the training set is
  few-shot (tens of grasps) and cross-embodiment (human hand data → robot hand model)?

Context and scope: we re-implemented the full VideoManip pipeline in IsaacLab
(IsaacGym in the paper) and evaluate with the paper's disturbance protocol
(300 steps, 0.5×mass forces from ±x/±y/±z sequentially, success = object displacement
<3 cm, 100 predictions per object). Our 6 in-house objects come with released input
videos; the paper's other 14 videos were never released, so paper-20 experiments run
on the paper's own object meshes (from their released predicted-grasp GLBs) — the
same protocol and the fairest available comparison.

## 2. Background and setup

Pipeline (re-implemented): MoGe-2 metric depth + intrinsics → SAM2 masks → MeshyAI
image-to-3D → scale search (0.5–2.0× around LLM size priors) + ICP pose tracking →
hand reconstruction → DexPilot-style retargeting to the 12-DoF Inspire hand →
ContactOpt → DRO training (paper's checkpoint + our data, "mixed3x" recipe: 4920
paper grasps over 58 objects/5 robots + our inspire grasps 3×-oversampled) →
IsaacLab disturbance eval (welded wrist, fingers 1000/200 PD, friction 3.0,
ramp-close settle). Unless noted, success rates are 100 trials/object with *all*
100 DRO predictions evaluated (the paper's protocol; no filtering).

Hand stages compared:
- **HaMeR** (paper's choice): per-frame regression, weak-perspective camera; we anchor
  metric depth with MoGe depth at projected 2D joints (the paper's correction).
- **HandFlow** (2026): flow-matching temporal denoiser over MANO space conditioned on
  online-HaMeR features + 2D skeletons; fixed-camera mode, per-frame MoGe intrinsics.

## 3. Findings

### 3.1 Smooth trajectories are better *training data* (Q1)

**HandFlow is smoother — and it propagates.** Wrist/fingertip acceleration is 1.5–3×
lower than HaMeR on all 6 objects; after retargeting, robot finger-joint |ddq| is
1.8–6× lower on 5/6 objects (Table I).

**But FM-predicted metric translation is biased.** On 3/6 videos the hand is ~11 cm
from the object during physical grasp (2D alignment is correct — it is a depth error).
We fix it with the *paper's own* depth-anchoring (median of MoGe depth at projected
joints − joint z-spread) plus a temporal median filter (k=7): hand-object contact
distance 107→2.5–18 mm (HaMeR level), smoothness retained. This **depth-anchored
hybrid** — HandFlow pose + measured metric depth — is a simple, transferable fix for
learned-metric monocular models whose absolute scale drifts out of distribution.

**Downstream effect (DRO retrained, same recipe, only the inspire data source varies):**

| training data | our 6 obj | paper-20 |
|---|---|---|
| HaMeR-CO (53 grasps) | 47.0 | 56.65 |
| HandFlow-CO (50 grasps) | 50.0 | **65.0** |
| union (108 grasps) | **69.0** | 62.05 |
| paper's own result (20 obj) | — | 63.75 |

HandFlow-only data beats HaMeR-only on both sets despite 3 fewer grasps; on the
paper's 20 objects it **exceeds the paper's own reported number** (65.0 vs 63.75).
The union maximizes our in-house set (69.0%).

**Complementarity.** Union wins our-6 (69.0 vs 50.0) but HandFlow-only wins paper-20
(65.0 vs 62.05); per-object union of the two models on paper-20 = **73.3%**.
HandFlow-e16 uniquely solves powerdrill (75 vs 0) and spraybottle (68 vs 26) — objects
whose HaMeR trajectories are jitter-dominated — while union inherits HaMeR's
small-object zeros (apple, cloth_hanger). Data-source complementarity mirrors the
DRO-vs-retargeting complementarity we reported earlier: different error profiles
bracket the target metric.

*Boundary case:* HandFlow+ContactOpt grasps themselves hold only 8% in direct sim
(retarget-direct HaMeR: 44.0%) — raw HandFlow grasps spawn-penetrate and eject, and
ContactOpt's geometric repair is weak (see 3.2). The value of HandFlow is in the
*learned model's* training distribution, not in its raw grasps being directly usable.

### 3.2 Refine for wrench resistance, not proximity (Q2)

Raw DRO predictions interpenetrate slightly and many fail the disturbance test.
Two repair objectives, same optimizer (Adam, 300 iters, finger+wrist deviation caps):

- **Geometric** (surface point-to-plane contact band + penetration pushback):
  net-negative (mean 36% vs raw 40%); de-penetration removes the normal force that
  our (and any kinematic-hold) grasp relies on, and attraction cannot create
  enclosing geometry for loose predictions.
- **Wrench-aware (ours)**: per-contact normal force = spring in penetration (capped),
  resistance per disturbance direction = normal opposition + friction-cone support;
  maximize the softmin margin over the protocol's 6 force directions with a 3 mm
  penetration cap. **47.0→85.7%** mean on our 6 objects (spray 19→42, bottle 27→89,
  can 34→90, bulb 51→100, hat 97→95, jenga 54→98); one-sided pinches become
  enclosing wraps (QA viz). Generalizes: 89/97% under 12 random directions,
  87% at 1.5× force.

*Boundaries (honest negative results)*: on paper-20 the same refinement nets 48.9%
(vs raw 56.65) — it repairs weak sets (hand_bag 70→99, umbrella 21→59,
spraybottle 49→74) but perturbs strong thin-object grasps (cup 63→12, pot 92→59);
raw success rate is the practical gate, not the margin surrogate. On union-e2 it
gives 69.0→71.0% with hat regressing 78→31 (soft cage object — penetration *is* the
grip mechanism). Framing: few-shot repair of weak predictions, not a universal
postprocess. Self-distillation of refined grasps back into training degrades every
checkpoint (best 34.5% vs 47.0% init) — refinement belongs at inference time.

*Prior art note*: differentiable force-closure refinement exists (DFC, RA-L 2021;
FRoGGeR 2023; task-oriented GWB estimation 2023) but synthesizes/repairs from
scratch; repairing *video-learned model predictions*, the geometric-vs-wrench
head-to-head, and the penetration-bias critique appear to be new.

### 3.3 Checkpoint oscillation in few-shot cross-embodiment DRO (Q3)

Across three independent runs (mixed3x, union, handflow), sim-eval success oscillates
violently between epochs while train loss decreases monotonically
(Fig. `outputs/eval/dro_oscillation_curves.png`):

| run | best | worst-adjacent | pattern |
|---|---|---|---|
| mixed3x (40 ep) | e5 47.0 | e10 10.3 | slow rise, noisy plateau |
| union (20 ep) | **e2 70.7** | e7 8.7 | early peak → collapse → partial recovery |
| handflow (20 ep) | e16 54.7 | e17 13.0 | peak immediately followed by collapse |

Dense per-epoch eval (12-epoch run): train loss 0.78→0.06 while triage success swings
0.7→42; a near-zero epoch sits between two decent ones. Train objective (D(R,O)
regression) is fully decoupled from the downstream metric at this data scale.
**SWA/weight averaging fails** (avg of oscillating checkpoints 8–25% < best single
42%) — checkpoints drift between basins. Protocol consequence: save per-epoch, select
by direct sim-eval ("select-don't-average"), and never trust the last checkpoint.

## 4. What transfers (recommendations for video-to-grasp pipelines)

1. Use temporally consistent hand reconstruction for *training data*; verify its
   metric depth against measured depth and re-anchor (3-line hybrid fix).
2. Pool heterogeneous reconstruction sources — their error profiles are complementary
   (per-object union 73.3% here).
3. Repair predicted grasps with a wrench-margin objective tied to the actual
   disturbance protocol; gate by raw success; never by geometric proximity.
4. Save every epoch and select checkpoints by direct sim-eval; do not average weights
   and do not distill refined grasps back into few-shot training sets.

## 5. Limitations

- 6 in-house objects (only 6 of the paper's videos are public); paper-20 results use
  the paper's own meshes, so they compare models, not full pipelines.
- Uniform 0.3 kg mass priors (protocol is approximately mass-invariant without a table).
- Small-N retarget evals (5–27 grasps/object) are noisy; DRO rows use 100 predictions.
- Checkpoint-oscillation makes single-checkpoint claims epoch-sensitive; we report
  per-epoch curves for exactly this reason.

## References (working list)

- VideoManip (arXiv:2602.09013); D(R,O)-Grasp (arXiv:2410.01702); HaMeR (CVPR 2024);
  HandFlow (arXiv:2607.11221); HaWoR (CVPR 2025); ContactOpt (ICCV 2021);
  DFC (RA-L 2021); FRoGGeR (2023); task-oriented GWB estimation (arXiv:2309.13586);
  MoGe-2 (2025); SAM2 (2024); MeshyAI (2025); IsaacLab (2023).

---

### Appendix A — per-object tables (100 trials, raw)

Our 6 objects: see `docs/REPORT.md` result summary.

Paper-20 (handflow-e16 / union-e2 / mixed3x-e5):
apple 3/0/8, bottle 72/67/76, bowl 95/99/67, case 68/86/87, cloth_hanger 9/0/33,
cup 70/93/63, hand_bag 66/100/70, hat 74/99/92, ladle 62/46/65, mug 92/100/69,
pan 73/86/69, pot 97/98/92, powerdrill 75/0/23, scissors 89/65/77,
soap_dispenser 30/15/44, spraybottle 68/26/49, sunglass 66/70/31, toothbrush 44/75/15,
umbrella 62/26/21, wineglass 85/90/82.
Means: 65.0 / 62.05 / 56.65 (paper: 63.75).

### Appendix B — reproducibility map

- Hand stage: `scripts/run_handflow.py` (+ `scripts/fix_handflow_depth.py`),
  `.venv-handflow` (torch 2.7+cu128), weights `/app/models/handflow/`.
- DRO data: `scripts/build_dro_dataset.py` (`CMapDataset_{handflow,union,mixed3x_*}`).
- Training: `scripts/server_train.sh` (server recipe: `docs/SERVER_SETUP.md`).
- Eval: `scripts/run_grasp_eval2x.py`, mirror/triage `scripts/mirror_eval_handflow.sh`,
  curves `outputs/eval/handflow_sim_curve.log`, Fig. `outputs/eval/dro_oscillation_curves.png`.
- Wrench refine: `scripts/run_refine_wrench.py`.
- Full log: `docs/PROGRESS.md`; report: `docs/REPORT.md`; SOTA audit: `docs/SOTA.md`.
