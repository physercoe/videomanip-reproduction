# Data Inventory

All data lives under `data/` (project-specific) and `/app/models`, `/app/datasets`
(shared, per user directive 2026-07-22). Sources: the paper's project site repo
<https://github.com/videomanip/videomanip.github.io> (the deployed site 404s on most videos;
the git repo has them under `static/data/` — fetch with `scripts/fetch_paper_data.sh`).

**Published artifacts (2026-07-27)**: code+docs at
<https://github.com/physercoe/videomanip-reproduction>; unique large data (trained
DRO checkpoints incl. full epoch series, derived CMapDataset_* datasets, hand records,
predicted grasps, GLB-derived meshes, all eval result jsons) at
<https://huggingface.co/datasets/physer/videomanip-reproduction>.

**Offline kit (2026-07-31)**: `/media/wb/T7/stardata/videomanip/` (T7 SSD, 198 GB) —
everything needed to reproduce on an air-gapped server (project tree, models, all venvs,
HF cache, uv wheel cache, git bundle, SHA256SUMS). Restore/run guide:
`docs/OFFLINE_SERVER.md`; rebuild kit: `scripts/make_offline_kit.sh`.

## Shared locations

| Path | Contents | Used by |
|---|---|---|
| `/app/models/drograsp/` | DRO-Grasp release data.zip + ckpt.zip contents + our built datasets (CMapDataset_{videomanip,handflow,union,mixed,mixed3x,mixed3x_handflow,mixed3x_union,selfdistill}); symlinked from third_party/drograsp/{data,ckpt} | DRO training |
| `/app/models/mano/mano_v1_2/` | MANO_LEFT/RIGHT.pkl, SMPLH, webuser (from user-downloaded mano_v1_2.zip) | HaMeR, ContactOpt (symlinked) |
| `/app/models/sam2/` | sam2.1_hiera_{tiny,small,base_plus,large}.pt | mask stage (symlinked from third_party/sam2/checkpoints) |
| `/app/models/hamer/` | hamer _DATA (hamer.ckpt, vitpose ckpts, mean params) | hand stage |
| `/app/models/handflow/` | handflow_denoiser.pt (668MB), detector.pt (WiLoR YOLO), normalization_stats.npz | HandFlow hand-stage upgrade |
| `~/.cache/huggingface` | HF weights (MoGe-2, GroundingDINO) via HF_ENDPOINT=https://hf-mirror.com | recon stages |

**Migrating to a new GPU server**: see `docs/SERVER_SETUP.md` — 3 commands
(server_bootstrap.sh / server_sync.sh / server_train.sh) + server_mirror.sh to pull
results back. Env manifests for all venvs in `envs/freeze-*.txt`.

Note: `third_party/hamer/_DATA/data/mano/*.pkl` are symlinks into /app/models/mano.
Note: pymeshfix is NOT suitable for our meshes (mangles frames/scale — rejected, see PROGRESS).

Hand-stage variant outputs (2026-07-24, see PROGRESS "Hand-stage upgrade"):
`outputs/<obj>/handflow/` (HandFlow records, same schema as hand/, depth-corrected
t_metric + t_metric_raw), `outputs/<obj>/contactopt_handflow/`,
`outputs/<obj>/retarget_handflow/`, `outputs/<obj>/retarget_contactopt_handflow/`,
eval npz/results: `retarget_direct{,_s1}_*`, `retarget_handflow{,_s1}_*`,
`retarget_contactopt_handflow_*`.

## data/videos/ — raw RGB input videos

| Dir | File | Size | Source (`static/data/...`) | Notes |
|---|---|---|---|---|
| spraybottle | rgb.mp4 | 5.2 MB | grasp/human_grasp_video.mp4 | 1280x960@30, 17.2s. **Only frames 0–297 (~9.9s) are the full-res spray-bottle clip; the rest is a 5x5 low-res montage — trim before use.** |
| bottle | rgb.mp4 | 0.8 MB | manip/in_scene/pourtea/rgb.mp4 | pour-tea task; grasp the bottle |
| can | rgb.mp4 | 1.0 MB | manip/in_scene/placebottle/rgb.mp4 | pick&place can |
| bulb | rgb.mp4 | 1.4 MB | manip/in_the_wild/bulb/rgb.mp4 | in-the-wild (needs GeoCalib gravity align) |
| hat | rgb.mp4 | 0.5 MB | manip/in_the_wild/hang/rgb.mp4 | in-the-wild |
| jengabox | rgb.mp4 | 0.5 MB | manip/in_the_wild/jenga_move/rgb.mp4 | in-the-wild |
| drawer | rgb.mp4 | 1.0 MB | manip/in_scene/closedrawer/rgb.mp4 | optional; drawer not a graspable object |
| (root) | teaser.mp4 | 8.5 MB | teaser.mp4 | rendered-results montage, NOT usable as input |

Also mirrored in each task dir on the repo: `human.mp4` (reconstruction viz),
`projection.mp4` (point-cloud reprojection), `robot.mp4` (real-robot rollout),
`<task>_1/2.mp4` (real rollouts) — not downloaded; available if needed for comparison.

## DRO training datasets (`third_party/drograsp/data/`)

| Dir | Contents |
|---|---|
| `CMapDataset_videomanip/` | our 1590 grasps (53 retargeted+ContactOpt, 30x augmented), inspire only |
| `CMapDataset_filtered/` | paper's released 24764 grasps (58 objects, 5 robots — NO inspire) |
| `CMapDataset_mixed/` | 4920 paper (uniform-per-object subsample) + 1590 ours = 6510 |
| `CMapDataset_mixed3x/` | 4920 paper + 3×1590 ours = 9690 (~49% ours; best run) |
| `CMapDataset_selfdistill/` | mixed3x + 600 wrench-refined predictions = 10290 (negative result, see PROGRESS) |
| `PointCloud/object/videomanip/` | 512×6 (xyz+normal) per object — MUST be 512 like the paper's, not 65536 |

Trained checkpoints: `third_party/drograsp/output/videomanip_{inspire_srv,mixed,mixed3x}/state_dict/`
(mirrored from the ephemeral GPU server; mixed3x epoch_5 is the final model).
Sim-eval result jsons: `outputs/<obj>/eval/*_results.json`; per-epoch curve:
`outputs/eval/mixed_sim_curve.log`; trackio db + loss plot: `outputs/eval/DROGrasp.db`,
`outputs/eval/dro_loss_curves.png`.

## data/reference/glb/ — paper's predicted grasps (20 objects)

`grasp/inspire/<object>/0.glb` — glTF binary (trimesh-generated), ~2.3 MB each, real GLBs.
Each contains the Inspire hand mesh posed in the D(R,O)-predicted grasp + the object mesh.
Objects: apple, bottle, bowl, case, cloth_hanger, cup, hand_bag, hat, ladle, mug, pan, pot,
powerdrill, scissors, soap_dispenser, spraybottle, sunglass, toothbrush, umbrella, wineglass.

Use: (a) ground-truth reference to compare our reproduction against the paper's outputs;
(b) potential source of object meshes / grasp configs for pipeline validation.

TODO: inspect GLB node structure (hand vs object meshes, units/scale, pose convention).
