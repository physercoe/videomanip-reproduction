# Server setup & migration guide

How to stand up the VideoManip training/eval stack on a fresh GPU server quickly.
Written for temp/test servers (the first was a shared 4× A800 80GB box reached over ssh;
pass `user@host` to the scripts or export `VIDEOMANIP_SERVER`) —
**assume the server can be wiped at any time; everything important lives in this repo
or under `/app/models` on the local machine and is synced TO the server.**

## What lives where (single source of truth)

| Content | Canonical local location | Server location |
|---|---|---|
| DRO-Grasp code (patched: trackio, DRO_DATASET_DIR, DRO_INIT_SD) | `third_party/drograsp/` | `/tmp/vm/drograsp/` |
| DRO datasets + robot assets (incl. our CMapDataset_* dirs) | `/app/models/drograsp/data/` (~1.1 GB) | `/tmp/vm/drograsp/data/` |
| DRO pretrain ckpts (pretrain_inspire.pth etc.) | `/app/models/drograsp/ckpt/pretrain/` | same rel. path |
| MANO | `/app/models/mano/mano_v1_2/models/` | (not needed for training) |
| HandFlow weights | `/app/models/handflow/` | (not needed for training) |
| Env manifests (uv pip freeze) | `envs/freeze-*.txt` | `envs/freeze-server-drograsp.txt` |
| Training outputs (state_dicts, trackio db) | mirrored back: `third_party/drograsp/output/`, `outputs/eval/DROGrasp.db` | `/tmp/vm/drograsp/output/`, `~/.cache/huggingface/trackio/` |

`third_party/drograsp/{data,ckpt}` are symlinks into `/app/models/drograsp` — datasets
built by `scripts/build_dro_dataset.py` land there directly.

## Migrate to a NEW server (3 commands)

```bash
# 1. env (~5-10 min): uv venv py3.11 + torch 2.7.0+cu126 + deps from envs/freeze-server-drograsp.txt
bash scripts/server_bootstrap.sh user@newserver /tmp/vm/drograsp
# 2. code + data + ckpts (~1.2 GB, few min)
bash scripts/server_sync.sh user@newserver /tmp/vm/drograsp
# 3. train (detached, survives disconnect)
bash scripts/server_train.sh <run_name> <dataset_dir> <gpus e.g. 0,1,2,3> [epochs=20] [batch/gpu=16] user@newserver
```

Mirror results back (run periodically — server is ephemeral):

```bash
bash scripts/server_mirror.sh user@newserver /tmp/vm/drograsp
```

## GPU server facts (first test box, shared tenancy)

- 4× A800 80GB PCIe (sm_80), driver 580.159.03, Ubuntu 22.04, sudo without password.
- **SHARED BOX — other tenants run jobs too.** Never kill processes whose cmdline does
  not match our exact run name (`name=videomanip_*`); check `nvidia-smi
  --query-compute-apps=pid,used_memory --format=csv` and `ps -p <pid> -o cmd` before
  killing anything. Tenant activity has wiped our venv and ckpts before — see below.
- Network: `proxy_on` in ~/.bashrc (100.64.0.1:8888), but DIRECT works for
  pypi/download.pytorch.org/astral; HF via `HF_ENDPOINT=https://hf-mirror.com` with proxies unset.
- `/tmp` does NOT survive reboot AND may be cleaned by others. Keep nothing important
  only on the server; mirror checkpoints every few minutes (server_mirror.sh / loop).
- Batch-size rule of thumb (DRO): 1 sample ≈ 4 GB GPU memory; batch 16/GPU ≈ 64 GB.
- NCCL on this box is flaky under tenant load (PCIe contention): 30-min watchdog
  timeouts hit DDP runs intermittently. GPUs 2,3 were the reliable pair on 2026-07-27;
  check occupancy before launching and prefer the pair a successful run just used.
  Retry works; long 4-GPU runs are the riskiest.

## Lessons baked into the scripts

- Launch long jobs with `setsid nohup ... &` — a plain `nohup` inside `ssh bash -c` gets
  reaped when the local ssh task times out.
- `cd <repo>` must be INSIDE the remote command of each launch; a leading
  `cd X && setsid ... &` backgrounds the whole chain and subsequent commands run from $HOME.
- NCCL 30-min timeout can kill 4-GPU DDP during slow dataset init; 2-3 GPUs proven stable.
- Kill zombie DDP ranks ONLY by PID list after confirming the cmdline matches our run
  name (`name=videomanip_*`) — the box is SHARED; other tenants' jobs must never be
  touched (user directive 2026-07-27). pkill patterns self-match the ssh wrapper.
- Torch wheel: cu126 for A800; driver ≥560 also accepts cu128. Change via `TORCH_TAG=cu128`.
- Checkpoint protocol (our finding): train success oscillates per epoch — ALWAYS
  `save_every_n_epoch=1` and select checkpoints by sim-eval, never take the last epoch.

## Local venvs (for reference / rebuilding this machine)

- `envs/freeze-isaaclab232.txt` — .venv (Isaac Sim 5.1 + IsaacLab 2.3.2, sim eval)
- `envs/freeze-recon.txt` — .venv-recon (MoGe/SAM2/HaMeR/drograsp inference/retarget)
- `envs/freeze-handflow.txt` — .venv-handflow (HandFlow stage)
- `envs/freeze-contactopt.txt` — .venv-contactopt (ContactOpt)
- `envs/freeze-lab3.txt` — .venv-lab3 (PARKED Isaac Sim 6.0.1 + IsaacLab 3.0 beta)

Note: freezes capture versions but some packages were installed from git or copied
manually (chumpy fork, manopth git, hamer `-e --no-deps`) — see AGENTS.md env section
and docs/PROGRESS.md before rebuilding recon/handflow/contactopt from scratch.
