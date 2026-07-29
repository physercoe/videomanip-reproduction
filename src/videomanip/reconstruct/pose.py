"""Stage 3+4: object scale estimation and 6D pose tracking via depth registration.

FoundationPose fallback (documented deviation): registers the scaled mesh to the
masked MoGe metric depth point cloud. Frame-0 init via centroid+PCA alignment,
per-frame point-to-point ICP refinement, coarse-to-fine scale search around the
size prior (Kimi-agent prior replaces GPT-4.1; see data/size_priors.json).
"""
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray,
                stride: int = 2) -> np.ndarray:
    """Back-project masked depth pixels to a camera-frame point cloud."""
    ys, xs = np.where(mask > 0)
    ys, xs = ys[::stride], xs[::stride]
    z = depth[ys, xs]
    valid = z > 0.02
    ys, xs, z = ys[valid], xs[valid], z[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    return np.stack([X, Y, z], axis=1)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean Chamfer distance between point sets."""
    ta, tb = cKDTree(a), cKDTree(b)
    return float(tb.query(a)[0].mean() + ta.query(b)[0].mean()) / 2.0


def icp(mesh_pts: np.ndarray, cloud: np.ndarray, T_init: np.ndarray,
        iters: int = 30, max_corr: float = 0.02) -> tuple[np.ndarray, float]:
    """Point-to-point ICP aligning mesh_pts -> cloud. Returns (T, mean_corr_dist)."""
    T = T_init.copy()
    tree = cKDTree(cloud)
    for _ in range(iters):
        src = (T[:3, :3] @ mesh_pts.T).T + T[:3, 3]
        dist, idx = tree.query(src)
        keep = dist < max_corr
        if keep.sum() < 10:
            break
        P, Q = src[keep], cloud[idx[keep]]
        cP, cQ = P.mean(0), Q.mean(0)
        H = (P - cP).T @ (Q - cQ)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cQ - R @ cP
        dT = np.eye(4)
        dT[:3, :3], dT[:3, 3] = R, t
        T = dT @ T
        if np.abs(t).max() < 1e-5:
            break
    src = (T[:3, :3] @ mesh_pts.T).T + T[:3, 3]
    dist, _ = tree.query(src)
    return T, float(dist[dist < max_corr].mean()) if (dist < max_corr).any() else max_corr


def pca_init(mesh_pts: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Coarse alignment: match centroids and PCA frames (best of 4 sign combos)."""
    cm, cc = mesh_pts.mean(0), cloud.mean(0)
    em = np.linalg.eigh(np.cov((mesh_pts - cm).T))[1]
    ec = np.linalg.eigh(np.cov((cloud - cc).T))[1]
    best, best_cost = None, np.inf
    for sx in (1, -1):
        for sy in (1, -1):
            sz = sx * sy
            R = ec @ np.diag([sx, sy, sz]) @ em.T
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = cc - R @ cm
            T, cost = icp(mesh_pts, cloud, T, iters=10)
            if cost < best_cost:
                best, best_cost = T, cost
    return best


def estimate_scale_and_pose(mesh: trimesh.Trimesh, depth: np.ndarray, mask: np.ndarray,
                            K: np.ndarray, size_prior: float,
                            scales: np.ndarray | None = None,
                            n_mesh_pts: int = 4000) -> dict:
    """Coarse-to-fine scale search + registration on one reference frame."""
    if scales is None:
        scales = np.linspace(0.5, 2.0, 16)
    cloud = backproject(depth, mask, K)
    mesh_pts0, _ = trimesh.sample.sample_surface(mesh, n_mesh_pts)
    # normalize mesh to unit size along its PCA-major axis for scale reasoning
    extent = float(np.linalg.norm(mesh_pts0.max(0) - mesh_pts0.min(0)))
    results = []
    for s in scales:
        # target: mesh bbox diagonal == size_prior * s
        factor = size_prior * s / extent
        pts = mesh_pts0 * factor
        T = pca_init(pts, cloud)
        T, cost = icp(pts, cloud, T, iters=20)
        results.append({"scale_mult": float(s), "factor": float(factor),
                        "cost": cost, "T": T.tolist()})
    best = min(results, key=lambda r: r["cost"])
    return {"best": best, "all": results, "cloud_n": len(cloud), "extent0": extent}


def track(mesh: trimesh.Trimesh, factor: float, T0: np.ndarray,
          frames_data: list[dict], n_mesh_pts: int = 4000) -> list[dict]:
    """Per-frame ICP tracking. frames_data: [{depth, mask, K}...]. Returns per-frame T."""
    mesh_pts, _ = trimesh.sample.sample_surface(mesh, n_mesh_pts)
    pts = mesh_pts * factor
    T = T0.copy()
    out = []
    median_cost = None
    for i, fd in enumerate(frames_data):
        cloud = backproject(fd["depth"], fd["mask"], fd["K"])
        if len(cloud) < 50:
            out.append({"frame": i, "T": T.tolist(), "cost": None, "reinit": False})
            continue
        T, cost = icp(pts, cloud, T, iters=25)
        reinit = False
        if median_cost is not None and cost > 3 * median_cost:
            T = pca_init(pts, cloud)
            T, cost = icp(pts, cloud, T, iters=25)
            reinit = True
        costs = [o["cost"] for o in out if o["cost"] is not None] + [cost]
        median_cost = float(np.median(costs))
        out.append({"frame": i, "T": T.tolist(), "cost": cost, "reinit": reinit})
    return out
