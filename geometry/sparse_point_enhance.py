"""稀疏点云增强：benchmark 测试集点云较稀时，借助 proxy mesh 密集采样辅助 NEO。

思路：
  1. 保留原始 GT 点（前 n_gt 行），从对齐的 proxy mesh 表面再采一批辅助点；
  2. 在「GT + 辅助」合并点云上跑 NEO，得到更稳定的 LBO 特征空间；
  3. 只取前 n_gt 行特征向量，供后续 spectral 子块分裂切片复用（渲染 / VLM 仍用原始稀疏点）。

纯 numpy + trimesh，无 torch 依赖。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from geometry.pointcloud_io import load_pointcloud, sample_points_from_mesh

LOGGER = logging.getLogger("geometry.sparse_point_enhance")

# compat200 GT 约 5000 点；低于此阈值视为稀疏，触发 mesh 辅助增强。
DEFAULT_SPARSE_THRESHOLD = 8000
DEFAULT_DENSIFY_TARGET = 20000


def estimate_mean_knn_distance(points: np.ndarray, k: int = 8) -> float:
    """估计点云平均 k 近邻距离（越大越稀疏）。"""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= 1:
        return 0.0
    from scipy.spatial import cKDTree

    kk = int(min(max(k, 1), n - 1))
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=kk + 1)
    if dists.ndim == 1:
        return float(dists[1]) if len(dists) > 1 else 0.0
    return float(np.mean(dists[:, 1:]))


def is_sparse_point_cloud(
    points: np.ndarray,
    *,
    n_points_threshold: int = DEFAULT_SPARSE_THRESHOLD,
    knn_k: int = 8,
    knn_dist_threshold: Optional[float] = None,
) -> bool:
    """判断点云是否稀疏（点数少或平均近邻距离大）。"""
    n = len(points)
    if n < n_points_threshold:
        return True
    if knn_dist_threshold is not None:
        return estimate_mean_knn_distance(points, k=knn_k) > float(knn_dist_threshold)
    return False


def adaptive_split_n_neighbors(
    n_points: int,
    *,
    base: int = 16,
    sparse_threshold: int = DEFAULT_SPARSE_THRESHOLD,
    max_k: int = 48,
) -> int:
    """稀疏点云上调 kNN 邻居数，改善谱聚类 / 测地图连通性。"""
    n = max(1, int(n_points))
    if n >= sparse_threshold:
        return int(base)
    scaled = int(round(base * sparse_threshold / n))
    return int(min(max_k, max(base, scaled)))


def densify_for_neo(
    gt_points: np.ndarray,
    proxy_mesh: str | Path,
    n_target: int = DEFAULT_DENSIFY_TARGET,
    *,
    seed: int = 0,
) -> Tuple[np.ndarray, int]:
    """从 proxy mesh 密集采样，与 GT 点合并，供 NEO 推理。

    返回 (combined[N,3], n_gt)。``combined[:n_gt]`` 与 ``gt_points`` 逐行对齐。
    """
    gt = np.asarray(gt_points, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"gt_points 必须是 (N,3)，得到 {gt.shape}")
    n_gt = len(gt)
    if n_gt == 0:
        raise ValueError("空点云无法增强")

    path = Path(proxy_mesh)
    if not path.is_file():
        raise FileNotFoundError(f"proxy mesh 不存在: {path}")

    n_target = max(n_gt + 1, int(n_target))
    n_aux = n_target - n_gt

    # 从 mesh 表面采样辅助点（与 GT 坐标系对齐的 bundle proxy）。
    aux, _src = load_pointcloud(path, num_sample_points=n_aux, seed=seed)
    if len(aux) < n_aux:
        # load_pointcloud 可能受 mesh 面积限制；再补一轮随机面采样。
        import trimesh

        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
            need = n_aux - len(aux)
            extra, _ = trimesh.sample.sample_surface(mesh, need)
            aux = np.vstack([aux, np.asarray(extra, dtype=np.float64)])
    aux = np.asarray(aux[:n_aux], dtype=np.float64)

    combined = np.ascontiguousarray(np.vstack([gt, aux]))
    LOGGER.info(
        "稀疏点云 mesh 增强: GT=%d + aux=%d -> NEO 输入 %d 点 (proxy=%s)",
        n_gt, len(aux), len(combined), path.name,
    )
    return combined, n_gt


def densify_for_neo_from_arrays(
    gt_points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    n_target: int = DEFAULT_DENSIFY_TARGET,
    *,
    seed: int = 0,
) -> Tuple[np.ndarray, int]:
    """已有 mesh 顶点/面时，直接从表面采样辅助点（免 IO）。"""
    gt = np.asarray(gt_points, dtype=np.float64)
    n_gt = len(gt)
    n_target = max(n_gt + 1, int(n_target))
    n_aux = n_target - n_gt
    aux = sample_points_from_mesh(vertices, faces, n_aux, seed=seed)
    combined = np.ascontiguousarray(np.vstack([gt, aux]))
    return combined, n_gt


def sparse_partition_params(n_points: int) -> Dict[str, int]:
    """稀疏 benchmark 点云的分块参数建议。"""
    return {
        "split_n_neighbors": adaptive_split_n_neighbors(n_points),
    }


__all__ = [
    "DEFAULT_DENSIFY_TARGET",
    "DEFAULT_SPARSE_THRESHOLD",
    "adaptive_split_n_neighbors",
    "densify_for_neo",
    "densify_for_neo_from_arrays",
    "estimate_mean_knn_distance",
    "is_sparse_point_cloud",
    "sparse_partition_params",
]
