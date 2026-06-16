"""Geodesic FPS + Lloyd Voronoi partition."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.csgraph import dijkstra

from geometry.spectral_cluster import vertex_labels_to_face_labels

from .common import (
    SubBlockSplit,
    extract_submesh,
)
from .scale import (
    VertexGraphView,
    build_vertex_graph_view,
    effective_graph_vertex_budget,
)

LOGGER = logging.getLogger("geometry.partition.geodesic_fps")


def geodesic_fps_seeds(
    graph: np.ndarray,
    sub_vertices: np.ndarray,
    k: int,
    *,
    start_idx: Optional[int] = None,
) -> List[int]:
    """Farthest-point sampling on a geodesic graph."""
    n_v = int(graph.shape[0])
    if k <= 0 or n_v == 0:
        return []
    k = min(k, n_v)

    if start_idx is None:
        center = sub_vertices.mean(axis=0)
        start_idx = int(np.argmax(np.linalg.norm(sub_vertices - center, axis=1)))

    seeds: List[int] = [int(start_idx)]
    dist = dijkstra(graph, indices=start_idx, directed=False)
    dist = np.where(np.isfinite(dist), dist, -np.inf)

    for _ in range(1, k):
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0.0:
            break
        seeds.append(nxt)
        new_dist = dijkstra(graph, indices=nxt, directed=False)
        new_dist = np.where(np.isfinite(new_dist), new_dist, -np.inf)
        dist = np.minimum(dist, new_dist)

    return seeds


def geodesic_voronoi_assign(graph: np.ndarray, seeds: Sequence[int]) -> np.ndarray:
    if not seeds:
        return np.zeros(graph.shape[0], dtype=np.int64)
    indices = np.asarray(list(seeds), dtype=np.int64)
    dists = dijkstra(graph, indices=indices, directed=False)
    return np.argmin(dists, axis=0).astype(np.int64)


def lloyd_relax_geodesic(
    graph: np.ndarray,
    sub_vertices: np.ndarray,
    seeds: List[int],
    *,
    n_iter: int = 5,
    reseed_min_frac: float = 0.15,
) -> Tuple[List[int], np.ndarray]:
    """测地 Lloyd 松弛；空/过小簇会被搬到「最差覆盖点」重新播种。

    若某个种子被高代价边界（如凹缝）困住、Voronoi 区域退化成几面的碎簇，
    普通 Lloyd 只会把它再钉回碎簇里出不来。这里检测大小 < ``reseed_min_frac``×
    平均簇大小的簇，把它的种子搬到当前距所有种子最远的点（通常落在最大、覆盖
    最差的区域），从而把大块劈开、均衡簇大小，避免「一边完美、一边整块连到身子」。
    """
    if not seeds:
        return [], np.zeros(graph.shape[0], dtype=np.int64)

    seeds = list(seeds)
    n = int(graph.shape[0])
    k = len(seeds)

    def _assign(seed_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        d = dijkstra(
            graph, indices=np.asarray(seed_list, dtype=np.int64), directed=False
        )
        d = np.asarray(d, dtype=np.float64)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        return np.argmin(d, axis=0).astype(np.int64), d

    labels, dists = _assign(seeds)
    if int(n_iter) <= 0:
        return seeds, labels

    min_size = max(1, int(float(reseed_min_frac) * n / max(k, 1)))

    for _ in range(int(n_iter)):
        new_seeds: List[int] = list(seeds)
        changed = False
        dead: List[int] = []
        for ci, cur in enumerate(seeds):
            mask = labels == ci
            sz = int(np.count_nonzero(mask))
            if sz < min_size:
                dead.append(ci)
                continue
            cluster_mean = sub_vertices[mask].mean(axis=0)
            d3 = np.linalg.norm(sub_vertices - cluster_mean, axis=1)
            new_seed = int(np.argmin(np.where(mask, d3, np.inf)))
            if new_seed != cur:
                changed = True
            new_seeds[ci] = new_seed

        if dead:
            # 距所有（健康）种子的最近距离 = 覆盖质量；最大处最该补种子
            cover = np.min(dists, axis=0)
            cover = np.where(np.isfinite(cover), cover, 0.0)
            chosen = set(int(s) for s in new_seeds)
            for ci in dead:
                cand = cover.copy()
                if chosen:
                    cand[list(chosen)] = -np.inf
                far = int(np.argmax(cand))
                if cand[far] <= 0.0:
                    continue
                new_seeds[ci] = far
                chosen.add(far)
                changed = True
                nd = dijkstra(graph, indices=far, directed=False)
                nd = np.where(np.isfinite(nd), nd, np.inf)
                cover = np.minimum(cover, nd)

        if not changed:
            break
        seeds = new_seeds
        labels, dists = _assign(seeds)
    return seeds, labels


def lloyd_relax_geodesic_view(
    view: VertexGraphView,
    full_vertices: np.ndarray,
    seeds_graph: List[int],
    *,
    n_iter: int = 5,
) -> Tuple[List[int], np.ndarray]:
    """Lloyd on pooled graph; cluster centroids picked on full-resolution vertices."""
    if not seeds_graph:
        return [], np.zeros(view.num_fine, dtype=np.int64)

    seeds = [int(s) for s in seeds_graph]
    labels_graph = geodesic_voronoi_assign(view.graph, seeds)
    labels_fine = view.labels_graph_to_fine(labels_graph)
    full_vertices = np.asarray(full_vertices, dtype=np.float64)

    if int(n_iter) <= 0:
        return seeds, labels_fine
    for _ in range(int(n_iter)):
        new_seeds: List[int] = []
        changed = False
        for k, cur in enumerate(seeds):
            mask_fine = labels_fine == k
            if not np.any(mask_fine):
                new_seeds.append(cur)
                continue
            mean = full_vertices[mask_fine].mean(axis=0)
            d3 = np.linalg.norm(full_vertices - mean, axis=1)
            d3_masked = np.where(mask_fine, d3, np.inf)
            new_fine = int(np.argmin(d3_masked))
            new_g = int(view.fine_to_graph[new_fine])
            if new_g != cur:
                changed = True
            new_seeds.append(new_g)
        if not changed:
            break
        seeds = new_seeds
        labels_graph = geodesic_voronoi_assign(view.graph, seeds)
        labels_fine = view.labels_graph_to_fine(labels_graph)
    return seeds, labels_fine


def split_block_by_geodesic_fps(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    seed_count: int,
    *,
    vertex_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
    lloyd_iter: int = 0,
    graph_vertex_budget: Optional[int] = None,
) -> SubBlockSplit:
    _ = vertex_scalars, scalar_weight
    face_ids_list = list(face_ids)
    seed_count = max(2, int(seed_count))
    if not face_ids_list:
        return SubBlockSplit(
            sub_face_groups=[], sub_labels=np.zeros(0, dtype=np.int64), num_clusters=0
        )

    sub_v, sub_f, _global_v_idx = extract_submesh(vertices, faces, face_ids_list)
    if len(sub_v) <= seed_count or len(sub_f) <= 1:
        return SubBlockSplit(
            sub_face_groups=[list(face_ids_list)],
            sub_labels=np.zeros(len(face_ids_list), dtype=np.int64),
            num_clusters=1,
            debug={"reason": "submesh 太小，跳过 FPS 分裂"},
        )

    budget = effective_graph_vertex_budget(len(sub_v), graph_vertex_budget)
    view = build_vertex_graph_view(sub_v, sub_f, max_vertices=budget)

    fps_seeds: List[int] = []
    seeds_final: List[int] = []
    fps_error: Optional[str] = None
    strategy = "geodesic_fps_voronoi_lloyd"
    if view.pooled:
        strategy = "geodesic_fps_voronoi_lloyd_pooled"

    try:
        fps_seeds = geodesic_fps_seeds(view.graph, view.coords, seed_count)
        if len(fps_seeds) < 2:
            raise RuntimeError(f"FPS 仅取得 {len(fps_seeds)} 个种子")
        if int(lloyd_iter) > 0:
            seeds_final, v_labels = lloyd_relax_geodesic_view(
                view, sub_v, list(fps_seeds), n_iter=lloyd_iter
            )
        else:
            seeds_final = list(fps_seeds)
            labels_graph = geodesic_voronoi_assign(view.graph, seeds_final)
            v_labels = view.labels_graph_to_fine(labels_graph)
    except Exception as exc:  # noqa: BLE001
        fps_error = str(exc)
        LOGGER.warning("测地 FPS+Voronoi 失败 (%s)，回退坐标 KMeans", exc)
        try:
            from geometry.spectral_cluster import _kmeans

            v_labels = _kmeans(sub_v.astype(np.float64), seed_count, random_state=0)
            strategy = "coord_kmeans_fallback"
        except Exception as exc2:  # noqa: BLE001
            return SubBlockSplit(
                sub_face_groups=[list(face_ids_list)],
                sub_labels=np.zeros(len(face_ids_list), dtype=np.int64),
                num_clusters=1,
                debug={
                    "error": f"{exc} | {exc2}",
                    "fallback": "all_failed",
                    "fps_error": fps_error,
                },
            )

    sub_face_labels = vertex_labels_to_face_labels(v_labels, sub_f)
    sub_face_labels_arr = np.asarray(sub_face_labels, dtype=np.int64)

    from .connectivity import build_k_connected_face_groups

    sub_face_labels_arr, groups = build_k_connected_face_groups(
        sub_v,
        sub_f,
        face_ids_list,
        sub_face_labels_arr,
        seed_count,
    )

    debug: Dict[str, Any] = {
        "strategy": strategy,
        "seed_count_requested": seed_count,
        "seed_count_actual": len(groups),
        "fps_seed_graph_v_idx": fps_seeds,
        "lloyd_seed_graph_v_idx": seeds_final,
        "fps_error": fps_error,
        "lloyd_iter": lloyd_iter,
        "graph_vertex_budget": budget,
        "graph_vertices": view.num_graph,
        "fine_vertices": view.num_fine,
        "pooled_graph": view.pooled,
    }
    return SubBlockSplit(
        sub_face_groups=groups,
        sub_labels=sub_face_labels_arr,
        num_clusters=len(groups),
        debug=debug,
    )
