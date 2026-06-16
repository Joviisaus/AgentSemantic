"""Skeleton-guided partition via Laplacian mesh contraction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

from geometry.spectral_cluster import vertex_labels_to_face_labels

from .common import (
    SubBlockSplit,
    extract_submesh,
)
from .geodesic_fps import (
    geodesic_fps_seeds,
    geodesic_voronoi_assign,
    lloyd_relax_geodesic_view,
    split_block_by_geodesic_fps,
)
from .scale import build_vertex_graph_view, effective_graph_vertex_budget

LOGGER = logging.getLogger("geometry.partition.skeleton")


def _build_combinatorial_laplacian(graph: sp.csr_matrix) -> sp.csr_matrix:
    n = int(graph.shape[0])
    degree = np.asarray(graph.sum(axis=1)).flatten()
    return sp.diags(degree, format="csr", dtype=np.float64) - graph.astype(np.float64)


def laplacian_contract_graph(
    graph: sp.csr_matrix,
    coords: np.ndarray,
    *,
    n_iters: int = 5,
    weight_scale: float = 1.0,
) -> np.ndarray:
    """在（可池化后的）顶点图上做 Laplacian 收缩。"""
    n = int(graph.shape[0])
    if n == 0:
        return np.asarray(coords, dtype=np.float64).copy()

    lap = _build_combinatorial_laplacian(graph)
    eye = sp.eye(n, format="csr", dtype=np.float64)
    contracted = np.asarray(coords, dtype=np.float64).copy()

    for i in range(max(1, int(n_iters))):
        w = float(weight_scale) * (2.0 ** i)
        system = (lap + w * eye).tocsc()
        rhs = w * contracted
        try:
            solve = splu(system).solve
            for dim in range(3):
                contracted[:, dim] = solve(rhs[:, dim])
        except Exception:  # noqa: BLE001
            for dim in range(3):
                contracted[:, dim] = spsolve(system, rhs[:, dim])

    return contracted


def _skeleton_candidate_indices(
    original: np.ndarray,
    contracted: np.ndarray,
    *,
    percentile: float = 40.0,
) -> np.ndarray:
    """收缩位移较大的顶点更靠近中轴骨架。"""
    disp = np.linalg.norm(contracted - original, axis=1)
    if len(disp) == 0:
        return np.zeros(0, dtype=np.int64)
    thresh = float(np.percentile(disp, percentile))
    mask = disp >= max(thresh, 1e-9)
    if not np.any(mask):
        mask = disp >= float(np.max(disp)) * 0.5
    return np.flatnonzero(mask).astype(np.int64)


def skeleton_fps_seeds(
    graph: sp.csr_matrix,
    original: np.ndarray,
    contracted: np.ndarray,
    k: int,
    *,
    candidate_idx: Optional[np.ndarray] = None,
) -> List[int]:
    """在收缩点云上最远点采样，得到沿骨架分布的种子（图顶点索引）。"""
    n_v = int(graph.shape[0])
    k = min(max(2, int(k)), n_v)
    if k <= 0 or n_v == 0:
        return []

    if candidate_idx is None or len(candidate_idx) == 0:
        candidate_idx = _skeleton_candidate_indices(original, contracted)
    if len(candidate_idx) == 0:
        candidate_idx = np.arange(n_v, dtype=np.int64)

    center = contracted[candidate_idx].mean(axis=0)
    start_local = int(
        candidate_idx[
            int(np.argmax(np.linalg.norm(contracted[candidate_idx] - center, axis=1)))
        ]
    )

    seeds: List[int] = [start_local]
    cand = np.asarray(candidate_idx, dtype=np.int64)
    min_d = np.linalg.norm(contracted[cand] - contracted[start_local], axis=1)

    for _ in range(1, k):
        pick_in_cand = int(np.argmax(min_d))
        nxt = int(cand[pick_in_cand])
        if min_d[pick_in_cand] <= 0.0:
            break
        seeds.append(nxt)
        new_d = np.linalg.norm(contracted[cand] - contracted[nxt], axis=1)
        min_d = np.minimum(min_d, new_d)

    if len(seeds) < k:
        extra = geodesic_fps_seeds(graph, original, k, start_idx=seeds[0])
        for s in extra:
            if s not in seeds:
                seeds.append(int(s))
            if len(seeds) >= k:
                break
    return seeds[:k]


def split_block_by_skeleton(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    seed_count: int,
    *,
    vertex_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
    contraction_iters: int = 5,
    lloyd_iter: int = 0,
    graph_vertex_budget: Optional[int] = None,
) -> SubBlockSplit:
    """Laplacian 收缩提取骨架走向，再测地 Voronoi 切分（大图自动池化图）。"""
    _ = vertex_scalars, scalar_weight
    face_ids_list = list(face_ids)
    seed_count = max(2, int(seed_count))
    if not face_ids_list:
        return SubBlockSplit(
            sub_face_groups=[], sub_labels=np.zeros(0, dtype=np.int64), num_clusters=0
        )

    sub_v, sub_f, _global_v_idx = extract_submesh(vertices, faces, face_ids_list)
    n_v = len(sub_v)
    if n_v <= seed_count or len(sub_f) <= 1:
        return SubBlockSplit(
            sub_face_groups=[list(face_ids_list)],
            sub_labels=np.zeros(len(face_ids_list), dtype=np.int64),
            num_clusters=1,
            debug={"reason": "submesh 太小，跳过 skeleton 分裂"},
        )

    budget = effective_graph_vertex_budget(n_v, graph_vertex_budget)
    view = build_vertex_graph_view(sub_v, sub_f, max_vertices=budget)

    fps_seeds: List[int] = []
    seeds_final: List[int] = []
    skel_error: Optional[str] = None
    num_skeleton_candidates = 0
    strategy = "laplacian_contract_skeleton_voronoi"
    if view.pooled:
        strategy = "laplacian_contract_skeleton_voronoi_pooled"

    LOGGER.info(
        "skeleton 分块: fine_v=%d graph_v=%d k=%d iters=%d",
        view.num_fine,
        view.num_graph,
        seed_count,
        int(contraction_iters),
    )
    try:
        contracted = laplacian_contract_graph(
            view.graph,
            view.coords,
            n_iters=int(contraction_iters),
        )
        cand = _skeleton_candidate_indices(view.coords, contracted)
        num_skeleton_candidates = int(len(cand))
        fps_seeds = skeleton_fps_seeds(
            view.graph, view.coords, contracted, seed_count, candidate_idx=cand
        )
        if len(fps_seeds) < 2:
            raise RuntimeError(f"骨架种子仅 {len(fps_seeds)} 个")
        if int(lloyd_iter) > 0:
            seeds_final, v_labels = lloyd_relax_geodesic_view(
                view, sub_v, list(fps_seeds), n_iter=lloyd_iter
            )
        else:
            seeds_final = list(fps_seeds)
            labels_graph = geodesic_voronoi_assign(view.graph, seeds_final)
            v_labels = view.labels_graph_to_fine(labels_graph)
    except Exception as exc:  # noqa: BLE001
        skel_error = str(exc)
        LOGGER.warning("骨架分块失败 (%s)，回退 geodesic_fps", exc)
        fb = split_block_by_geodesic_fps(
            vertices,
            faces,
            face_ids,
            seed_count,
            lloyd_iter=lloyd_iter,
            graph_vertex_budget=budget,
        )
        dbg = dict(fb.debug or {})
        dbg["fallback"] = "geodesic_fps_after_skeleton_error"
        dbg["skeleton_error"] = skel_error
        return SubBlockSplit(
            sub_face_groups=fb.sub_face_groups,
            sub_labels=fb.sub_labels,
            num_clusters=fb.num_clusters,
            debug=dbg,
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
        "skeleton_seed_graph_v_idx": fps_seeds,
        "lloyd_seed_graph_v_idx": seeds_final,
        "contraction_iters": int(contraction_iters),
        "lloyd_iter": lloyd_iter,
        "skeleton_error": skel_error,
        "num_skeleton_candidates": num_skeleton_candidates,
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
