"""Spectral partition: geodesic 2D embed → Voronoi seeds → connected face groups."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.spectral_cluster import spectral_cluster_faces

from .common import SubBlockSplit, extract_submesh
from .connectivity import connected_face_groups_from_labels

LOGGER = logging.getLogger("geometry.partition.spectral")


def _connected_groups_from_face_labels(
    faces: np.ndarray,
    face_labels: np.ndarray,
    face_ids_list: Sequence[int],
) -> Tuple[np.ndarray, List[List[int]]]:
    """每个输出 group 都是面邻接图上的一个连通分量。"""
    labels = np.asarray(face_labels, dtype=np.int64).reshape(-1)
    labels, groups_local = connected_face_groups_from_labels(faces, labels)
    groups = [
        [int(face_ids_list[i]) for i in g]
        for g in groups_local
        if g
    ]
    return labels, groups


def split_block_by_spectral_seeds(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    seed_count: int,
    *,
    vertex_scalars: Optional[np.ndarray] = None,
    geodesic_weight: Optional[float] = None,
    curvature_weight: Optional[float] = None,
    concavity_weight: Optional[float] = None,
    graph_vertex_budget: Optional[int] = None,
) -> SubBlockSplit:
    _ = vertex_scalars, graph_vertex_budget
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
            debug={"reason": "submesh 太小，跳过分裂"},
        )

    face_labels, k_actual, debug = spectral_cluster_faces(
        sub_v,
        sub_f,
        seed_count,
        geodesic_weight=geodesic_weight,
        curvature_weight=curvature_weight,
        concavity_weight=concavity_weight,
    )
    local_labels, groups = _connected_groups_from_face_labels(
        sub_f, face_labels, face_ids_list
    )
    groups = [g for g in groups if g]

    return SubBlockSplit(
        sub_face_groups=groups,
        sub_labels=local_labels,
        num_clusters=len(groups),
        debug={**debug, "spectral_k_actual": int(k_actual), "connected_groups": len(groups)},
    )
