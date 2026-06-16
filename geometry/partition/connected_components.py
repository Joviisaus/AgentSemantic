"""Split a block into connected components."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.sparse.csgraph import connected_components

from geometry.spectral_cluster import vertex_labels_to_face_labels

from .common import build_submesh_vertex_graph, extract_submesh


def split_block_by_connected_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    *,
    min_component_faces: int = 1,
) -> Optional[List[List[int]]]:
    """
    Split ``face_ids`` into connected components.

    Returns ``None`` when there is only one component (caller continues with
    geodesic / spectral split).
    """
    face_ids_list = list(face_ids)
    if len(face_ids_list) <= 1:
        return None

    sub_v, sub_f, _ = extract_submesh(vertices, faces, face_ids_list)
    if len(sub_v) == 0 or len(sub_f) == 0:
        return None

    graph = build_submesh_vertex_graph(sub_v, sub_f)
    n_cc, v_labels = connected_components(graph, directed=False)
    if n_cc <= 1:
        return None

    sub_face_labels = vertex_labels_to_face_labels(v_labels, sub_f)
    sub_face_labels_arr = np.asarray(sub_face_labels, dtype=np.int64)

    raw_groups: List[List[int]] = []
    for k in range(n_cc):
        idx = np.flatnonzero(sub_face_labels_arr == k)
        if idx.size == 0:
            continue
        raw_groups.append([face_ids_list[i] for i in idx.tolist()])

    if len(raw_groups) <= 1:
        return None

    raw_groups.sort(key=len, reverse=True)
    if min_component_faces > 1 and len(raw_groups) > 1:
        big = raw_groups[0]
        kept: List[List[int]] = [big]
        for g in raw_groups[1:]:
            if len(g) < min_component_faces:
                big.extend(g)
            else:
                kept.append(g)
        raw_groups = kept

    if len(raw_groups) <= 1:
        return None
    return raw_groups
