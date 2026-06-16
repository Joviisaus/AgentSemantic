"""Find shortest concave seams (geodesic cuts along dihedral minima)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from geometry.partition.common import (
    build_face_adjacency_csr,
    face_centroids_and_normals,
)


def face_edge_concavity_cost(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    concave_discount: float = 0.15,
    convex_penalty: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per undirected face-adjacency edge: (row, col, weight, is_concave).

    Concave edges (inward fold) get low weight → seam prefers them.
    """
    centroids, normals = face_centroids_and_normals(vertices, faces)
    adj = build_face_adjacency_csr(faces)
    indptr, indices = adj.indptr, adj.indices
    m = int(len(faces))
    rows, cols, weights, conc = [], [], [], []

    for i in range(m):
        for j in indices[indptr[i]: indptr[i + 1]]:
            if int(j) <= i:
                continue
            diff = centroids[j] - centroids[i]
            dist = max(float(np.linalg.norm(diff)), 1e-9)
            cos_n = float(np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0))
            is_concave = (
                np.dot(normals[i], diff) + np.dot(normals[j], -diff)
            ) > 0.0 and cos_n < 0.95
            if is_concave:
                w = dist * float(concave_discount)
            elif cos_n < 0.0:
                w = dist * 0.5
            else:
                w = dist * float(convex_penalty)
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([w, w])
            conc.extend([is_concave, is_concave])

    if not rows:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=bool),
        )
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(conc, dtype=bool),
    )


def _build_weighted_face_graph(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    m: int,
) -> sp.csr_matrix:
    return sp.csr_matrix(
        (weights.astype(np.float64), (rows, cols)),
        shape=(m, m),
    )


def find_concave_seam_between_seeds(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed_faces_a: Sequence[int],
    seed_faces_b: Sequence[int],
    *,
    concave_discount: float = 0.15,
) -> Tuple[List[int], Dict[str, object]]:
    """
    Shortest path on the face graph from region A to region B through concave edges.

    Returns ordered local face indices forming the seam (may be empty).
    """
    m = int(len(faces))
    debug: Dict[str, object] = {}
    if m <= 1 or not seed_faces_a or not seed_faces_b:
        return [], {"reason": "missing seeds"}

    er, ec, ew, _ = face_edge_concavity_cost(
        vertices, faces, concave_discount=concave_discount
    )
    graph = _build_weighted_face_graph(er, ec, ew, m)

    dist_a = dijkstra(graph, indices=np.asarray(list(seed_faces_a), dtype=np.int64), directed=False)
    dist_a = np.min(dist_a, axis=0)
    dist_b = dijkstra(graph, indices=np.asarray(list(seed_faces_b), dtype=np.int64), directed=False)
    dist_b = np.min(dist_b, axis=0)

    valid = np.isfinite(dist_a) & np.isfinite(dist_b)
    if not np.any(valid):
        return [], {"reason": "disconnected regions"}

    meeting = int(np.argmin(dist_a + dist_b))
    debug["meeting_face"] = meeting

    # Trace back from meeting to A and B
    parent_a = _trace_shortest_path(graph, seed_faces_a, meeting)
    parent_b = _trace_shortest_path(graph, seed_faces_b, meeting)
    seam = parent_a + parent_b[::-1]
    seam = list(dict.fromkeys(seam))
    debug["seam_length"] = len(seam)
    return seam, debug


def _trace_shortest_path(
    graph: sp.csr_matrix,
    sources: Sequence[int],
    target: int,
) -> List[int]:
    """Greedy parent walk from target toward nearest source."""
    m = int(graph.shape[0])
    dist = dijkstra(
        graph,
        indices=np.asarray(list(sources), dtype=np.int64),
        directed=False,
    )
    dist = np.min(dist, axis=0)
    path = [int(target)]
    cur = int(target)
    visited = {cur}
    indptr, indices, data = graph.indptr, graph.indices, graph.data

    while cur not in sources and len(path) < m:
        nbrs = indices[indptr[cur]: indptr[cur + 1]]
        wts = data[indptr[cur]: indptr[cur + 1]]
        best = -1
        best_d = float(dist[cur])
        for nb, w in zip(nbrs, wts):
            nb_i = int(nb)
            if nb_i in visited:
                continue
            cand = float(dist[nb_i]) + float(w)
            if cand < best_d - 1e-9:
                best_d = cand
                best = nb_i
        if best < 0:
            break
        cur = best
        path.append(cur)
        visited.add(cur)
    return path


def cut_labels_by_seam(
    faces: np.ndarray,
    face_labels: np.ndarray,
    seam_faces: Sequence[int],
    *,
    anchor_label: int,
    other_label: int,
) -> np.ndarray:
    """Assign ``other_label`` to all faces separated from anchor by the seam."""
    from collections import deque

    labels = np.asarray(face_labels, dtype=np.int64).copy()
    m = int(len(labels))
    seam_set = set(int(f) for f in seam_faces)
    adj = build_face_adjacency_csr(faces)
    indptr, indices = adj.indptr, adj.indices

    for fi in seam_set:
        labels[fi] = other_label

    # Flood anchor_label from faces currently labeled anchor
    queue: deque[int] = deque()
    seen = np.zeros(m, dtype=bool)
    for fi in range(m):
        if labels[fi] == anchor_label and fi not in seam_set:
            queue.append(fi)
            seen[fi] = True

    while queue:
        fi = queue.popleft()
        for nb in indices[indptr[fi]: indptr[fi + 1]]:
            nb_i = int(nb)
            if seen[nb_i] or nb_i in seam_set:
                continue
            seen[nb_i] = True
            labels[nb_i] = anchor_label
            queue.append(nb_i)

    # Remaining → other_label
    for fi in range(m):
        if not seen[fi]:
            labels[fi] = other_label
    return labels
