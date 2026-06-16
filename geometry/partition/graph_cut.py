"""Graph-cut partition with concavity-aware edge weights."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from .common import (
    SubBlockSplit,
    build_face_adjacency_csr,
    enforce_per_label_face_connectivity,
    extract_submesh,
    face_centroids_and_normals,
)


def _face_geodesic_to_seeds(
    adj: sp.csr_matrix,
    seed_face_indices: Sequence[int],
) -> np.ndarray:
    if not seed_face_indices:
        return np.zeros((0, adj.shape[0]), dtype=np.float64)
    seeds = np.asarray(list(seed_face_indices), dtype=np.int64)
    dists = dijkstra(adj, indices=seeds, directed=False)
    return np.where(np.isfinite(dists), dists, np.inf)


def _edge_concavity_weights(
    centroids: np.ndarray,
    normals: np.ndarray,
    adj: sp.csr_matrix,
    *,
    lambda_smooth: float,
    lambda_concave: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = int(adj.shape[0])
    rows, cols, data = [], [], []
    indptr, indices = adj.indptr, adj.indices
    for i in range(m):
        for j in indices[indptr[i]: indptr[i + 1]]:
            if j <= i:
                continue
            diff = centroids[i] - centroids[j]
            dist = max(float(np.linalg.norm(diff)), 1e-9)
            cos_n = float(np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0))
            concave = max(0.0, -cos_n)
            convex = max(0.0, cos_n)
            w = (
                lambda_smooth
                * np.exp(-dist / max(dist, 1e-6))
                * (1.0 + lambda_concave * convex - 0.5 * lambda_concave * concave)
            )
            w = max(float(w), 1e-6)
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([w, w])
    if not rows:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(data, dtype=np.float64),
    )


def _graph_cut_binary(
    unary_source: np.ndarray,
    unary_sink: np.ndarray,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    edge_weights: np.ndarray,
    *,
    max_iter: int = 12,
) -> np.ndarray:
    n = len(unary_source)
    labels = (unary_sink < unary_source).astype(np.int64)
    if n <= 1:
        return labels

    adj_map: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for r, c, w in zip(edge_rows, edge_cols, edge_weights):
        adj_map[int(r)].append((int(c), float(w)))
        adj_map[int(c)].append((int(r), float(w)))

    for _ in range(max_iter):
        changed = False
        order = np.random.default_rng(0).permutation(n)
        for i in order:
            cost0 = float(unary_source[i])
            cost1 = float(unary_sink[i])
            for j, w in adj_map[i]:
                if labels[j] == 0:
                    cost0 += w
                else:
                    cost1 += w
            new_lab = 0 if cost0 <= cost1 else 1
            if new_lab != labels[i]:
                labels[i] = new_lab
                changed = True
        if not changed:
            break
    return labels


def _alpha_expansion_labels(
    unary: np.ndarray,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    edge_weights: np.ndarray,
    *,
    n_rounds: int = 3,
) -> np.ndarray:
    m, k = unary.shape
    labels = np.argmin(unary, axis=1).astype(np.int64)
    if k <= 1:
        return labels

    for _ in range(n_rounds):
        for alpha in range(k):
            source_cost = np.where(labels == alpha, unary[:, alpha], 1e6)
            sink_cost = np.where(labels != alpha, unary[:, alpha], 1e6)
            binary = _graph_cut_binary(
                source_cost, sink_cost, edge_rows, edge_cols, edge_weights
            )
            labels = np.where(binary == 0, alpha, labels)
    return labels


def split_block_by_graph_cut(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    seed_count: int,
    *,
    lambda_smooth: float = 1.0,
    lambda_concave: float = 0.85,
    vertex_scalars=None,
    scalar_weight: float = 0.35,
) -> SubBlockSplit:
    face_ids_list = list(face_ids)
    seed_count = max(2, int(seed_count))
    if not face_ids_list:
        return SubBlockSplit(
            sub_face_groups=[], sub_labels=np.zeros(0, dtype=np.int64), num_clusters=0
        )

    sub_v, sub_f, _ = extract_submesh(vertices, faces, face_ids_list)
    m = len(sub_f)
    if m <= seed_count:
        return SubBlockSplit(
            sub_face_groups=[list(face_ids_list)],
            sub_labels=np.zeros(len(face_ids_list), dtype=np.int64),
            num_clusters=1,
            debug={"reason": "submesh 太小"},
        )

    centroids, normals = face_centroids_and_normals(sub_v, sub_f)
    adj = build_face_adjacency_csr(sub_f)

    center = centroids.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(centroids - center, axis=1)))
    seeds: List[int] = [start]
    dist_to_seeds = np.linalg.norm(centroids - centroids[start], axis=1)
    for _ in range(1, min(seed_count, m)):
        nxt = int(np.argmax(dist_to_seeds))
        if dist_to_seeds[nxt] <= 0:
            break
        seeds.append(nxt)
        new_d = np.linalg.norm(centroids - centroids[nxt], axis=1)
        dist_to_seeds = np.minimum(dist_to_seeds, new_d)

    geo_dists = _face_geodesic_to_seeds(adj, seeds)
    unary = geo_dists.T.copy()
    for col in range(len(seeds)):
        col_data = unary[:, col]
        finite = col_data[np.isfinite(col_data)]
        scale = float(np.median(finite)) if finite.size else 1.0
        unary[:, col] = col_data / max(scale, 1e-6)

    er, ec, ew = _edge_concavity_weights(
        centroids,
        normals,
        adj,
        lambda_smooth=float(lambda_smooth),
        lambda_concave=float(lambda_concave),
    )
    labels = _alpha_expansion_labels(unary, er, ec, ew)
    from .connectivity import smooth_face_label_boundaries

    labels = smooth_face_label_boundaries(sub_f, labels, passes=2)
    labels = enforce_per_label_face_connectivity(sub_f, labels)

    used = sorted({int(v) for v in labels})
    groups: List[List[int]] = []
    for lab in used:
        idx = np.flatnonzero(labels == lab)
        group = [face_ids_list[i] for i in idx.tolist()]
        if group:
            groups.append(group)

    return SubBlockSplit(
        sub_face_groups=groups,
        sub_labels=labels,
        num_clusters=len(groups),
        debug={
            "strategy": "graph_cut",
            "seed_faces_local": seeds,
            "lambda_smooth": float(lambda_smooth),
            "lambda_concave": float(lambda_concave),
            "k_actual": len(groups),
        },
    )
