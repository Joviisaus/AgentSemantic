"""Laplacian spectral decomposition for bottleneck / natural seam hints."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from geometry.skin_optimize import build_mesh_laplacian

_MAX_LAPLACIAN_VERTICES = 12_000


def compute_laplacian_eigenvectors(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    num_eigenvectors: int = 3,
    skip_constant: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cotangent Laplace-Beltrami eigenvectors on the sub-mesh.

    Returns (eigenvalues, eigenvectors) with shape (k,) and (n_v, k).
    """
    n_v = int(len(vertices))
    if n_v <= 2:
        return np.zeros(0, dtype=np.float64), np.zeros((n_v, 0), dtype=np.float64)
    if n_v > _MAX_LAPLACIAN_VERTICES:
        return np.zeros(0, dtype=np.float64), np.zeros((n_v, 0), dtype=np.float64)

    lap = build_mesh_laplacian(vertices, faces, cotangent=True)
    k = min(max(1, int(num_eigenvectors)), n_v - 1)
    if skip_constant:
        k = min(k + 1, n_v - 1)
    try:
        vals, vecs = eigsh(lap.astype(np.float64), k=k, which="SM")
    except Exception:
        return np.zeros(0, dtype=np.float64), np.zeros((n_v, 0), dtype=np.float64)

    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    if skip_constant and vecs.shape[1] > 1:
        vals = vals[1:]
        vecs = vecs[:, 1:]
    return vals.astype(np.float64), vecs.astype(np.float64)


def vertex_to_face_scalar(
    vertex_values: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Average vertex scalar field onto faces."""
    tri = faces.astype(np.int64)
    vals = vertex_values[tri]
    return vals.mean(axis=1)


def bottleneck_faces_from_eigenvector_on_mesh(
    faces: np.ndarray,
    face_scalar: np.ndarray,
    *,
    gradient_percentile: float = 75.0,
) -> np.ndarray:
    """Faces with high |∇f| on the face adjacency graph."""
    from geometry.partition.common import build_face_adjacency_csr

    m = int(len(face_scalar))
    if m <= 1:
        return np.zeros(0, dtype=np.int64)

    adj = build_face_adjacency_csr(faces)
    indptr, indices = adj.indptr, adj.indices
    grads = np.zeros(m, dtype=np.float64)
    for fi in range(m):
        nbrs = indices[indptr[fi]: indptr[fi + 1]]
        if nbrs.size == 0:
            continue
        diffs = np.abs(face_scalar[nbrs] - face_scalar[fi])
        grads[fi] = float(np.max(diffs))

    thresh = float(np.percentile(grads[grads > 0], gradient_percentile)) if np.any(grads > 0) else 0.0
    return np.flatnonzero(grads >= thresh).astype(np.int64)


def partition_faces_by_fiedler(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed_faces_a: Sequence[int],
    seed_faces_b: Sequence[int],
    *,
    num_eigenvectors: int = 2,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Binary face partition guided by the Fiedler vector and seed regions.

    Seeds vote on which side of the Fiedler median each face belongs to.
    """
    m = int(len(faces))
    labels = np.zeros(m, dtype=np.int64)
    debug: Dict[str, object] = {}
    if m <= 1 or (not seed_faces_a and not seed_faces_b):
        return labels, {"reason": "degenerate submesh"}

    _vals, vecs = compute_laplacian_eigenvectors(
        vertices, faces, num_eigenvectors=num_eigenvectors
    )
    if vecs.shape[1] == 0:
        return _partition_faces_by_seed_distance(
            vertices, faces, seed_faces_a, seed_faces_b, debug
        )

    fiedler_face = vertex_to_face_scalar(vecs[:, 0], faces)
    debug["fiedler_range"] = (
        float(np.min(fiedler_face)),
        float(np.max(fiedler_face)),
    )
    bottleneck = bottleneck_faces_from_eigenvector_on_mesh(faces, fiedler_face)
    debug["bottleneck_faces"] = bottleneck.tolist()

    median = float(np.median(fiedler_face))
    labels = (fiedler_face >= median).astype(np.int64)

    if seed_faces_a:
        mean_a = float(np.mean(fiedler_face[list(seed_faces_a)]))
        mean_b = float(np.mean(fiedler_face[list(seed_faces_b)])) if seed_faces_b else mean_a + 1.0
        if mean_a > mean_b:
            labels = 1 - labels

        for fi in seed_faces_a:
            labels[int(fi)] = 0
        for fi in seed_faces_b:
            labels[int(fi)] = 1

    return labels, debug


def _partition_faces_by_seed_distance(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed_faces_a: Sequence[int],
    seed_faces_b: Sequence[int],
    debug: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fast fallback: assign each face to nearer seed centroid (no eigsh)."""
    from geometry.partition.common import face_centroids_and_normals

    m = int(len(faces))
    labels = np.zeros(m, dtype=np.int64)
    if m <= 1:
        debug["partition"] = "degenerate"
        return labels, debug

    centroids, _ = face_centroids_and_normals(vertices, faces)
    if seed_faces_a and seed_faces_b:
        ca = centroids[list(seed_faces_a)].mean(axis=0)
        cb = centroids[list(seed_faces_b)].mean(axis=0)
        da = np.linalg.norm(centroids - ca, axis=1)
        db = np.linalg.norm(centroids - cb, axis=1)
        labels = (db < da).astype(np.int64)
        for fi in seed_faces_a:
            labels[int(fi)] = 0
        for fi in seed_faces_b:
            labels[int(fi)] = 1
    debug["partition"] = "seed_distance_fallback"
    return labels, debug
