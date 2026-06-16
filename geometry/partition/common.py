"""Shared types and mesh helpers for block partition strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


@dataclass
class SubBlockSplit:
    sub_face_groups: List[List[int]]
    sub_labels: np.ndarray
    num_clusters: int
    debug: Dict[str, Any] = field(default_factory=dict)


def extract_submesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sub-mesh for ``face_ids``; return (sub_v, sub_f, global_v_idx)."""
    face_ids_arr = np.asarray(list(face_ids), dtype=np.int64)
    sub_face = faces[face_ids_arr]
    unique_v, inverse = np.unique(sub_face.reshape(-1), return_inverse=True)
    new_faces = inverse.reshape(sub_face.shape).astype(np.int64)
    sub_vertices = vertices[unique_v]
    return sub_vertices, new_faces, unique_v


def build_submesh_vertex_graph(sub_vertices: np.ndarray, sub_faces: np.ndarray) -> sp.csr_matrix:
    """Vertex adjacency graph with Euclidean edge lengths as weights."""
    n_v = int(sub_vertices.shape[0])
    if sub_faces.size == 0:
        return sp.csr_matrix((n_v, n_v))

    tri = sub_faces.astype(np.int64)
    a = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    b = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    diff = sub_vertices[rows] - sub_vertices[cols]
    weights = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
    return sp.csr_matrix((weights, (rows, cols)), shape=(n_v, n_v))


def build_face_adjacency_csr(faces: np.ndarray) -> sp.csr_matrix:
    """Face adjacency graph (shared edge = neighbor)."""
    m = int(len(faces))
    if m == 0:
        return sp.csr_matrix((0, 0), dtype=np.int8)
    f = np.asarray(faces, dtype=np.int64)
    a0, a1, a2 = f[:, 0], f[:, 1], f[:, 2]
    fid = np.arange(m, dtype=np.int64)
    edges = np.vstack([
        np.column_stack([np.minimum(a0, a1), np.maximum(a0, a1), fid]),
        np.column_stack([np.minimum(a1, a2), np.maximum(a1, a2), fid]),
        np.column_stack([np.minimum(a2, a0), np.maximum(a2, a0), fid]),
    ])
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    es = edges[order]
    same = (es[1:, 0] == es[:-1, 0]) & (es[1:, 1] == es[:-1, 1])
    if not np.any(same):
        return sp.csr_matrix((m, m), dtype=np.int8)
    src = es[:-1, 2][same]
    dst = es[1:, 2][same]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix((data, (rows, cols)), shape=(m, m))


def face_centroids_and_normals(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces.astype(np.int64)]
    centroids = tri.mean(axis=1)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    normals = fn / np.maximum(norms, 1e-12)
    return centroids, normals


def enforce_per_label_face_connectivity(
    faces: np.ndarray,
    face_labels: np.ndarray,
    *,
    max_iter: int = 8,
) -> np.ndarray:
    """Ensure each label spans a single connected face region."""
    labels = np.asarray(face_labels, dtype=np.int64).copy()
    m = int(len(faces))
    if m <= 1:
        return labels

    adj = build_face_adjacency_csr(faces)
    if adj.nnz == 0:
        return labels
    indptr = adj.indptr
    indices = adj.indices

    for _it in range(max_iter):
        row_idx = np.repeat(np.arange(m, dtype=np.int64), np.diff(indptr))
        col_idx = indices.astype(np.int64)
        same_label = labels[row_idx] == labels[col_idx]
        if np.any(same_label):
            sub = sp.csr_matrix(
                (
                    np.ones(int(same_label.sum()), dtype=np.int8),
                    (row_idx[same_label], col_idx[same_label]),
                ),
                shape=(m, m),
            )
            _n_cc, cc_labels = connected_components(sub, directed=False)
        else:
            cc_labels = np.arange(m, dtype=np.int64)
        cc_labels = cc_labels.astype(np.int64)

        unique_labs = np.unique(labels)
        problematic: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for lab in unique_labs:
            mask = labels == lab
            if not mask.any():
                continue
            sub_cc = cc_labels[mask]
            cc_ids, cc_sizes = np.unique(sub_cc, return_counts=True)
            if len(cc_ids) <= 1:
                continue
            problematic.append((int(lab), cc_ids, cc_sizes))

        if not problematic:
            break

        reassign = np.full(m, -1, dtype=np.int64)
        for lab, cc_ids, cc_sizes in problematic:
            largest_cc = int(cc_ids[int(np.argmax(cc_sizes))])
            for cc_id in cc_ids:
                cc_id_int = int(cc_id)
                if cc_id_int == largest_cc:
                    continue
                frag_idx = np.where(cc_labels == cc_id_int)[0]
                if frag_idx.size == 0:
                    continue
                neighbor_labs: List[int] = []
                for fi in frag_idx:
                    nbrs = indices[indptr[fi]:indptr[fi + 1]]
                    if nbrs.size == 0:
                        continue
                    out_mask = cc_labels[nbrs] != cc_id_int
                    if not np.any(out_mask):
                        continue
                    neighbor_labs.extend(labels[nbrs[out_mask]].tolist())
                if not neighbor_labs:
                    continue
                vals, counts = np.unique(
                    np.asarray(neighbor_labs, dtype=np.int64), return_counts=True
                )
                new_lab = int(vals[int(np.argmax(counts))])
                reassign[frag_idx] = new_lab

        changed_mask = reassign >= 0
        if not np.any(changed_mask):
            break
        labels[changed_mask] = reassign[changed_mask]

    return labels
