"""Post-surgery boundary refinement via concavity-aware graph cut."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from geometry.partition.common import (
    build_face_adjacency_csr,
    enforce_per_label_face_connectivity,
    extract_submesh,
    face_centroids_and_normals,
)
from geometry.partition.graph_cut import (
    _alpha_expansion_labels,
    _edge_concavity_weights,
)


def build_data_unary_costs(
    face_semantic_index: Sequence[int],
    semantic_parts: Sequence[str],
    region_faces: Sequence[int],
    *,
    confident_faces: Optional[Set[int]] = None,
    uncertain_faces: Optional[Set[int]] = None,
    weight_high: float = 1e4,
    weight_low: float = 0.05,
    weight_default: float = 1.0,
) -> Tuple[np.ndarray, List[int]]:
    """
    Per-face unary costs for each semantic label in the region.

    Returns (unary[m, k], active_semantic_indices).
    """
    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    region = [int(f) for f in region_faces]
    active = sorted({int(fpi[f]) for f in region if 0 <= int(fpi[f]) < len(semantic_parts)})
    if not active:
        return np.zeros((len(region), 0), dtype=np.float64), []

    k = len(active)
    m = len(region)
    unary = np.full((m, k), float(weight_default), dtype=np.float64)
    idx_map = {sem: col for col, sem in enumerate(active)}
    conf = confident_faces or set()
    unc = uncertain_faces or set()

    for li, gfi in enumerate(region):
        cur = int(fpi[gfi])
        w_hi = float(weight_high) if gfi in conf else float(weight_default)
        w_lo = float(weight_low) if gfi in unc else float(weight_default) * 0.25
        for sem, col in idx_map.items():
            if sem == cur:
                unary[li, col] = w_lo
            else:
                unary[li, col] = w_hi
    return unary, active


def optimize_region_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    region_faces: Sequence[int],
    semantic_indices: Sequence[int],
    *,
    confident_faces: Optional[Set[int]] = None,
    uncertain_faces: Optional[Set[int]] = None,
    lambda_smooth: float = 1.0,
    lambda_concave: float = 0.5,
    forbidden_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    forbidden_penalty: float = 5.0,
    n_rounds: int = 3,
) -> Tuple[Dict[int, int], Dict[str, object]]:
    """
    Graph-cut energy minimization on a local region.

    E = Σ E_data(f, l) + λ Σ E_smooth(f_i, f_j, l_i, l_j)

    Returns global face_id → semantic_index overrides.
    """
    region = sorted(set(int(f) for f in region_faces))
    debug: Dict[str, object] = {"region_faces": len(region)}
    if len(region) < 2:
        return {}, debug

    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    sub_v, sub_f, _ = extract_submesh(vertices, faces, region)
    local_fpi = fpi[np.asarray(region, dtype=np.int64)]

    sem_list = [int(s) for s in semantic_indices]
    sem_to_col = {s: i for i, s in enumerate(sem_list)}
    m = len(sub_f)
    k = len(sem_list)
    unary = np.full((m, k), 1.0, dtype=np.float64)

    conf = confident_faces or set()
    unc = uncertain_faces or set()
    for li, gfi in enumerate(region):
        cur = int(local_fpi[li])
        if cur not in sem_to_col:
            continue
        for sem, col in sem_to_col.items():
            if gfi in conf:
                unary[li, col] = 0.01 if sem == cur else 1e4
            elif gfi in unc:
                unary[li, col] = 0.05 if sem == cur else 0.5
            else:
                unary[li, col] = 0.1 if sem == cur else 2.0

    centroids, normals = face_centroids_and_normals(sub_v, sub_f)
    adj = build_face_adjacency_csr(sub_f)
    er, ec, ew = _edge_concavity_weights(
        centroids,
        normals,
        adj,
        lambda_smooth=float(lambda_smooth),
        lambda_concave=float(lambda_concave),
    )

    if forbidden_pairs:
        forb = {tuple(sorted((int(a), int(b)))) for a, b in forbidden_pairs}
        indptr, indices = adj.indptr, adj.indices
        for i in range(m):
            la = int(local_fpi[i])
            for j in indices[indptr[i]: indptr[i + 1]]:
                j_i = int(j)
                if j_i <= i:
                    continue
                lb = int(local_fpi[j_i])
                if tuple(sorted((la, lb))) in forb:
                    mask = (er == i) & (ec == j_i)
                    ew[mask] *= float(forbidden_penalty)

    labels_col = _alpha_expansion_labels(unary, er, ec, ew, n_rounds=n_rounds)
    labels_col = enforce_per_label_face_connectivity(sub_f, labels_col)

    overrides: Dict[int, int] = {}
    for li, gfi in enumerate(region):
        col = int(labels_col[li])
        new_sem = sem_list[col]
        if int(fpi[gfi]) != new_sem:
            overrides[gfi] = new_sem

    debug["overrides"] = len(overrides)
    debug["lambda_smooth"] = float(lambda_smooth)
    debug["lambda_concave"] = float(lambda_concave)
    return overrides, debug


def refine_surgery_boundary(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    semantic_parts: Sequence[str],
    surgery_faces: Sequence[int],
    *,
    core_margin_rings: int = 3,
    lambda_smooth: float = 1.0,
    lambda_concave: float = 0.5,
) -> Tuple[Dict[int, int], Dict[str, object]]:
    """
    Phase-3 optimizer: high-confidence cores + low-weight surgery band.
    """
    from .local_region import expand_face_rings

    surgery = set(int(f) for f in surgery_faces)
    region = expand_face_rings(faces, list(surgery), rings=core_margin_rings)
    core = set(region) - surgery

    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    active = sorted({int(fpi[f]) for f in region if 0 <= int(fpi[f]) < len(semantic_parts)})

    return optimize_region_labels(
        vertices,
        faces,
        fpi.tolist(),
        sorted(region),
        active,
        confident_faces=core,
        uncertain_faces=surgery,
        lambda_smooth=lambda_smooth,
        lambda_concave=lambda_concave,
    )
