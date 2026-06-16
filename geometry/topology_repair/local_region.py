"""Extract local sub-meshes around semantic boundaries for topology surgery."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from geometry.partition.common import build_face_adjacency_csr, extract_submesh


def semantic_boundary_faces(
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    part_a: int,
    part_b: int,
) -> List[int]:
    """Faces on the shared boundary between two semantic labels."""
    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    m = int(len(faces))
    boundary: List[int] = []
    adj = build_face_adjacency_csr(faces)
    indptr, indices = adj.indptr, adj.indices
    for fi in range(m):
        la = int(fpi[fi])
        if la not in (part_a, part_b):
            continue
        nbrs = indices[indptr[fi]: indptr[fi + 1]]
        for nb in nbrs:
            lb = int(fpi[int(nb)])
            if la != lb and {la, lb} == {part_a, part_b}:
                boundary.append(fi)
                break
    return boundary


def expand_face_rings(
    faces: np.ndarray,
    seed_faces: Sequence[int],
    *,
    rings: int = 2,
) -> Set[int]:
    """Expand seed faces by ``rings`` face-adjacency hops."""
    if rings <= 0 or not seed_faces:
        return set(int(f) for f in seed_faces)

    adj = build_face_adjacency_csr(faces)
    indptr, indices = adj.indptr, adj.indices
    region: Set[int] = set(int(f) for f in seed_faces)
    frontier = list(region)
    for _ in range(int(rings)):
        nxt: Set[int] = set()
        for fi in frontier:
            for nb in indices[indptr[fi]: indptr[fi + 1]]:
                nb_i = int(nb)
                if nb_i not in region:
                    nxt.add(nb_i)
        if not nxt:
            break
        region.update(nxt)
        frontier = list(nxt)
    return region


def faces_for_semantic(
    face_semantic_index: Sequence[int],
    part_idx: int,
) -> Set[int]:
    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    return set(int(i) for i in np.flatnonzero(fpi == int(part_idx)).tolist())


def extract_boundary_region(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    part_a: int,
    part_b: int,
    *,
    ring_rings: int = 2,
    include_core: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Build a local sub-mesh around the boundary of two semantics.

    Returns (sub_v, sub_f, global_face_ids, local_to_global_face).
    """
    boundary = semantic_boundary_faces(faces, face_semantic_index, part_a, part_b)
    region: Set[int] = set(boundary)
    region.update(expand_face_rings(faces, boundary, rings=ring_rings))
    if include_core:
        region.update(faces_for_semantic(face_semantic_index, part_a))
        region.update(faces_for_semantic(face_semantic_index, part_b))

    global_ids = sorted(region)
    if not global_ids:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64), {}

    sub_v, sub_f, _ = extract_submesh(vertices, faces, global_ids)
    local_to_global = {i: int(global_ids[i]) for i in range(len(global_ids))}
    global_face_arr = np.asarray(global_ids, dtype=np.int64)
    return sub_v, sub_f, global_face_arr, local_to_global


def boundary_face_seeds(
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    part_a: int,
    part_b: int,
    *,
    local_to_global: Dict[int, int],
) -> Tuple[List[int], List[int]]:
    """Map global boundary faces to local indices for each semantic."""
    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    global_to_local = {g: l for l, g in local_to_global.items()}
    seeds_a: List[int] = []
    seeds_b: List[int] = []
    for gfi in semantic_boundary_faces(faces, face_semantic_index, part_a, part_b):
        if gfi not in global_to_local:
            continue
        lfi = global_to_local[gfi]
        if int(fpi[gfi]) == part_a:
            seeds_a.append(lfi)
        else:
            seeds_b.append(lfi)
    return seeds_a, seeds_b
