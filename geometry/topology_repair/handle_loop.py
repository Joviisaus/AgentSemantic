"""Detect and cut pseudo-bridge handle loops (genus-increasing connections)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from geometry.partition.common import build_face_adjacency_csr, extract_submesh

from .concave_seam import face_edge_concavity_cost, find_concave_seam_between_seeds
from .local_region import expand_face_rings, semantic_boundary_faces


def mesh_euler_characteristic(vertices: np.ndarray, faces: np.ndarray) -> int:
    """χ = V - E + F for a closed or open triangle mesh."""
    f = int(len(faces))
    v = int(len(vertices))
    edges: Set[Tuple[int, int]] = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, vv in ((a, b), (b, c), (c, a)):
            e = (min(u, vv), max(u, vv))
            edges.add(e)
    e = len(edges)
    return v - e + f


def estimate_genus(vertices: np.ndarray, faces: np.ndarray, *, boundary_loops: int = 1) -> float:
    """
    Rough genus estimate for a mesh with ``boundary_loops`` boundary components.

    g = (2 - χ - b) / 2
    """
    chi = mesh_euler_characteristic(vertices, faces)
    return (2 - chi - int(boundary_loops)) / 2.0


def bridge_faces_in_region(
    faces: np.ndarray,
    region_faces: Sequence[int],
) -> List[int]:
    """
    Articulation faces in the region: removing them increases connected components.
    """
    region = sorted(set(int(f) for f in region_faces))
    if len(region) <= 2:
        return []

    global_to_local = {g: i for i, g in enumerate(region)}
    sub_f = faces[np.asarray(region, dtype=np.int64)]
    adj = build_face_adjacency_csr(sub_f)
    m = int(len(sub_f))
    articulation: List[int] = []

    n_cc, _ = connected_components(adj, directed=False)
    if n_cc <= 1:
        pass

    for li in range(m):
        # Remove face li and test connectivity
        mask = np.ones(m, dtype=bool)
        mask[li] = False
        if not np.any(mask):
            continue
        sub_adj = adj[mask][:, mask]
        n_sub, _ = connected_components(sub_adj, directed=False)
        if n_sub > 1:
            articulation.append(region[li])
    return articulation


def find_handle_loop_cycle(
    vertices: np.ndarray,
    faces: np.ndarray,
    region_faces: Sequence[int],
    seed_faces_a: Sequence[int],
    seed_faces_b: Sequence[int],
) -> Tuple[List[int], Dict[str, object]]:
    """
    Locate a handle loop cycle around a pseudo-bridge between two lobes.

    Uses two shortest paths that meet in the bridge, then forms a cycle.
    Returns global face ids on the loop.
    """
    region = sorted(set(int(f) for f in region_faces))
    debug: Dict[str, object] = {}
    if len(region) < 4:
        return [], {"reason": "region too small"}

    global_to_local = {g: i for i, g in enumerate(region)}
    sub_v, sub_f, _ = extract_submesh(vertices, faces, region)
    local_a = [global_to_local[int(f)] for f in seed_faces_a if int(f) in global_to_local]
    local_b = [global_to_local[int(f)] for f in seed_faces_b if int(f) in global_to_local]
    if not local_a or not local_b:
        return [], {"reason": "seeds outside region"}

    seam, seam_dbg = find_concave_seam_between_seeds(sub_v, sub_f, local_a, local_b)
    debug.update(seam_dbg)
    if not seam:
        return [], debug

    # Widen seam into a cut band (1-ring)
    seam_set = set(seam)
    adj = build_face_adjacency_csr(sub_f)
    indptr, indices = adj.indptr, adj.indices
    band = set(seam_set)
    for fi in seam:
        for nb in indices[indptr[fi]: indptr[fi + 1]]:
            band.add(int(nb))

    loop_global = [region[int(fi)] for fi in sorted(band)]
    debug["loop_faces"] = len(loop_global)
    debug["estimated_genus"] = estimate_genus(sub_v, sub_f)
    return loop_global, debug


def cut_handle_bridge(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    part_a: int,
    part_b: int,
    *,
    ring_rings: int = 2,
) -> Tuple[Dict[int, int], Dict[str, object]]:
    """
    Cut a pseudo-bridge between two semantics (e.g. two legs welded together).

    Returns global face_id → new semantic index overrides for the bridge band.
    """
    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    boundary = semantic_boundary_faces(faces, fpi, part_a, part_b)
    region = expand_face_rings(faces, boundary, rings=ring_rings)
    region.update(int(f) for f in np.flatnonzero(fpi == part_a))
    region.update(int(f) for f in np.flatnonzero(fpi == part_b))

    seeds_a = [int(f) for f in np.flatnonzero(fpi == part_a) if int(f) in region]
    seeds_b = [int(f) for f in np.flatnonzero(fpi == part_b) if int(f) in region]
    loop, debug = find_handle_loop_cycle(
        vertices, faces, sorted(region), seeds_a[: min(8, len(seeds_a))], seeds_b[: min(8, len(seeds_b))]
    )
    if not loop:
        return {}, debug

    # Reassign bridge band: alternate toward nearest semantic by geodesic
    sub_region = sorted(set(loop))
    global_to_local = {g: i for i, g in enumerate(sub_region)}
    overrides: Dict[int, int] = {}
    for gfi in loop:
        nbrs_same = []
        adj_full = build_face_adjacency_csr(faces)
        indptr, indices = adj_full.indptr, adj_full.indices
        for nb in indices[indptr[gfi]: indptr[gfi + 1]]:
            nb_i = int(nb)
            if int(fpi[nb_i]) in (part_a, part_b) and nb_i not in loop:
                nbrs_same.append(int(fpi[nb_i]))
        if nbrs_same:
            vals, counts = np.unique(np.asarray(nbrs_same), return_counts=True)
            overrides[gfi] = int(vals[int(np.argmax(counts))])
    debug["overrides"] = len(overrides)
    return overrides, debug
