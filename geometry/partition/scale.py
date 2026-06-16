"""Scale-aware graph budgets and vertex pooling for large sub-meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp

from .common import build_submesh_vertex_graph

DEFAULT_GRAPH_VERTEX_BUDGET = 12_000
_MIN_GRAPH_VERTEX_BUDGET = 6_000


@dataclass
class VertexGraphView:
    """Full mesh vertices mapped to a (possibly pooled) graph for geodesic ops."""

    graph: sp.csr_matrix
    coords: np.ndarray
    fine_to_graph: np.ndarray
    graph_to_fine: np.ndarray
    pooled: bool = False

    @property
    def num_graph(self) -> int:
        return int(self.graph.shape[0])

    @property
    def num_fine(self) -> int:
        return int(len(self.fine_to_graph))

    def labels_graph_to_fine(self, labels_graph: np.ndarray) -> np.ndarray:
        lg = np.asarray(labels_graph, dtype=np.int64).reshape(-1)
        return lg[self.fine_to_graph]


def resolve_graph_vertex_budget(
    n_vertices: int,
    n_faces: int,
    seed_count: int,
) -> int:
    """Target graph size for Dijkstra / Laplacian ops (full mesh if small enough)."""
    _ = n_faces, seed_count
    n = int(n_vertices)
    if n <= DEFAULT_GRAPH_VERTEX_BUDGET:
        return n
    if n <= 50_000:
        return DEFAULT_GRAPH_VERTEX_BUDGET
    if n <= 150_000:
        return 10_000
    return _MIN_GRAPH_VERTEX_BUDGET


def effective_graph_vertex_budget(
    n_vertices: int,
    graph_vertex_budget: Optional[int] = None,
) -> int:
    """None 或 ≤0 表示不池化，使用全分辨率顶点图。"""
    n = int(n_vertices)
    if graph_vertex_budget is None or int(graph_vertex_budget) <= 0:
        return n
    return int(graph_vertex_budget)


def scaled_lloyd_iter(block_faces: int, default: int = 4) -> int:
    faces = int(block_faces)
    if faces >= 200_000:
        return min(default, 2)
    if faces >= 80_000:
        return min(default, 3)
    return default


def scaled_contraction_iters(n_vertices: int, default: int = 5) -> int:
    n = int(n_vertices)
    if n <= DEFAULT_GRAPH_VERTEX_BUDGET:
        return default
    if n <= 80_000:
        return min(default, 4)
    return min(default, 3)


def scaled_num_rays(n_faces: int, default: int = 8) -> int:
    faces = int(n_faces)
    if faces >= 100_000:
        return min(default, 4)
    if faces >= 50_000:
        return min(default, 6)
    return default


def resolve_partition_scale(
    n_vertices: int,
    n_faces: int,
    seed_count: int,
    *,
    strategy: str = "",
) -> Dict[str, Any]:
    """Unified scale hints for partition strategies."""
    _ = strategy
    return {
        "graph_vertex_budget": resolve_graph_vertex_budget(
            n_vertices, n_faces, seed_count
        ),
        "lloyd_iter": scaled_lloyd_iter(n_faces),
        "contraction_iters": scaled_contraction_iters(n_vertices),
        "num_rays": scaled_num_rays(n_faces),
    }


def build_vertex_graph_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_vertices: int = DEFAULT_GRAPH_VERTEX_BUDGET,
) -> VertexGraphView:
    """Build vertex graph; pool spatially when ``len(vertices) > max_vertices``."""
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    n = len(verts)
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return VertexGraphView(
            graph=sp.csr_matrix((0, 0)),
            coords=verts,
            fine_to_graph=empty,
            graph_to_fine=empty,
            pooled=False,
        )

    budget = max(int(max_vertices), 256)
    if n <= budget:
        idx = np.arange(n, dtype=np.int64)
        return VertexGraphView(
            graph=build_submesh_vertex_graph(verts, tris),
            coords=verts,
            fine_to_graph=idx,
            graph_to_fine=idx,
            pooled=False,
        )

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    span = np.maximum(vmax - vmin, 1e-9)

    n_axis = max(2, int(round(budget ** (1.0 / 3.0))))
    fine_to_graph: np.ndarray
    graph_to_fine: np.ndarray
    pool_coords: np.ndarray

    while True:
        cell_size = span / float(n_axis)
        cell_ids = np.floor((verts - vmin) / cell_size).astype(np.int64)
        cell_ids = np.clip(cell_ids, 0, n_axis - 1)
        linear = (
            cell_ids[:, 0] * n_axis * n_axis
            + cell_ids[:, 1] * n_axis
            + cell_ids[:, 2]
        )
        unique_cells, fine_to_graph = np.unique(linear, return_inverse=True)
        n_pool = len(unique_cells)
        if n_pool <= budget or n_axis <= 2:
            break
        n_axis = max(2, (n_axis * 2) // 3)

    graph_to_fine = np.zeros(n_pool, dtype=np.int64)
    pool_coords = np.zeros((n_pool, 3), dtype=np.float64)
    cell_size = span / float(n_axis)

    for p in range(n_pool):
        members = np.flatnonzero(fine_to_graph == p)
        cell_linear = int(unique_cells[p])
        ci = cell_linear // (n_axis * n_axis)
        rem = cell_linear % (n_axis * n_axis)
        cj = rem // n_axis
        ck = rem % n_axis
        cell_center = vmin + (np.array([ci, cj, ck], dtype=np.float64) + 0.5) * cell_size
        if len(members) == 1:
            rep = int(members[0])
        else:
            d = np.linalg.norm(verts[members] - cell_center, axis=1)
            rep = int(members[int(np.argmin(d))])
        graph_to_fine[p] = rep
        pool_coords[p] = verts[rep]

    pool_a = fine_to_graph[tris[:, 0]]
    pool_b = fine_to_graph[tris[:, 1]]
    pool_c = fine_to_graph[tris[:, 2]]
    pa = np.concatenate([pool_a, pool_b, pool_c])
    pb = np.concatenate([pool_b, pool_c, pool_a])
    diff = pool_coords[pa] - pool_coords[pb]
    weights = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
    graph = sp.csr_matrix(
        (np.concatenate([weights, weights]), (np.concatenate([pa, pb]), np.concatenate([pb, pa]))),
        shape=(n_pool, n_pool),
    )

    return VertexGraphView(
        graph=graph,
        coords=pool_coords,
        fine_to_graph=fine_to_graph.astype(np.int64),
        graph_to_fine=graph_to_fine,
        pooled=True,
    )
