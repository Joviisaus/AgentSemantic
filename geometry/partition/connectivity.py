"""Ensure K-way partitions yield K connected face groups with smooth boundaries."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.spectral_cluster import vertex_labels_to_face_labels

from .common import (
    SubBlockSplit,
    build_face_adjacency_csr,
    enforce_per_label_face_connectivity,
    extract_submesh,
)
from .geodesic_fps import geodesic_fps_seeds, lloyd_relax_geodesic_view
from .scale import build_vertex_graph_view, effective_graph_vertex_budget

LOGGER = logging.getLogger("geometry.partition.connectivity")


def cluster_medoid_vertex_seeds(
    vertices: np.ndarray,
    vertex_labels: np.ndarray,
    target_k: int,
) -> List[int]:
    """Each cluster → medoid vertex (fine mesh index)."""
    vlabels = np.asarray(vertex_labels, dtype=np.int64)
    seeds: List[int] = []
    for lab in range(int(target_k)):
        mask = vlabels == lab
        if not np.any(mask):
            continue
        idx = np.flatnonzero(mask)
        if len(idx) == 1:
            seeds.append(int(idx[0]))
            continue
        pts = vertices[idx]
        dsum = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2).sum(axis=1)
        seeds.append(int(idx[int(np.argmax(dsum))]))
    return seeds


def smooth_face_label_boundaries(
    faces: np.ndarray,
    face_labels: np.ndarray,
    *,
    passes: int = 2,
    neighbor_ratio: float = 0.65,
) -> np.ndarray:
    """边界平滑：去掉沿分块边缘的「棋盘格」锯齿/孤立面。

    判据只看**邻面**（不含自身）标签：
      - 邻面全部一致且与自身不同 → 必翻（去孤立棋盘格点）；
      - 邻面里某个异标签占比 ≥ neighbor_ratio → 翻到该标签（拉直锯齿边界）。
    多遍迭代直到收敛（或用尽 passes）。仅在边界处微调，块内部不受影响。
    """
    labels = np.asarray(face_labels, dtype=np.int64).copy()
    m = int(len(faces))
    if m <= 1 or passes <= 0:
        return labels

    adj = build_face_adjacency_csr(faces)
    indptr, indices = adj.indptr, adj.indices

    # 就地（异步 Gauss-Seidel）更新：同步更新会让棋盘格整体翻转而来回震荡、无法收敛，
    # 就地更新能让边界锯齿/孤立面逐步并入邻块并稳定下来。
    for _ in range(int(passes)):
        changed = False
        for fi in range(m):
            nbrs = indices[indptr[fi]: indptr[fi + 1]]
            if nbrs.size == 0:
                continue
            nbr_labels = labels[nbrs].astype(np.int64)
            counts = np.bincount(nbr_labels)
            majority = int(np.argmax(counts))
            maj_count = int(counts[majority])
            if majority == int(labels[fi]):
                continue
            # 邻面一致异标签 → 必翻；否则需异标签邻面占比达阈值
            if maj_count == int(nbrs.size) or maj_count >= neighbor_ratio * float(
                nbrs.size
            ):
                labels[fi] = majority
                changed = True
        if not changed:
            break
    return labels


def _split_largest_face_group(
    vertices: np.ndarray,
    faces: np.ndarray,
    group_face_idx: Sequence[int],
) -> Optional[Tuple[List[int], List[int]]]:
    """把最大子块再切成两个连通块（测地 FPS k=2）。"""
    idx = list(group_face_idx)
    if len(idx) < 2:
        return None
    sub_v, sub_f, _ = extract_submesh(vertices, faces, idx)
    if len(sub_f) < 2:
        return None
    budget = effective_graph_vertex_budget(len(sub_v), None)
    view = build_vertex_graph_view(sub_v, sub_f, max_vertices=budget)
    seeds = geodesic_fps_seeds(view.graph, view.coords, 2)
    if len(seeds) < 2:
        return None
    _, vlabels = lloyd_relax_geodesic_view(view, sub_v, seeds, n_iter=2)
    fl = np.asarray(vertex_labels_to_face_labels(vlabels, sub_f), dtype=np.int64)
    fl = enforce_per_label_face_connectivity(sub_f, fl)
    g0 = [idx[i] for i in np.flatnonzero(fl == 0).tolist()]
    g1 = [idx[i] for i in np.flatnonzero(fl == 1).tolist()]
    if not g0 or not g1:
        return None
    return g0, g1


def connected_face_groups_from_labels(
    faces: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    把面标签拆成连通组：同一 label 若含多个连通分量，拆成多个 group。

    保证 ``groups`` 中每一项都是面邻接图上的一个连通分量。
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    labels = np.asarray(labels, dtype=np.int64).copy()
    labels = smooth_face_label_boundaries(faces, labels, passes=2)
    labels = enforce_per_label_face_connectivity(faces, labels)
    m = int(len(faces))
    if m == 0:
        return labels, []

    adj = build_face_adjacency_csr(faces)
    indptr, indices = adj.indptr, adj.indices
    groups: List[List[int]] = []

    for lab in sorted(set(int(x) for x in labels)):
        idx = np.flatnonzero(labels == lab).astype(np.int64)
        if idx.size == 0:
            continue
        if idx.size == 1:
            groups.append([int(idx[0])])
            continue

        local_map = {int(g): li for li, g in enumerate(idx)}
        rows: List[int] = []
        cols: List[int] = []
        for gi in idx:
            gi = int(gi)
            for nb in indices[indptr[gi] : indptr[gi + 1]]:
                nb = int(nb)
                if nb in local_map:
                    rows.append(local_map[gi])
                    cols.append(local_map[nb])
        if not rows:
            groups.extend([[int(i)] for i in idx])
            continue

        sub = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(len(idx), len(idx)),
        )
        _n_cc, cc_id = connected_components(sub, directed=False)
        for cc in range(int(_n_cc)):
            cc_faces = idx[cc_id == cc].tolist()
            if cc_faces:
                groups.append([int(fi) for fi in cc_faces])

    new_labels = np.zeros(m, dtype=np.int64)
    for gid, g in enumerate(groups):
        for fi in g:
            new_labels[fi] = gid
    return new_labels, groups


def _merge_smallest_group_into_neighbor(
    faces: np.ndarray,
    groups: List[List[int]],
    labels: np.ndarray,
) -> bool:
    """合并最小碎块到边界最多的邻块。"""
    if len(groups) <= 1:
        return False
    sizes = [len(g) for g in groups]
    si = int(np.argmin(sizes))
    if sizes[si] >= sizes[int(np.argmax(sizes))]:
        return False

    adj = build_face_adjacency_csr(faces)
    indptr, indices = adj.indptr, adj.indices
    group_set = {fi: gi for gi, g in enumerate(groups) for fi in g}
    neighbor_votes: Dict[int, int] = {}

    for fi in groups[si]:
        for nb in indices[indptr[fi]: indptr[fi + 1]]:
            gj = group_set.get(int(nb))
            if gj is not None and gj != si:
                neighbor_votes[gj] = neighbor_votes.get(gj, 0) + 1

    if not neighbor_votes:
        return False

    target = max(neighbor_votes, key=neighbor_votes.get)
    target_lab = int(labels[groups[target][0]])
    for fi in groups[si]:
        labels[fi] = target_lab
    groups[target].extend(groups[si])
    del groups[si]
    return True


def build_k_connected_face_groups(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids_list: Sequence[int],
    face_labels: np.ndarray,
    target_k: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """保证输出恰好 target_k 个非空连通 face 组（必要时合并/再切）。"""
    labels = np.asarray(face_labels, dtype=np.int64).copy()
    m = len(faces)
    target_k = max(2, int(target_k))

    labels = enforce_per_label_face_connectivity(faces, labels)
    labels = smooth_face_label_boundaries(faces, labels, passes=2)

    def _groups_from_labels() -> List[List[int]]:
        out: List[List[int]] = []
        for lab in sorted(set(int(x) for x in labels)):
            idx = np.flatnonzero(labels == lab)
            if idx.size:
                out.append(idx.tolist())
        return out

    groups = _groups_from_labels()
    split_guard = 0
    merge_guard = 0

    while len(groups) < target_k and split_guard < target_k * 2:
        split_guard += 1
        if not groups:
            break
        largest_i = int(np.argmax([len(g) for g in groups]))
        split = _split_largest_face_group(
            vertices, faces, groups[largest_i]
        )
        if split is None:
            break
        g0, g1 = split
        max_lab = int(labels.max()) if labels.size else -1
        for fi in g0:
            labels[fi] = max_lab + 1
        for fi in g1:
            labels[fi] = max_lab + 2
        groups = _groups_from_labels()

    while len(groups) > target_k and merge_guard < len(groups) * 4:
        merge_guard += 1
        if not _merge_smallest_group_into_neighbor(faces, groups, labels):
            break
        groups = _groups_from_labels()

    # 重编号 0..k-1
    groups = _groups_from_labels()
    remap: Dict[int, int] = {}
    for new_id, g in enumerate(groups):
        lab = int(labels[g[0]])
        remap[lab] = new_id
    for fi in range(m):
        lab = int(labels[fi])
        if lab in remap:
            labels[fi] = remap[lab]

    groups = [
        [face_ids_list[i] for i in np.flatnonzero(labels == k).tolist()]
        for k in range(len(groups))
    ]
    groups = [g for g in groups if g]
    return labels, groups


def refine_vertex_labels_to_k_connected(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
    target_k: int,
    *,
    lloyd_iter: int = 4,
    graph_vertex_budget: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[int]], Dict[str, Any]]:
    """
    用 K 个 medoid 种子做测地 Voronoi，得到 K 个连通子块；再平滑边界并校正组数。
    """
    sub_v = np.asarray(vertices, dtype=np.float64)
    sub_f = np.asarray(faces, dtype=np.int64)
    target_k = max(2, int(target_k))
    budget = effective_graph_vertex_budget(len(sub_v), graph_vertex_budget)
    view = build_vertex_graph_view(sub_v, sub_f, max_vertices=budget)

    seeds_fine = cluster_medoid_vertex_seeds(sub_v, vertex_labels, target_k)
    seeds_graph = [int(view.fine_to_graph[s]) for s in seeds_fine]

    if len(seeds_graph) < target_k:
        for s in geodesic_fps_seeds(view.graph, view.coords, target_k):
            if int(s) not in seeds_graph:
                seeds_graph.append(int(s))
            if len(seeds_graph) >= target_k:
                break
    seeds_graph = seeds_graph[:target_k]

    seeds_final, vlabels_fine = lloyd_relax_geodesic_view(
        view, sub_v, seeds_graph, n_iter=lloyd_iter
    )

    face_labels = np.asarray(
        vertex_labels_to_face_labels(vlabels_fine, sub_f), dtype=np.int64
    )
    face_labels = enforce_per_label_face_connectivity(sub_f, face_labels)
    face_labels = smooth_face_label_boundaries(sub_f, face_labels, passes=2)

    # local face index groups → global face_ids handled by caller
    local_groups: List[List[int]] = []
    for lab in sorted(set(int(x) for x in face_labels)):
        idx = np.flatnonzero(face_labels == lab)
        if idx.size:
            local_groups.append(idx.tolist())

    debug: Dict[str, Any] = {
        "refine": "geodesic_voronoi_from_medoids",
        "target_k": target_k,
        "seeds_graph": seeds_graph,
        "seeds_final_graph": seeds_final,
        "pooled_graph": view.pooled,
        "graph_vertices": view.num_graph,
        "groups_before_fix": len(local_groups),
    }
    return face_labels, local_groups, debug


_MIN_FRAGMENT_FACE_RATIO = 0.02


def finalize_spectral_face_split(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids_list: Sequence[int],
    face_labels: np.ndarray,
    target_k: int,
) -> SubBlockSplit:
    """
    谱聚类面标签后处理：连通性校正、合并过小碎屑与多余簇，不强制补切。

    测地 Voronoi 已保证主区域连通；此处把嵌入产生的纯碎屑并回邻块。
    """
    face_ids_list = list(face_ids_list)
    sub_v, sub_f, _ = extract_submesh(vertices, faces, face_ids_list)
    labels = np.asarray(face_labels, dtype=np.int64).copy()
    m = len(sub_f)
    target_k = max(2, int(target_k))

    labels = enforce_per_label_face_connectivity(sub_f, labels)

    def _groups_from_labels() -> List[List[int]]:
        out: List[List[int]] = []
        for lab in sorted(set(int(x) for x in labels)):
            idx = np.flatnonzero(labels == lab)
            if idx.size:
                out.append(idx.tolist())
        return out

    groups = _groups_from_labels()
    min_faces = max(3, int(m * _MIN_FRAGMENT_FACE_RATIO))
    tiny_guard = 0
    while tiny_guard < len(groups) * 4:
        tiny_guard += 1
        groups = _groups_from_labels()
        if len(groups) <= 1:
            break
        sizes = [len(g) for g in groups]
        if min(sizes) >= min_faces:
            break
        if not _merge_smallest_group_into_neighbor(sub_f, groups, labels):
            break

    groups = _groups_from_labels()
    merge_guard = 0
    while len(groups) > target_k and merge_guard < len(groups) * 4:
        merge_guard += 1
        if not _merge_smallest_group_into_neighbor(sub_f, groups, labels):
            break
        groups = _groups_from_labels()

    groups = _groups_from_labels()
    remap: Dict[int, int] = {}
    for new_id, g in enumerate(groups):
        lab = int(labels[g[0]])
        remap[lab] = new_id
    for fi in range(m):
        lab = int(labels[fi])
        if lab in remap:
            labels[fi] = remap[lab]

    global_groups = [
        [face_ids_list[i] for i in np.flatnonzero(labels == k).tolist()]
        for k in range(len(groups))
    ]
    global_groups = [g for g in global_groups if g]

    return SubBlockSplit(
        sub_face_groups=global_groups,
        sub_labels=labels,
        num_clusters=len(global_groups),
        debug={
            "refine": "spectral_face_connectivity_only",
            "target_k": target_k,
            "min_fragment_faces": min_faces,
            "groups_after_fix": len(global_groups),
        },
    )


def finalize_submesh_k_connected_split(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids_list: Sequence[int],
    vertex_labels: np.ndarray,
    target_k: int,
    *,
    lloyd_iter: int = 4,
    graph_vertex_budget: Optional[int] = None,
) -> SubBlockSplit:
    """谱聚类/KMeans 等顶点标签 → K 连通 face 组（统一出口）。"""
    face_ids_list = list(face_ids_list)
    sub_v, sub_f, _ = extract_submesh(vertices, faces, face_ids_list)

    face_labels, local_groups, dbg = refine_vertex_labels_to_k_connected(
        sub_v,
        sub_f,
        vertex_labels,
        target_k,
        lloyd_iter=lloyd_iter,
        graph_vertex_budget=graph_vertex_budget,
    )

    face_labels, global_groups = build_k_connected_face_groups(
        sub_v,
        sub_f,
        face_ids_list,
        face_labels,
        target_k,
    )

    dbg["groups_after_fix"] = len(global_groups)
    sub_labels = face_labels

    return SubBlockSplit(
        sub_face_groups=global_groups,
        sub_labels=sub_labels,
        num_clusters=len(global_groups),
        debug=dbg,
    )
