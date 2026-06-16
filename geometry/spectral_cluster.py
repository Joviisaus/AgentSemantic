"""
Mesh 测地聚类：3D 网格 → 测地坐标系（2D）→ K-means 定种子 → 测地 Voronoi 分块。

流程：
  1. FPS 选 2 个坐标地标，每点嵌入 (d1, d2) = 到地标的测地距离
  2. 在 2D 测地坐标里 K-means，质心吸附回网格点 → k 个聚类种子
  3. 在测地图上对 k 个种子做 Dijkstra Voronoi 分配（沿网格边，保证单连通）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix


DEFAULT_MIN_CLUSTERS = 3
EMBED_DIM = 2
# 以凹度为主导：实测中沿凹槽/接缝/关节走界的切分质量最好，因此默认让 concavity_weight
# 显著高于测地与曲率项。测地项保留少量权重以避免凹腔种子被饿成碎簇、把平滑区域淹成巨块。
# 凹陷/接缝特别明显时可由上层（VLM/CLI）再上调 concavity_weight 到 0.75~0.85。
DEFAULT_GEODESIC_WEIGHT = 0.20
DEFAULT_CURVATURE_WEIGHT = 0.15
DEFAULT_CONCAVITY_WEIGHT = 0.70


def resolve_spectral_energy_weights(
    geodesic_weight: Optional[float] = None,
    curvature_weight: Optional[float] = None,
    concavity_weight: Optional[float] = None,
) -> Tuple[float, float, float]:
    """归一化谱聚类三项能量权重（测地 / 曲率 / 凹度）。"""
    raw = [
        float(geodesic_weight)
        if geodesic_weight is not None
        else DEFAULT_GEODESIC_WEIGHT,
        float(curvature_weight)
        if curvature_weight is not None
        else DEFAULT_CURVATURE_WEIGHT,
        float(concavity_weight)
        if concavity_weight is not None
        else DEFAULT_CONCAVITY_WEIGHT,
    ]
    raw = [max(0.0, x) for x in raw]
    total = float(sum(raw))
    if total <= 1e-12:
        return (
            DEFAULT_GEODESIC_WEIGHT,
            DEFAULT_CURVATURE_WEIGHT,
            DEFAULT_CONCAVITY_WEIGHT,
        )
    return (raw[0] / total, raw[1] / total, raw[2] / total)


def resolve_n_clusters(manual: Optional[int]) -> Optional[int]:
    if manual is None:
        return None
    k = int(manual)
    if k < 1:
        raise ValueError("手动簇数必须 >= 1")
    return k


@dataclass
class SpectralClusterResult:
    ok: bool
    work_dir: Path
    num_clusters: int = 0
    vertex_labels: List[int] = field(default_factory=list)
    face_labels: List[int] = field(default_factory=list)
    message: str = ""
    panel_path: Optional[Path] = None
    overview_path: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _kmeans(
    data: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
    max_iter: int = 100,
    init_centers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """轻量 KMeans（仅依赖 numpy）。"""
    n = len(data)
    if n_clusters <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int64)
    n_clusters = min(n_clusters, n)

    rng = np.random.default_rng(random_state)
    if init_centers is not None and init_centers.shape == (n_clusters, data.shape[1]):
        centers = np.asarray(init_centers, dtype=data.dtype).copy()
    else:
        init_idx = rng.choice(n, size=n_clusters, replace=False)
        centers = data[init_idx].copy()

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = data[mask].mean(axis=0)
            else:
                centers[k] = data[rng.integers(0, n)]
    return labels


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """面积加权顶点法向。"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(fn_norm, 1e-12)

    vn = np.zeros_like(vertices, dtype=np.float64)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    vn_norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(vn_norm, 1e-12)


def build_vertex_geodesic_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> csr_matrix:
    """顶点邻接测地图，边权 = 共享边欧氏长度（测地近似）。"""
    from geometry.partition.common import build_submesh_vertex_graph

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    return build_submesh_vertex_graph(verts, tris)


def build_face_geodesic_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    geodesic_weight: Optional[float] = None,
    curvature_weight: Optional[float] = None,
    concavity_weight: Optional[float] = None,
) -> csr_matrix:
    """
    面邻接测地图，边权 = 三项能量加权和：

        E_geo  = 归一化测地步长
        E_curv = (1 - cosθ) 归一化，刻画一般弯曲/折角
        E_conc = 凹边(minima rule) 上的 (1 - cosθ)，刻画凹槽/接缝

        w(i,j) = wg·E_geo + wc·E_curv + wn·E_conc
    """
    from geometry.partition.common import build_face_adjacency_csr, face_centroids_and_normals

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    m = int(len(tris))
    if m == 0:
        return csr_matrix((0, 0), dtype=np.float64)

    centroids, normals = face_centroids_and_normals(verts, tris)
    topo = build_face_adjacency_csr(tris).tocoo()
    upper = topo.row < topo.col
    r = topo.row[upper].astype(np.int64)
    c = topo.col[upper].astype(np.int64)
    if r.size == 0:
        return csr_matrix((m, m), dtype=np.float64)

    dc = centroids[c] - centroids[r]
    geo = np.maximum(np.linalg.norm(dc, axis=1), 1e-12)
    cos_ang = np.clip(np.sum(normals[r] * normals[c], axis=1), -1.0, 1.0)

    wg, wc, wn = resolve_spectral_energy_weights(
        geodesic_weight, curvature_weight, concavity_weight
    )

    # 凹边判据：两面质心相互位于对方法向正侧 → 表面向内折（minima rule）
    concave = (
        np.sum(normals[r] * dc, axis=1) + np.sum(normals[c] * (-dc), axis=1)
    ) > 0.0
    curv_energy = 1.0 - cos_ang
    conc_energy = np.where(concave, curv_energy, 0.0)

    geo_n = geo / max(float(np.mean(geo)), 1e-12)
    curv_n = curv_energy / max(float(np.mean(curv_energy)), 1e-12)
    if np.any(concave):
        conc_mean = float(np.mean(conc_energy[concave]))
        conc_n = conc_energy / max(conc_mean, 1e-12)
    else:
        conc_n = conc_energy

    dist = np.maximum(wg * geo_n + wc * curv_n + wn * conc_n, 1e-9)

    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    data = np.concatenate([dist, dist])
    return csr_matrix((data, (rows, cols)), shape=(m, m), dtype=np.float64)


def _normalize_distance_features(feats: np.ndarray) -> np.ndarray:
    """按列归一化测地距离特征到 [0, 1]。"""
    feats = np.asarray(feats, dtype=np.float64)
    out = feats.copy()
    for c in range(out.shape[1]):
        col = out[:, c]
        finite = np.isfinite(col)
        if not np.any(finite):
            out[:, c] = 0.0
            continue
        scale = float(np.max(col[finite]))
        scale = max(scale, 1e-12)
        out[:, c] = np.where(finite, col / scale, 1.0)
    return out


def geodesic_embed_2d(
    graph: csr_matrix,
    coords: np.ndarray,
    *,
    embed_dim: int = EMBED_DIM,
    seed_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    投影到测地坐标系：FPS 选 ``embed_dim`` 个地标，
    每点坐标为到各地标的 Dijkstra 测地距离。
    """
    from scipy.sparse.csgraph import dijkstra

    from geometry.partition.geodesic_fps import geodesic_fps_seeds

    n = int(graph.shape[0])
    d = min(max(1, int(embed_dim)), max(1, n))
    if n <= 1:
        return np.zeros((n, d), dtype=np.float64), []

    if seed_indices is not None and len(seed_indices) >= d:
        seeds = [int(s) for s in seed_indices[:d]]
    else:
        seeds = geodesic_fps_seeds(graph, np.asarray(coords, dtype=np.float64), d)

    if not seeds:
        return np.zeros((n, d), dtype=np.float64), []

    dists = dijkstra(
        graph,
        indices=np.asarray(seeds, dtype=np.int64),
        directed=False,
    )
    dists = np.asarray(dists, dtype=np.float64)
    if dists.ndim == 1:
        dists = dists.reshape(1, -1)
    dists = dists.T
    dists = np.where(np.isfinite(dists), dists, np.inf)

    if dists.shape[1] < d:
        pad = np.full((n, d - dists.shape[1]), np.inf, dtype=np.float64)
        dists = np.hstack([dists, pad])

    feats = _normalize_distance_features(dists[:, :d])
    return feats, seeds


def _farthest_point_init(data: np.ndarray, k: int) -> np.ndarray:
    """确定性最远点初始化（不依赖 RNG，提升对称稳定性）。"""
    n = len(data)
    center = data.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(data - center, axis=1)))
    chosen = [first]
    min_d = np.linalg.norm(data - data[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(data - data[nxt], axis=1))
    return data[np.asarray(chosen, dtype=np.int64)].copy()


def _kmeans_centers(
    data: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
) -> np.ndarray:
    """K-means 质心（确定性最远点初始化，仅用于在测地坐标系里定位种子）。"""
    _ = random_state
    n = len(data)
    k = min(max(1, int(n_clusters)), n)
    if k <= 1 or n <= 1:
        return data[:1].copy()

    centers = _farthest_point_init(data, k)
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(100):
        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                centers[ci] = data[mask].mean(axis=0)
    return centers


def _cluster_seeds_from_embed_kmeans(
    embed_2d: np.ndarray,
    graph: csr_matrix,
    coords: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
) -> List[int]:
    """在测地 2D 坐标里 K-means 找质心，再吸附到最近网格点作为 Voronoi 种子。"""
    from geometry.partition.geodesic_fps import geodesic_fps_seeds

    n = int(len(embed_2d))
    k = min(max(1, int(n_clusters)), n)
    centers = _kmeans_centers(embed_2d, k, random_state=random_state)

    seeds: List[int] = []
    used: set[int] = set()
    for ci in range(k):
        dist = np.linalg.norm(embed_2d - centers[ci], axis=1)
        for idx in np.argsort(dist):
            pick = int(idx)
            if pick not in used:
                seeds.append(pick)
                used.add(pick)
                break

    if len(seeds) < k:
        for s in geodesic_fps_seeds(graph, np.asarray(coords, dtype=np.float64), k):
            si = int(s)
            if si not in used:
                seeds.append(si)
                used.add(si)
            if len(seeds) >= k:
                break
    return seeds[:k]


def _potts_regularize_labels(
    graph: csr_matrix,
    seed_dists: np.ndarray,
    labels: np.ndarray,
    *,
    smooth_beta: float = 1.0,
    max_iter: int = 12,
) -> np.ndarray:
    """对图 Voronoi 标签做 Potts 能量正则（ICM），消除等距带上的锯齿/棋盘格。

    纯 ``argmin`` 的离散测地 Voronoi 在两种子近等距的带状区域里，会因图距离
    呈台阶状而让分界面来回锯齿（棋盘格）。这里在**数据项 = 到种子的测地距离**
    的基础上加一个 **Potts 平滑项 = 相邻面异标签数 × β**，用 ICM 就地迭代最小化

        E(f, l) = dist[l, f] + β · #{邻面标签 ≠ l}

    分界面会被拉直到「测地距离 + 边界长度」的能量最优位置：仅在数据项几乎打平
    的锯齿处翻转，块内部/真实凹缝边界不动。β 按图边权中位数自适应定标。
    """
    from collections import deque

    n = int(graph.shape[0])
    if n <= 1 or labels.size == 0:
        return labels
    k = int(seed_dists.shape[0])
    if k <= 1:
        return labels

    indptr = graph.indptr
    indices = graph.indices
    data = graph.data
    if data.size == 0:
        return labels
    # 平滑项尺度 = β × 边权中位数：相邻翻转代价 ~ 数步测地距离
    med_w = float(np.median(data[data > 0])) if np.any(data > 0) else 1.0
    beta = float(smooth_beta) * med_w
    if beta <= 0.0:
        return labels

    big = float(np.max(seed_dists[np.isfinite(seed_dists)]) + 1.0) if np.any(
        np.isfinite(seed_dists)
    ) else 1.0
    cost = np.where(np.isfinite(seed_dists), seed_dists, big)
    labels = labels.astype(np.int64).copy()

    def _neighbors(fi: int) -> np.ndarray:
        return indices[indptr[fi]: indptr[fi + 1]]

    def _best_label(fi: int) -> int:
        nbrs = _neighbors(fi)
        if nbrs.size == 0:
            return int(labels[fi])
        nbr_labels = labels[nbrs]
        cur = int(labels[fi])
        cand = set(int(x) for x in nbr_labels.tolist())
        cand.add(cur)
        if len(cand) == 1:
            return cur
        best_lab = cur
        best_cost = cost[cur, fi] + beta * float(np.count_nonzero(nbr_labels != cur))
        for lab in cand:
            if lab == cur:
                continue
            c = cost[lab, fi] + beta * float(np.count_nonzero(nbr_labels != lab))
            if c < best_cost - 1e-9:
                best_cost = c
                best_lab = lab
        return best_lab

    # 仅处理「边界面」（存在异标签邻居），用活动队列推进；ICM 每次接受的翻转都
    # 严格降低总能量，故必然收敛，只在分界面附近做局部调整。
    in_queue = np.zeros(n, dtype=bool)
    queue: "deque[int]" = deque()
    for fi in range(n):
        nbrs = _neighbors(fi)
        if nbrs.size and np.any(labels[nbrs] != labels[fi]):
            queue.append(fi)
            in_queue[fi] = True

    flips_cap = max_iter * max(n, 1)
    flips = 0
    while queue and flips < flips_cap:
        fi = queue.popleft()
        in_queue[fi] = False
        new_lab = _best_label(fi)
        if new_lab == int(labels[fi]):
            continue
        labels[fi] = new_lab
        flips += 1
        for nb in _neighbors(fi):
            nb = int(nb)
            if not in_queue[nb]:
                queue.append(nb)
                in_queue[nb] = True
    return labels


def _cluster_from_geodesic_graph(
    graph: csr_matrix,
    coords: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
    embed_dim: int = EMBED_DIM,
    face_array: Optional[np.ndarray] = None,
    smooth_beta: float = 1.0,
    lloyd_iter: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    测地 FPS 定种子 → Lloyd 松弛均衡 → 测地 Voronoi + Potts 正则 最终分块。

    种子用**测地最远点采样（FPS）**铺开，保证在网格上均匀、对称分布——
    旧版「2D 嵌入 + K-means」在对称结构（如两条腿）上会把多点坍缩到同一坐标，
    导致种子在一端扎堆、另一端缺失，分出来一边完美、另一边一块从腿连到半个身子。
    FPS 之后再做 Lloyd 松弛把种子推到各自区域质心、均衡块大小，最后 Potts 正则
    拉直分界面、消除棋盘格锯齿。
    """
    from scipy.sparse.csgraph import dijkstra

    from geometry.partition.geodesic_fps import (
        geodesic_fps_seeds,
        lloyd_relax_geodesic,
    )

    _ = embed_dim, random_state
    n = int(graph.shape[0])
    k = int(np.clip(n_clusters, 1, n))
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int64), {}

    coords = np.asarray(coords, dtype=np.float64)
    seeds = [int(s) for s in geodesic_fps_seeds(graph, coords, k)]
    if not seeds:
        return np.zeros(n, dtype=np.int64), {}

    if int(lloyd_iter) > 0:
        seeds, _ = lloyd_relax_geodesic(
            graph, coords, list(seeds), n_iter=int(lloyd_iter)
        )

    seed_dists = np.asarray(
        dijkstra(
            graph,
            indices=np.asarray(seeds, dtype=np.int64),
            directed=False,
        ),
        dtype=np.float64,
    )
    if seed_dists.ndim == 1:
        seed_dists = seed_dists.reshape(1, -1)
    labels = np.argmin(seed_dists, axis=0).astype(np.int64)
    labels = _potts_regularize_labels(
        graph, seed_dists, labels, smooth_beta=smooth_beta
    )

    if face_array is not None and len(face_array) > 0:
        from geometry.partition.common import enforce_per_label_face_connectivity

        labels = enforce_per_label_face_connectivity(
            np.asarray(face_array, dtype=np.int64),
            labels,
        )

    return labels, {
        "cluster_seed_indices": [int(s) for s in seeds],
        "assignment": "geodesic_fps_lloyd_voronoi_potts",
        "lloyd_iter": int(lloyd_iter),
        "potts_beta": float(smooth_beta),
    }


def spectral_cluster_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_clusters: int,
    *,
    geodesic_weight: Optional[float] = None,
    curvature_weight: Optional[float] = None,
    concavity_weight: Optional[float] = None,
    random_state: int = 0,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """面级聚类：测地 2D 嵌入定种子 → 测地 Voronoi 分面。"""
    from geometry.partition.common import face_centroids_and_normals

    tris = np.asarray(faces, dtype=np.int64)
    m = int(len(tris))
    if m == 0:
        raise ValueError("空 mesh")

    k = int(np.clip(n_clusters, 1, m))
    if k == 1:
        return np.zeros(m, dtype=np.int64), 1, {
            "method": "face_geodesic_trivial",
            "num_faces": m,
        }

    verts = np.asarray(vertices, dtype=np.float64)
    wg, wc, wn = resolve_spectral_energy_weights(
        geodesic_weight, curvature_weight, concavity_weight
    )
    graph = build_face_geodesic_graph(
        verts,
        tris,
        geodesic_weight=wg,
        curvature_weight=wc,
        concavity_weight=wn,
    )
    centroids, _ = face_centroids_and_normals(verts, tris)
    labels, extra = _cluster_from_geodesic_graph(
        graph,
        centroids,
        k,
        random_state=random_state,
        face_array=tris,
    )

    return labels, k, {
        "method": "face_geodesic_fps_lloyd_voronoi",
        "num_faces": m,
        "requested_clusters": k,
        "geodesic_weight": wg,
        "curvature_weight": wc,
        "concavity_weight": wn,
        **extra,
    }


def spectral_cluster_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_clusters: Optional[int] = None,
    *,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    random_state: int = 0,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """顶点级聚类：测地 2D 嵌入定种子 → 测地 Voronoi 分顶点。"""
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    n = len(verts)
    if n == 0:
        raise ValueError("空 mesh")

    resolved = resolve_n_clusters(n_clusters)
    k = int(np.clip(resolved if resolved is not None else min_clusters, 1, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64), 1, {
            "method": "vertex_geodesic_trivial",
            "num_vertices": n,
        }

    graph = build_vertex_geodesic_graph(verts, tris)
    labels, extra = _cluster_from_geodesic_graph(
        graph, verts, k, random_state=random_state
    )
    from geometry.partition.common import enforce_per_label_face_connectivity

    face_labels = np.asarray(vertex_labels_to_face_labels(labels, tris), dtype=np.int64)
    face_labels = enforce_per_label_face_connectivity(tris, face_labels)
    labels = face_labels_to_vertex_labels(tris, face_labels, n)

    return labels.astype(np.int64), k, {
        "num_vertices": n,
        "requested_clusters": k,
        "method": "vertex_geodesic_2d_voronoi",
        "auto_clusters": resolved is None,
        **extra,
    }


def face_labels_to_vertex_labels(
    faces: np.ndarray,
    face_labels: Sequence[int],
    n_vertices: int,
) -> np.ndarray:
    """面标签多数投票回传至顶点。"""
    f = np.asarray(faces, dtype=np.int64)
    fl = np.asarray(face_labels, dtype=np.int64)
    if f.size == 0:
        return np.zeros(n_vertices, dtype=np.int64)

    max_lab = int(fl.max()) if fl.size else 0
    votes = np.zeros((n_vertices, max_lab + 1), dtype=np.int64)
    for vi in range(3):
        np.add.at(votes, (f[:, vi], fl), 1)
    has_votes = votes.sum(axis=1) > 0
    out = np.zeros(n_vertices, dtype=np.int64)
    out[has_votes] = np.argmax(votes[has_votes], axis=1).astype(np.int64)
    return out


def vertex_labels_to_face_labels(vertex_labels: Sequence[int], faces: np.ndarray) -> List[int]:
    """三角面片标签 = 三个顶点标签的众数。"""
    vlabels = np.asarray(vertex_labels, dtype=np.int64)
    out: List[int] = []
    for tri in faces:
        votes = vlabels[tri]
        counts = np.bincount(votes, minlength=int(votes.max()) + 1)
        out.append(int(np.argmax(counts)))
    return out


def export_cluster_labels_json(
    path: Path,
    face_labels: Sequence[int],
    num_clusters: int,
    num_vertices: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "method": "geodesic_coordinate_clustering",
        "num_clusters": num_clusters,
        "num_vertices": num_vertices,
        "num_faces": len(face_labels),
        "face_labels": [int(x) for x in face_labels],
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def render_spectral_cluster_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    output_dir: Path,
    *,
    render_anim_name: str = "",
    render_frame_index: int = -1,
    render_time: float = 0.0,
    render_medoid_score: float = 0.0,
    num_clusters: int = 0,
    image_size: int = 640,
    cols: int = 4,
) -> Tuple[Path, Path]:
    """渲染 12 视角拼图与单视角概览图。"""
    from rendering.heatmap_renderer import (
        DEFAULT_OVERVIEW_VIEW,
        render_segmentation_panel,
        render_segmentation_view,
    )

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("渲染需要 Pillow: pip install pillow") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (image_size, image_size)
    elev, azim = DEFAULT_OVERVIEW_VIEW

    title = (
        f"[测地聚类] {render_anim_name or 'medoid'}  "
        f"frame={render_frame_index}  t={render_time:.3f}s  "
        f"medoid={render_medoid_score:.4f}  parts={num_clusters}"
    )
    panel_path = render_segmentation_panel(
        vertices,
        faces,
        face_labels,
        out_dir / "medoid_spectral_clustering.png",
        title=title,
        image_size=size,
        cols=cols,
    )

    overview_img = render_segmentation_view(
        vertices, faces, face_labels, elev, azim, image_size=size
    )
    overview_path = out_dir / "overview.png"
    Image.fromarray(overview_img).save(overview_path)
    return panel_path, overview_path


def run_spectral_clustering_on_render_pose(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_dir: str | Path,
    *,
    n_clusters: Optional[int] = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    render: bool = True,
    image_size: int = 640,
    cols: int = 4,
    render_anim_name: str = "",
    render_frame_index: int = -1,
    render_time: float = 0.0,
    render_medoid_score: float = 0.0,
) -> SpectralClusterResult:
    """在 medoid 姿态上执行测地坐标聚类并导出标签 JSON。"""
    work_dir = Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    result = SpectralClusterResult(ok=False, work_dir=work_dir)
    requested_clusters = resolve_n_clusters(n_clusters)

    try:
        verts = np.asarray(vertices, dtype=np.float64)
        tris = np.asarray(faces, dtype=np.int64)
        if requested_clusters is not None:
            face_label_arr, num_clusters, debug = spectral_cluster_faces(
                verts,
                tris,
                requested_clusters,
            )
            vertex_labels = face_labels_to_vertex_labels(tris, face_label_arr, len(verts))
            face_labels = [int(x) for x in face_label_arr]
        else:
            vertex_labels, num_clusters, debug = spectral_cluster_vertices(
                verts,
                tris,
                n_clusters=None,
                min_clusters=min_clusters,
            )
            face_labels = vertex_labels_to_face_labels(vertex_labels, tris)
    except (ValueError, ImportError, OSError) as exc:
        result.message = f"测地聚类失败: {exc}"
        return result

    labels_path = export_cluster_labels_json(
        work_dir / "face_cluster_labels.json",
        face_labels,
        num_clusters,
        len(vertex_labels),
        extra={
            "requested_clusters": requested_clusters,
            "auto_clusters": requested_clusters is None,
            "debug": debug,
        },
    )
    result.vertex_labels = [int(x) for x in vertex_labels]
    result.face_labels = face_labels
    result.num_clusters = num_clusters
    result.extra["labels_json"] = str(labels_path)
    result.extra["requested_clusters"] = requested_clusters
    result.extra["debug"] = debug

    if render:
        try:
            panel_path, overview_path = render_spectral_cluster_views(
                vertices,
                faces,
                face_labels,
                work_dir,
                render_anim_name=render_anim_name,
                render_frame_index=render_frame_index,
                render_time=render_time,
                render_medoid_score=render_medoid_score,
                num_clusters=num_clusters,
                image_size=image_size,
                cols=cols,
            )
            result.panel_path = panel_path
            result.overview_path = overview_path
            result.extra["panel_png"] = str(panel_path)
            result.extra["overview_png"] = str(overview_path)
        except (ImportError, OSError, ValueError) as exc:
            result.message = f"测地聚类完成但渲染失败: {exc}"
            result.ok = True
            return result

    result.ok = True
    auto_note = "" if requested_clusters is not None else f"（默认 k={min_clusters}）"
    result.message = (
        f"测地聚类完成: {num_clusters} 个部件{auto_note}, "
        f"标签 {labels_path.name}"
    )
    return result
