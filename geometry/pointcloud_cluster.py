"""
点云部件划分：把 mesh 版 ``geometry/spectral_cluster.py`` 与 ``geometry/seed_split.py``
的几何分块算法迁移到「裸点云」（无三角面）上。纯点云**默认不考虑连通性**。

提供的分块后端：
1. feature_spectral（默认，迁移自 ``spectral_cluster``）——
   ``feature_spectral_cluster_points``：归一化坐标 + 局部 PCA 法向 + 曲率
   （surface variation，+ 可选凹凸/凸包指标、标量）→ kNN 高斯亲和图
   → 归一化图拉普拉斯 → 最小特征向量 → KMeans。两处「网格专属」环节被替换：
   面积加权法向 → PCA 法向；三角面 1-邻边 → kNN 图。其余算子
   （``normalized_graph_laplacian`` / ``_solve_smallest_eigs`` /
   ``estimate_cluster_count`` / ``_kmeans``）直接复用 mesh 版，保证同算法同参数。
2. geodesic_fps（可选，迁移自 ``seed_split``）——
   ``split_points_by_geodesic_fps``：kNN 图上测地最远点采样 + Dijkstra Voronoi
   + Lloyd 迭代，子块连通；复用 ``seed_split`` 里只依赖「图 + 点坐标」的通用算法。
3. lbo_spectral（可选）—— ``robust_laplacian`` 内蕴 LBO 谱嵌入（scipy eigsh 或
   传入 NEO 预算的 ``precomputed_eigvecs``）。
4. kmeans（兜底）—— 直接 3D 坐标 KMeans。

连通性（``split_cc_first`` / ``enforce_connectivity`` / ``split_points_by_connected_components``）
默认关闭，仅作可选项保留。下游标签单元从 ``face_id`` 改为 ``point_id``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.linalg import ArpackNoConvergence
except ImportError as exc:  # pragma: no cover
    raise ImportError("点云谱聚类需要 scipy: pip install scipy") from exc

# 复用 mesh 版谱聚类的核心算子，确保「同一套算法 + 同一套参数」迁移到点云：
#   - normalized_graph_laplacian / _solve_smallest_eigs：图拉普拉斯 + 最小特征对
#   - estimate_cluster_count：特征值间隙自动估簇数
#   - _kmeans：轻量 KMeans
# 点云相对 mesh 只替换两处「网格专属」环节：
#   ① compute_vertex_normals（面积加权）→ 局部 PCA 点法向
#   ② build_weighted_adjacency（三角面 1-邻边）→ kNN 图
from geometry.spectral_cluster import (
    DEFAULT_FEATURE_SIGMA,
    DEFAULT_MAX_CLUSTERS,
    DEFAULT_MIN_CLUSTERS,
    _kmeans,
    estimate_cluster_count,
    normalized_graph_laplacian,
    resolve_n_clusters,
    _solve_smallest_eigs,
)

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover
    raise ImportError("点云特征谱聚类需要 scipy.spatial: pip install scipy") from exc

# 点云特征默认权重（与 mesh build_vertex_features 的 scalar_weight 同风格）。
DEFAULT_NORMAL_WEIGHT = 1.0
DEFAULT_CURVATURE_WEIGHT = 0.6
DEFAULT_KNN = 16

# robust_laplacian 是纯 C++/pybind 包，无 torch 依赖，py3.14 亦有 wheel。
try:
    import robust_laplacian
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "点云 Laplacian 需要 robust_laplacian: pip install robust_laplacian"
    ) from exc


# 解析 dense 特征分解的规模上限（点数高于此值且 ARPACK 失败时直接放弃 dense 回退）。
_DENSE_EIGH_MAX_N = 4000


@dataclass
class PointCloudClusterResult:
    ok: bool
    work_dir: Path
    num_clusters: int = 0
    point_labels: List[int] = field(default_factory=list)
    message: str = ""
    eigenspace_backend: str = "scipy"
    panel_path: Optional[Path] = None
    overview_path: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 点云预处理 + 算子构造
# ---------------------------------------------------------------------------


def normalize_points(points: np.ndarray) -> np.ndarray:
    """中心化 + 缩放到单位球（与 NEO ``normalize_pc`` 对齐，但不做随机旋转）。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points 必须是 (N, 3)，得到 {pts.shape}")
    pts = pts - pts.mean(axis=0, keepdims=True)
    max_abs = float(np.max(np.abs(pts)))
    return pts / (max_abs + 1e-12)


def build_pointcloud_operators(
    points: np.ndarray,
    *,
    mollify_factor: float = 1e-5,
    n_neighbors: int = 30,
) -> Tuple[csr_matrix, csr_matrix, np.ndarray]:
    """
    用 robust_laplacian 从点坐标构造 (L, M)。

    返回 (L, M, mass)：
      L     —— (N, N) 半正定稀疏 Laplacian
      M     —— (N, N) 对角质量矩阵（稀疏）
      mass  —— (N,) M 的对角，即 per-point 面积权重
    """
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    L, M = robust_laplacian.point_cloud_laplacian(
        pts, mollify_factor=mollify_factor, n_neighbors=n_neighbors
    )
    L = csr_matrix(L).astype(np.float64)
    M = csr_matrix(M).astype(np.float64)
    mass = np.asarray(M.diagonal(), dtype=np.float64)
    return L, M, mass


# ---------------------------------------------------------------------------
# 特征空间（LBO 低频特征向量）—— scipy 后端
# ---------------------------------------------------------------------------


def solve_lbo_eigenspace_scipy(
    L: csr_matrix,
    M: csr_matrix,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    解广义特征问题 ``L u = λ M u``，取最小 k 个特征对（含近零的常数模）。

    优先 shift-invert（sigma≈0, which="LM"）求最小端；失败按
      ① which="SM" 直接求小端；
      ② 规模小则 dense 广义 eigh；
    顺序退化。返回 (eigvals[k], eigvecs[N, k])，按特征值升序。
    """
    n = L.shape[0]
    k = int(min(max(1, k), max(1, n - 1)))

    eigvals: Optional[np.ndarray] = None
    eigvecs: Optional[np.ndarray] = None

    # shift-invert 求最小端（数值上最稳，前提是 L - sigma*M 可分解）。
    try:
        eigvals, eigvecs = eigsh(L, k=k, M=M, sigma=1e-8, which="LM")
    except (ArpackNoConvergence, RuntimeError, ValueError):
        pass

    if eigvals is None:
        try:
            eigvals, eigvecs = eigsh(
                L, k=k, M=M, which="SM", tol=1e-4, maxiter=max(2000, n * 10)
            )
        except (ArpackNoConvergence, RuntimeError, ValueError):
            pass

    if eigvals is None:
        if n > _DENSE_EIGH_MAX_N:
            raise ArpackNoConvergence(
                f"point-cloud eigsh failed for n={n}, k={k}; dense fallback skipped",
                np.array([]),
                np.zeros((n, 0)),
            )
        from scipy.linalg import eigh as dense_eigh  # 局部导入

        L_d = L.toarray()
        M_d = M.toarray()
        L_d = 0.5 * (L_d + L_d.T)
        M_d = 0.5 * (M_d + M_d.T)
        w, v = dense_eigh(L_d, M_d)
        eigvals, eigvecs = w[:k], v[:, :k]

    order = np.argsort(np.real(eigvals))
    return np.real(eigvals)[order], np.real(eigvecs)[:, order]


# ---------------------------------------------------------------------------
# 特征谱聚类后端：把 mesh spectral_cluster 的「特征 + 亲和图」算法迁移到点云
#
# mesh 版流程（geometry/spectral_cluster.py）：
#   build_vertex_features  : 归一化坐标 + 面积加权法向 + 可选标量(畸变)
#   build_weighted_adjacency: 三角面 1-邻边的高斯亲和图（特征距离）
#   normalized_graph_laplacian → _solve_smallest_eigs → KMeans
#
# 点云没有面，故：法向用局部 PCA 估、邻接用 kNN 图。曲率（PCA surface variation）
# 与「凸包/凹凸指标」作为额外几何特征加入，使切分贴合几何部件边界。
# ---------------------------------------------------------------------------


def compute_point_normals_and_curvature(
    points: np.ndarray,
    n_neighbors: int = DEFAULT_KNN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    局部 PCA 估计每点法向与曲率（向量化）。

    对每个点取 kNN 邻域做协方差分解，特征值升序 λ0≤λ1≤λ2：
      - 法向     = λ0 对应特征向量（最小主方向 ≈ 表面法线）
      - 曲率     = λ0 / (λ0+λ1+λ2)  —— Pauly 的 surface variation，
                   平面→0、棱角/高曲率→大，对应 mesh 的「曲率指标」。
    返回 (normals[N,3], curvature[N])。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0)
    k = int(min(max(n_neighbors + 1, 3), n))
    tree = cKDTree(pts)
    _dist, idx = tree.query(pts, k=k)
    idx = np.atleast_2d(idx)
    neigh = pts[idx]                                    # (N, k, 3)
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(k)  # (N, 3, 3)
    evals, evecs = np.linalg.eigh(cov)                 # 升序：evals[:,0] 最小
    normals = evecs[:, :, 0]
    total = evals.sum(axis=1)
    curvature = np.where(total > 1e-12, evals[:, 0] / np.maximum(total, 1e-12), 0.0)
    return normals, curvature


def compute_point_convexity(
    points: np.ndarray,
    normals: np.ndarray,
    n_neighbors: int = DEFAULT_KNN,
) -> np.ndarray:
    """
    每点凹凸指标（对应 mesh 的「凸包/凹陷指标」），范围约 [-1, 1]：
      取邻域质心相对当前点的方向，与点法向做内积。
      > 0 凸（外凸），< 0 凹（凹槽），平面 ≈ 0。

    实现上等价于离散平均曲率符号：(centroid - p)·n，按邻域尺度归一。
    """
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros(0)
    k = int(min(max(n_neighbors + 1, 3), n))
    tree = cKDTree(pts)
    _dist, idx = tree.query(pts, k=k)
    idx = np.atleast_2d(idx)
    neigh = pts[idx]                                    # (N, k, 3)
    centroid = neigh.mean(axis=1)                       # (N, 3)
    offset = centroid - pts                             # 指向邻域质心
    scale = np.maximum(np.linalg.norm(offset, axis=1), 1e-12)
    signed = np.sum(offset * nrm, axis=1) / scale       # 归一到 [-1, 1]
    return signed


def build_point_features(
    points: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    *,
    point_scalars: Optional[np.ndarray] = None,
    convexity: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
    curvature_weight: float = DEFAULT_CURVATURE_WEIGHT,
    convexity_weight: float = 0.0,
    normal_weight: float = DEFAULT_NORMAL_WEIGHT,
) -> np.ndarray:
    """
    点云谱聚类特征（对齐 mesh build_vertex_features）：
        归一化坐标 + 法向(×normal_weight) + 曲率(×curvature_weight)
        [+ 凹凸(×convexity_weight)] [+ 可选标量(×scalar_weight)]
    返回 (N, D) float64。
    """
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    scale = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    pos = centered / scale
    chunks: List[np.ndarray] = [pos, np.asarray(normals, dtype=np.float64) * float(normal_weight)]

    c = np.asarray(curvature, dtype=np.float64).reshape(-1, 1)
    c = c / max(float(np.max(np.abs(c))), 1e-12)
    chunks.append(c * float(curvature_weight))

    if convexity is not None and convexity_weight > 0:
        cv = np.asarray(convexity, dtype=np.float64).reshape(-1, 1)
        cv = cv / max(float(np.max(np.abs(cv))), 1e-12)
        chunks.append(cv * float(convexity_weight))

    if point_scalars is not None and len(point_scalars) == len(pts):
        s = np.asarray(point_scalars, dtype=np.float64).reshape(-1, 1)
        s = s / max(float(np.max(np.abs(s))), 1e-12)
        chunks.append(s * float(scalar_weight))
    return np.hstack(chunks)


def build_knn_adjacency(
    features: np.ndarray,
    points: np.ndarray,
    *,
    n_neighbors: int = DEFAULT_KNN,
    sigma: float = DEFAULT_FEATURE_SIGMA,
) -> csr_matrix:
    """
    由空间 kNN 构造对称稀疏高斯亲和矩阵（对齐 mesh build_weighted_adjacency）。

    边集来自空间 kNN（替代三角面 1-邻边），权重用「特征距离」高斯核——
    几何相近且特征（法向/曲率/凹凸）一致的点权重大，部件边界处权重小。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= 1:
        return csr_matrix((n, n), dtype=np.float64)
    k = int(min(max(n_neighbors + 1, 2), n))
    tree = cKDTree(pts)
    _dist, idx = tree.query(pts, k=k)
    idx = np.atleast_2d(idx)

    a = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    b = idx.reshape(-1).astype(np.int64)
    valid = a != b
    a, b = a[valid], b[valid]

    diff = features[a] - features[b]
    dists = np.linalg.norm(diff, axis=1)

    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    dist_full = np.concatenate([dists, dists])

    if sigma <= 0:
        sigma = max(float(np.median(dist_full)), 1e-6)

    weights = np.exp(-0.5 * (dist_full / sigma) ** 2)
    adj = csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)
    adj = 0.5 * (adj + adj.T)
    adj.eliminate_zeros()
    return adj


def feature_spectral_cluster_points(
    points: np.ndarray,
    n_clusters: Optional[int] = None,
    *,
    point_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
    curvature_weight: float = DEFAULT_CURVATURE_WEIGHT,
    convexity_weight: float = 0.0,
    normal_weight: float = DEFAULT_NORMAL_WEIGHT,
    n_neighbors: int = DEFAULT_KNN,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    random_state: int = 0,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    点云「特征谱聚类」：mesh spectral_cluster_vertices 的点云迁移版。

    与 mesh 版同构：构造几何特征（法向 + 曲率 + 可选凹凸/标量）→ 高斯亲和图
    → 归一化图拉普拉斯 → 最小特征向量 → KMeans。差异仅在法向(PCA)与邻接(kNN)。
    返回 (point_labels, n_clusters, debug)。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("空点云")
    n_clusters = resolve_n_clusters(n_clusters)

    normals, curvature = compute_point_normals_and_curvature(pts, n_neighbors=n_neighbors)
    convexity = None
    if convexity_weight > 0:
        convexity = compute_point_convexity(pts, normals, n_neighbors=n_neighbors)

    features = build_point_features(
        pts,
        normals,
        curvature,
        point_scalars=point_scalars,
        convexity=convexity,
        scalar_weight=scalar_weight,
        curvature_weight=curvature_weight,
        convexity_weight=convexity_weight,
        normal_weight=normal_weight,
    )
    adjacency = build_knn_adjacency(features, pts, n_neighbors=n_neighbors)
    laplacian = normalized_graph_laplacian(adjacency)

    k_eigs = min(max(max_clusters + 1, min_clusters + 1), max(2, n - 1))
    eigenvalues, eigenvectors = _solve_smallest_eigs(laplacian, k_eigs)

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if n_clusters is None:
        n_clusters = estimate_cluster_count(
            eigenvalues, min_clusters=min_clusters, max_clusters=max_clusters
        )
    n_clusters = int(np.clip(n_clusters, 1, n))

    if n_clusters == 1:
        labels = np.zeros(n, dtype=np.int64)
    else:
        emb = eigenvectors[:, 1 : n_clusters + 1]
        if emb.shape[1] == 0:
            emb = eigenvectors[:, :1]
        labels = _kmeans(emb, n_clusters, random_state=random_state)

    debug: Dict[str, Any] = {
        "num_points": n,
        "backend": "feature_spectral",
        "requested_clusters": int(n_clusters),
        "feature_dim": int(features.shape[1]),
        "n_neighbors": int(n_neighbors),
        "scalar_weight": float(scalar_weight),
        "curvature_weight": float(curvature_weight),
        "convexity_weight": float(convexity_weight),
        "normal_weight": float(normal_weight),
        "eigenvalues": eigenvalues[: min(len(eigenvalues), max_clusters + 2)].tolist(),
    }
    return labels.astype(np.int64), n_clusters, debug


# ---------------------------------------------------------------------------
# 主聚类入口
# ---------------------------------------------------------------------------


def spectral_cluster_points(
    points: np.ndarray,
    n_clusters: Optional[int] = None,
    *,
    precomputed_eigvecs: Optional[np.ndarray] = None,
    precomputed_eigvals: Optional[np.ndarray] = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    random_state: int = 0,
    normalize: bool = True,
    mollify_factor: float = 1e-5,
    n_neighbors: int = 30,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    对点云做谱聚类，返回 (point_labels, n_clusters, debug)。

    特征空间来源：
      - 传入 ``precomputed_eigvecs`` (N, K) 时直接用它（NEO 后端的产物），
        ``backend="neo"``；
      - 否则用 robust_laplacian + scipy eigsh 现算，``backend="scipy"``。
    KMeans 在「跳过第 0 个常数模后的前 n_clusters 个特征向量」上做。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("空点云")
    n_clusters = resolve_n_clusters(n_clusters)

    backend = "scipy"
    eigvals: Optional[np.ndarray] = None
    if precomputed_eigvecs is not None:
        backend = "neo"
        eigvecs = np.asarray(precomputed_eigvecs, dtype=np.float64)
        if eigvecs.shape[0] != n:
            raise ValueError(
                f"precomputed_eigvecs 行数 {eigvecs.shape[0]} 与点数 {n} 不一致"
            )
        if precomputed_eigvals is not None:
            eigvals = np.asarray(precomputed_eigvals, dtype=np.float64)
    else:
        work_pts = normalize_points(pts) if normalize else pts
        L, M, _mass = build_pointcloud_operators(
            work_pts, mollify_factor=mollify_factor, n_neighbors=n_neighbors
        )
        k_eigs = min(max(max_clusters + 1, min_clusters + 1), max(2, n - 1))
        eigvals, eigvecs = solve_lbo_eigenspace_scipy(L, M, k_eigs)

    if n_clusters is None:
        if eigvals is not None and len(eigvals) >= 2:
            n_clusters = estimate_cluster_count(
                eigvals, min_clusters=min_clusters, max_clusters=max_clusters
            )
        else:
            n_clusters = min_clusters
    n_clusters = int(np.clip(n_clusters, 1, n))

    if n_clusters == 1:
        labels = np.zeros(n, dtype=np.int64)
    else:
        # 跳过第 0 个近零常数模，取后续 n_clusters 个作为嵌入。
        if eigvecs.shape[1] > 1:
            emb = eigvecs[:, 1 : n_clusters + 1]
            if emb.shape[1] == 0:
                emb = eigvecs[:, :1]
        else:
            emb = eigvecs[:, :1]
        labels = _kmeans(emb, n_clusters, random_state=random_state)

    debug: Dict[str, Any] = {
        "num_points": n,
        "backend": backend,
        "requested_clusters": int(n_clusters),
        "embedding_dim": int(eigvecs.shape[1]),
    }
    if eigvals is not None:
        debug["eigenvalues"] = eigvals[: min(len(eigvals), max_clusters + 2)].tolist()
    return labels.astype(np.int64), n_clusters, debug


# ---------------------------------------------------------------------------
# 连通性结构：kNN 图（点云版的"邻接图"，替代 mesh 的三角面顶点邻接）
#
# mesh 版（geometry/seed_split.py）用三角面 1-邻边构造顶点图，边权=欧氏边长，
# 在其上做「测地 FPS + Dijkstra Voronoi + Lloyd」——因为是在连通图上跑最短路，
# 每个 Voronoi 块天然连通。点云没有面，这里用 kNN 图当同样的连通性结构。
# ---------------------------------------------------------------------------


def build_point_knn_graph(
    points: np.ndarray,
    n_neighbors: int = DEFAULT_KNN,
) -> csr_matrix:
    """kNN 图（对称，边权=欧氏距离），作为点云的连通性结构。

    与 mesh ``_build_submesh_vertex_graph`` 对齐：边权取欧氏距离、双向对称、
    最小权 clamp 防零权路径退化。下游测地 FPS/Voronoi 直接在此图上跑。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= 1:
        return csr_matrix((n, n), dtype=np.float64)
    k = int(min(max(n_neighbors + 1, 2), n))
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=k)
    idx = np.atleast_2d(idx)
    dist = np.atleast_2d(dist)

    a = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    b = idx.reshape(-1).astype(np.int64)
    w = dist.reshape(-1).astype(np.float64)
    valid = a != b
    a, b, w = a[valid], b[valid], np.maximum(w[valid], 1e-9)

    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    data = np.concatenate([w, w])
    graph = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return graph


def enforce_per_label_point_connectivity(
    graph: csr_matrix,
    labels: np.ndarray,
    *,
    max_iter: int = 8,
) -> np.ndarray:
    """把点标签清洗成「每个 label 只对应一片（kNN 图上）连通的点」。

    迁移自 mesh ``enforce_per_label_face_connectivity``，邻接换成 kNN 图：
      ① 在「同 label 子图」上找连通分量；
      ② 每个 label 只保留最大那片；
      ③ 其余碎片按「跨 label 邻接最多」的邻居 label 重新分配；
      ④ 迭代至稳定。返回清洗后的 labels（不改输入）。
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    n = int(graph.shape[0])
    if n <= 1 or graph.nnz == 0:
        return labels
    indptr = graph.indptr
    indices = graph.indices.astype(np.int64)

    for _ in range(max_iter):
        row_idx = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        col_idx = indices
        same = labels[row_idx] == labels[col_idx]
        if np.any(same):
            sub = csr_matrix(
                (np.ones(int(same.sum()), dtype=np.int8),
                 (row_idx[same], col_idx[same])),
                shape=(n, n),
            )
            _ncc, cc = connected_components(sub, directed=False)
        else:
            cc = np.arange(n, dtype=np.int64)
        cc = cc.astype(np.int64)

        problematic: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for lab in np.unique(labels):
            mask = labels == lab
            if not mask.any():
                continue
            ids, sizes = np.unique(cc[mask], return_counts=True)
            if len(ids) > 1:
                problematic.append((int(lab), ids, sizes))
        if not problematic:
            break

        reassign = np.full(n, -1, dtype=np.int64)
        for _lab, cc_ids, cc_sizes in problematic:
            largest = int(cc_ids[int(np.argmax(cc_sizes))])
            for cc_id in cc_ids:
                cc_id = int(cc_id)
                if cc_id == largest:
                    continue
                frag = np.where(cc == cc_id)[0]
                if frag.size == 0:
                    continue
                neigh_labs: List[int] = []
                for fi in frag:
                    nbrs = indices[indptr[fi]:indptr[fi + 1]]
                    if nbrs.size == 0:
                        continue
                    out_mask = cc[nbrs] != cc_id
                    if np.any(out_mask):
                        neigh_labs.extend(labels[nbrs[out_mask]].tolist())
                if not neigh_labs:
                    continue
                vals, counts = np.unique(np.asarray(neigh_labs, dtype=np.int64),
                                         return_counts=True)
                reassign[frag] = int(vals[int(np.argmax(counts))])

        changed = reassign >= 0
        if not np.any(changed):
            break
        labels[changed] = reassign[changed]

    return labels


# ---------------------------------------------------------------------------
# 分裂
# ---------------------------------------------------------------------------


@dataclass
class PointSubSplit:
    sub_point_groups: List[List[int]]  # 每个子块的全局 point_id 列表
    sub_labels: np.ndarray             # (M,) 子标签，按 point_ids 顺序
    num_clusters: int
    debug: Dict[str, Any] = field(default_factory=dict)


def split_points_by_connected_components(
    points: np.ndarray,
    point_ids: Sequence[int],
    *,
    n_neighbors: int = DEFAULT_KNN,
    min_component_points: int = 1,
) -> Optional[List[List[int]]]:
    """按 kNN 图连通分量拆 block；返回每个 CC 的全局 point_id 列表。

    单 CC 时返回 None（让调用方继续走 FPS / 谱聚类）。迁移自 mesh
    ``split_block_by_connected_components``，把面邻接换成 kNN 图。
    """
    ids = np.asarray(list(point_ids), dtype=np.int64)
    if ids.size <= 1:
        return None
    sub_pts = np.asarray(points, dtype=np.float64)[ids]
    graph = build_point_knn_graph(sub_pts, n_neighbors=n_neighbors)
    n_cc, cc = connected_components(graph, directed=False)
    if n_cc <= 1:
        return None

    raw: List[List[int]] = []
    for k in range(n_cc):
        sel = np.flatnonzero(cc == k)
        if sel.size:
            raw.append(ids[sel].tolist())
    if len(raw) <= 1:
        return None

    raw.sort(key=len, reverse=True)
    if min_component_points > 1 and len(raw) > 1:
        big = raw[0]
        kept = [big]
        for g in raw[1:]:
            if len(g) < min_component_points:
                big.extend(g)
            else:
                kept.append(g)
        raw = kept
    if len(raw) <= 1:
        return None
    return raw


def split_points_by_geodesic_fps(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
    *,
    n_neighbors: int = DEFAULT_KNN,
    lloyd_iter: int = 8,
    random_state: int = 0,
) -> PointSubSplit:
    """点云版「测地 FPS + Voronoi + Lloyd」，保证每个子块连通。

    完整迁移自 mesh ``split_block_by_geodesic_fps``：
      ① kNN 图（边权=欧氏距离）作连通性结构；
      ② 测地最远点采样 (FPS) 选 seed_count 个种子；
      ③ Lloyd 迭代：Dijkstra Voronoi 划分 → 每簇取簇内 3D 重心最近点作新种子；
      ④ 逐 label 连通性清洗（修补边界孤立小片）。
    数值失败回退坐标 KMeans。
    """
    # 复用 mesh seed_split 里"只依赖图 + 点坐标"的通用算法（与网格版完全同算法）。
    from geometry.seed_split import (
        _geodesic_fps_seeds,
        _lloyd_relax_geodesic,
    )

    ids = np.asarray(list(point_ids), dtype=np.int64)
    seed_count = max(2, int(seed_count))
    if ids.size == 0:
        return PointSubSplit([], np.zeros(0, dtype=np.int64), 0)
    sub_pts = np.asarray(points, dtype=np.float64)[ids]
    if ids.size <= seed_count:
        return PointSubSplit(
            sub_point_groups=[ids.tolist()],
            sub_labels=np.zeros(ids.size, dtype=np.int64),
            num_clusters=1,
            debug={"reason": "子点集太小，跳过 FPS 分裂"},
        )

    strategy = "geodesic_fps_voronoi_lloyd"
    fps_error: Optional[str] = None
    try:
        graph = build_point_knn_graph(sub_pts, n_neighbors=n_neighbors)
        fps_seeds = _geodesic_fps_seeds(graph, sub_pts, seed_count)
        if len(fps_seeds) < 2:
            raise RuntimeError(f"FPS 仅取得 {len(fps_seeds)} 个种子")
        _seeds_final, labels = _lloyd_relax_geodesic(
            graph, sub_pts, list(fps_seeds), n_iter=lloyd_iter
        )
        labels = enforce_per_label_point_connectivity(graph, labels)
    except Exception as exc:  # noqa: BLE001 — 图不连通 / Dijkstra 退化等
        fps_error = str(exc)
        labels = _kmeans(sub_pts, seed_count, random_state=random_state)
        strategy = "coord_kmeans_fallback"

    out = _groups_from_labels(ids, labels, strategy=strategy)
    if fps_error:
        out.debug["fps_error"] = fps_error
    out.debug["n_neighbors"] = int(n_neighbors)
    return out


def split_points(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
    *,
    method: str = "feature_spectral",
    point_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
    curvature_weight: float = DEFAULT_CURVATURE_WEIGHT,
    convexity_weight: float = 0.0,
    normal_weight: float = DEFAULT_NORMAL_WEIGHT,
    n_neighbors: int = DEFAULT_KNN,
    lloyd_iter: int = 8,
    split_cc_first: bool = True,
    cc_min_points: int = 1,
    enforce_connectivity: bool = True,
    precomputed_eigvecs: Optional[np.ndarray] = None,
    random_state: int = 0,
) -> PointSubSplit:
    """
    把一个点云分块切成 seed_count 个子块。

    method:
      - ``"feature_spectral"``（默认）—— 迁移自 mesh ``spectral_cluster``：归一化坐标
        + PCA 法向 + 曲率（+ 可选凹凸/标量）→ kNN 高斯亲和图 → 归一化图拉普拉斯
        → 最小特征向量 → KMeans，切分贴合几何部件边界。
      - ``"geodesic_fps"`` —— 迁移自 mesh ``seed_split`` 的测地 FPS + Voronoi + Lloyd
        （kNN 图上），子块连通；纯点云一般不需要，作为可选项保留。
      - ``"lbo_spectral"`` —— robust_laplacian 内蕴 LBO 谱嵌入（可配合 NEO）。
      - ``"kmeans"`` —— 直接 3D 坐标 KMeans（最快，几何感知最弱）。

    连通性默认开启（对齐 mesh ``seed_split``）：
      - ``split_cc_first``：block 若含多片拓扑断开的点，先按 kNN 连通分量拆开；
      - ``enforce_connectivity``：KMeans 标签做逐 label 连通清洗，消除「同标签两片远岛」。
    任何高阶方法数值失败都会自动退化到坐标 KMeans。
    返回的 ``sub_point_groups`` / ``sub_labels`` 按 ``point_ids`` 顺序对齐。
    """
    ids = np.asarray(list(point_ids), dtype=np.int64)
    seed_count = max(2, int(seed_count))
    if ids.size == 0:
        return PointSubSplit([], np.zeros(0, dtype=np.int64), 0)

    all_pts = np.asarray(points, dtype=np.float64)
    sub_pts = all_pts[ids]
    if ids.size <= seed_count:
        return PointSubSplit(
            sub_point_groups=[ids.tolist()],
            sub_labels=np.zeros(ids.size, dtype=np.int64),
            num_clusters=1,
            debug={"reason": "子点集太小，跳过分裂"},
        )

    # CC 优先（可选）：block 若由多片拓扑断开的点组成，先免费按连通分量切。
    if split_cc_first:
        cc_groups = split_points_by_connected_components(
            all_pts, ids, n_neighbors=n_neighbors, min_component_points=cc_min_points
        )
        if cc_groups is not None and len(cc_groups) >= 2:
            lab = np.empty(ids.size, dtype=np.int64)
            pos = {int(pid): i for i, pid in enumerate(ids.tolist())}
            for gi, g in enumerate(cc_groups):
                for pid in g:
                    lab[pos[int(pid)]] = gi
            return PointSubSplit(
                sub_point_groups=[list(g) for g in cc_groups],
                sub_labels=lab,
                num_clusters=len(cc_groups),
                debug={"strategy": "connected_components",
                       "component_sizes": [len(g) for g in cc_groups]},
            )

    if method == "geodesic_fps":
        return split_points_by_geodesic_fps(
            all_pts, ids, seed_count,
            n_neighbors=n_neighbors, lloyd_iter=lloyd_iter, random_state=random_state,
        )

    sub_scalars = None
    if point_scalars is not None and len(point_scalars) == len(all_pts):
        sub_scalars = np.asarray(point_scalars, dtype=np.float64)[ids]

    if method == "feature_spectral":
        try:
            sub_labels_arr, _k, _dbg = feature_spectral_cluster_points(
                sub_pts,
                n_clusters=seed_count,
                point_scalars=sub_scalars,
                scalar_weight=scalar_weight,
                curvature_weight=curvature_weight,
                convexity_weight=convexity_weight,
                normal_weight=normal_weight,
                n_neighbors=n_neighbors,
                min_clusters=2,
                max_clusters=max(seed_count, 2),
                random_state=random_state,
            )
            return _finalize_kmeans_labels(
                ids, sub_pts, sub_labels_arr,
                strategy="feature_spectral",
                enforce_connectivity=enforce_connectivity,
                n_neighbors=n_neighbors,
            )
        except Exception:  # noqa: BLE001 — ARPACK/数值奇异等，退坐标 KMeans
            return _finalize_kmeans_labels(
                ids, sub_pts, _kmeans(sub_pts, seed_count, random_state=random_state),
                strategy="coord_kmeans_fallback",
                enforce_connectivity=enforce_connectivity,
                n_neighbors=n_neighbors,
            )

    if method == "lbo_spectral":
        emb: Optional[np.ndarray] = None
        if precomputed_eigvecs is not None:
            ev = np.asarray(precomputed_eigvecs, dtype=np.float64)
            if ev.shape[0] >= int(ids.max()) + 1 and ev.shape[1] > 1:
                emb = ev[ids][:, 1 : seed_count + 1]
        if emb is not None:
            labels = _kmeans(emb, seed_count, random_state=random_state)
            return _finalize_kmeans_labels(
                ids, sub_pts, labels,
                strategy="lbo_embedding_kmeans",
                enforce_connectivity=enforce_connectivity,
                n_neighbors=n_neighbors,
            )
        try:
            sub_labels_arr, _k, _dbg = spectral_cluster_points(
                sub_pts,
                n_clusters=seed_count,
                min_clusters=2,
                max_clusters=max(seed_count, 2),
                random_state=random_state,
            )
            return _finalize_kmeans_labels(
                ids, sub_pts, sub_labels_arr,
                strategy="subcloud_lbo_spectral",
                enforce_connectivity=enforce_connectivity,
                n_neighbors=n_neighbors,
            )
        except Exception:  # noqa: BLE001
            return _finalize_kmeans_labels(
                ids, sub_pts, _kmeans(sub_pts, seed_count, random_state=random_state),
                strategy="coord_kmeans_fallback",
                enforce_connectivity=enforce_connectivity,
                n_neighbors=n_neighbors,
            )

    # method == "kmeans" 或未知方法：坐标 KMeans。
    labels = _kmeans(sub_pts, seed_count, random_state=random_state)
    return _finalize_kmeans_labels(
        ids, sub_pts, labels,
        strategy="coord_kmeans",
        enforce_connectivity=enforce_connectivity,
        n_neighbors=n_neighbors,
    )


def split_points_by_kmeans(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
    *,
    precomputed_eigvecs: Optional[np.ndarray] = None,
    random_state: int = 0,
) -> PointSubSplit:
    """向后兼容包装：旧调用走 LBO 谱嵌入路径（见 ``split_points``）。"""
    return split_points(
        points,
        point_ids,
        seed_count,
        method="lbo_spectral",
        precomputed_eigvecs=precomputed_eigvecs,
        random_state=random_state,
    )


def _groups_from_labels(
    ids: np.ndarray, sub_labels: np.ndarray, *, strategy: str
) -> PointSubSplit:
    sub_labels = np.asarray(sub_labels, dtype=np.int64)
    used = sorted({int(v) for v in sub_labels})
    groups: List[List[int]] = []
    for k in used:
        idx = np.flatnonzero(sub_labels == k)
        if idx.size:
            groups.append(ids[idx].tolist())
    return PointSubSplit(
        sub_point_groups=groups,
        sub_labels=sub_labels,
        num_clusters=len(groups),
        debug={"strategy": strategy, "seed_count_actual": len(used)},
    )


def _finalize_kmeans_labels(
    ids: np.ndarray,
    sub_pts: np.ndarray,
    labels: np.ndarray,
    *,
    strategy: str,
    enforce_connectivity: bool,
    n_neighbors: int,
) -> PointSubSplit:
    """KMeans 标签后处理：可选逐 label 连通清洗，保证每个子块在 kNN 图上单连通。"""
    labels = np.asarray(labels, dtype=np.int64)
    if enforce_connectivity and len(sub_pts) > 1:
        graph = build_point_knn_graph(sub_pts, n_neighbors=n_neighbors)
        labels = enforce_per_label_point_connectivity(graph, labels)
        strategy = f"{strategy}_cc_clean"
    return _groups_from_labels(ids, labels, strategy=strategy)


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------


def export_point_cluster_labels_json(
    path: Path,
    point_labels: Sequence[int],
    num_clusters: int,
    *,
    backend: str = "scipy",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "method": "pointcloud_spectral_clustering",
        "eigenspace_backend": backend,
        "num_clusters": int(num_clusters),
        "num_points": len(point_labels),
        "point_labels": [int(x) for x in point_labels],
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
