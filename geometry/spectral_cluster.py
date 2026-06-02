"""
Mesh 谱聚类分割：基于网格邻接图与几何/畸变特征，对 medoid 姿态 mesh 做部件划分。

供 mesh_distortion_analyzer.py 调用；渲染结果写入 output/<模型>/spectral_clustering/。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.sparse import csr_matrix, diags, eye
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.linalg import ArpackNoConvergence
except ImportError as exc:
    raise ImportError("谱聚类需要 scipy: pip install scipy") from exc


DEFAULT_MIN_CLUSTERS = 3
DEFAULT_MAX_CLUSTERS = 24
DEFAULT_FEATURE_SIGMA = 0.0  # 0 = 用边特征距离中位数自动估计


def resolve_n_clusters(manual: Optional[int]) -> Optional[int]:
    """
    解析手动簇数。
    - None: 自动估计
    - int >= 1: 手动指定
    """
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
    """轻量 KMeans（仅依赖 numpy）。
    init_centers: 可选，shape (n_clusters, D)，跳过随机初始化。"""
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


def build_vertex_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
) -> np.ndarray:
    """
    构造谱聚类特征：归一化坐标 + 法向 + 可选标量（如畸变）。
    返回 (V, D) float64。
    """
    center = vertices.mean(axis=0)
    centered = vertices - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    scale = max(scale, 1e-6)
    pos = centered / scale
    normals = compute_vertex_normals(vertices, faces)

    chunks = [pos, normals]
    if vertex_scalars is not None and len(vertex_scalars) == len(vertices):
        s = np.asarray(vertex_scalars, dtype=np.float64).reshape(-1, 1)
        s = s / max(float(np.max(np.abs(s))), 1e-12)
        chunks.append(s * float(scalar_weight))
    return np.hstack(chunks)


def build_weighted_adjacency(
    faces: np.ndarray,
    features: np.ndarray,
    sigma: float = DEFAULT_FEATURE_SIGMA,
) -> csr_matrix:
    """由三角面片 1-邻接边构造对称稀疏亲和矩阵（向量化版本）。"""
    n = len(features)
    f = np.asarray(faces, dtype=np.int64)
    if f.size == 0:
        return csr_matrix((n, n), dtype=np.float64)

    # 三个有向边：(0,1),(1,2),(2,0) —— 后面再加反向边对称化
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    valid = a != b  # 退化边
    a, b = a[valid], b[valid]

    diff = features[a] - features[b]
    dists = np.linalg.norm(diff, axis=1)

    # 加上反向，得到对称连接
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    dist_full = np.concatenate([dists, dists])

    if sigma <= 0:
        sigma = float(np.median(dist_full))
        sigma = max(sigma, 1e-6)

    weights = np.exp(-0.5 * (dist_full / sigma) ** 2)
    adj = csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)
    # csr_matrix 在重复 (i,j) 上会累加，这里再做一次 sym 平均确保对称
    adj = 0.5 * (adj + adj.T)
    adj.eliminate_zeros()
    return adj


def normalized_graph_laplacian(adjacency: csr_matrix) -> csr_matrix:
    n = adjacency.shape[0]
    degree = np.asarray(adjacency.sum(axis=1)).flatten()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    d_mat = diags(inv_sqrt, format="csr")
    return eye(n, format="csr", dtype=np.float64) - d_mat @ adjacency @ d_mat


def estimate_cluster_count(
    eigenvalues: np.ndarray,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
) -> int:
    """根据特征值间隙估计聚类数（跳过第一个近零特征值）。"""
    if len(eigenvalues) <= 2:
        return min_clusters
    vals = np.sort(np.real(eigenvalues))
    search = vals[1 : min(len(vals), max_clusters + 2)]
    if len(search) < 2:
        return min_clusters
    gaps = np.diff(search)
    k = int(np.argmax(gaps)) + 1
    return int(np.clip(k, min_clusters, max_clusters))


_DENSE_EIGH_MAX_N = 3000


def _solve_smallest_eigs(
    laplacian, k_eigs: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    取 normalized Laplacian 的 k 个最小特征对。

    优先用 ARPACK eigsh(which="SM")；若不收敛则按以下顺序退化：
      1. ARPACK 放宽 tol / 增加 maxiter；
      2. 矩阵规模小 (n <= _DENSE_EIGH_MAX_N) 时直接 dense numpy.linalg.eigh；
      3. 仍然失败则抛 ArpackNoConvergence，让上层走坐标 KMeans 退化路径。

    注意：刻意不再尝试 shift-invert (sigma=0)，因为它需要 LU 分解，
    在退化拓扑（孤立顶点 / 重复点）上更容易直接段错误，比单纯不收敛更糟。
    """
    n = laplacian.shape[0]
    k_eigs = min(k_eigs, max(1, n - 1))

    try:
        return eigsh(
            laplacian,
            k=k_eigs,
            which="SM",
            tol=1e-4,
            maxiter=max(1000, n * 5),
        )
    except ArpackNoConvergence:
        pass

    try:
        return eigsh(
            laplacian,
            k=k_eigs,
            which="SM",
            tol=1e-2,
            maxiter=max(5000, n * 20),
        )
    except ArpackNoConvergence:
        pass

    if n <= _DENSE_EIGH_MAX_N:
        dense = (
            laplacian.toarray()
            if hasattr(laplacian, "toarray")
            else np.asarray(laplacian)
        )
        dense = 0.5 * (dense + dense.T)
        eigvals, eigvecs = np.linalg.eigh(dense)
        return eigvals[:k_eigs], eigvecs[:, :k_eigs]

    raise ArpackNoConvergence(
        f"eigsh failed for n={n}, k={k_eigs}; dense fallback skipped (n>{_DENSE_EIGH_MAX_N})",
        np.array([]),
        np.zeros((n, 0)),
    )


def spectral_cluster_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_clusters: Optional[int] = None,
    *,
    vertex_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    random_state: int = 0,
    init_vertex_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    对网格顶点做谱聚类。
    返回 vertex_labels, n_clusters, debug_info。
    """
    n = len(vertices)
    if n == 0:
        raise ValueError("空 mesh")
    n_clusters = resolve_n_clusters(n_clusters)

    features = build_vertex_features(vertices, faces, vertex_scalars, scalar_weight)
    adjacency = build_weighted_adjacency(faces, features)
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
        init_centers = None
        if init_vertex_indices is not None:
            idx = np.asarray(list(init_vertex_indices), dtype=np.int64)
            if idx.size == n_clusters and idx.min() >= 0 and idx.max() < n:
                init_centers = emb[idx].copy()
        labels = _kmeans(
            emb, n_clusters, random_state=random_state, init_centers=init_centers
        )

    debug = {
        "num_vertices": n,
        "requested_clusters": n_clusters,
        "eigenvalues": eigenvalues[: min(len(eigenvalues), max_clusters + 2)].tolist(),
        "feature_dim": int(features.shape[1]),
        "scalar_weight": float(scalar_weight),
    }
    return labels.astype(np.int64), n_clusters, debug


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
        "method": "spectral_clustering",
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
        f"[谱聚类] {render_anim_name or 'medoid'}  "
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
    vertex_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    render: bool = True,
    image_size: int = 640,
    cols: int = 4,
    render_anim_name: str = "",
    render_frame_index: int = -1,
    render_time: float = 0.0,
    render_medoid_score: float = 0.0,
) -> SpectralClusterResult:
    """
    在 medoid 输出姿态上执行谱聚类，导出标签 JSON，并可选渲染到 output_dir。

    n_clusters:
        None — 按特征值间隙自动估计簇数
        int >= 1 — 手动指定簇数
    """
    work_dir = Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    result = SpectralClusterResult(ok=False, work_dir=work_dir)
    requested_clusters = resolve_n_clusters(n_clusters)

    try:
        vertex_labels, num_clusters, debug = spectral_cluster_vertices(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
            n_clusters=requested_clusters,
            vertex_scalars=vertex_scalars,
            scalar_weight=scalar_weight,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )
        face_labels = vertex_labels_to_face_labels(vertex_labels, faces)
    except (ValueError, ImportError, OSError) as exc:
        result.message = f"谱聚类失败: {exc}"
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
            result.message = f"谱聚类完成但渲染失败: {exc}"
            result.ok = True
            return result

    result.ok = True
    auto_note = "" if requested_clusters is not None else "（自动估计簇数）"
    result.message = (
        f"谱聚类完成: {num_clusters} 个部件{auto_note}, "
        f"标签 {labels_path.name}"
    )
    return result
