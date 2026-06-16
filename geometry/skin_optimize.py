"""
语义 rig 蒙皮：默认在建骨 u0（刚性+接缝过渡带）上做网格扩散，自然得到连续、多骨蒙皮。

流程（先松后紧）：
1. Agent 给定分块边界能量；优化用体积比 / 表面积比 / 最大凹陷深度指标
2. 硬权重在网格上的拉普拉斯退火扩散，得到软化锚点 u0
3. 以 u0 为锚的调和蒙皮 + 四大 GRPO 指标（热力/体积/稀疏/变形）

不强行给顶点塞第二根骨；距离核/硬 cap 已默认关闭。可选 ``dirichlet`` / ``u0``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from html import escape as html_escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise ImportError("skin_optimize 需要 trimesh") from exc

try:
    from scipy import sparse
    from scipy.sparse import linalg as splinalg
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover
    raise ImportError("skin_optimize 需要 scipy") from exc

from .semantic_rig import (
    RigAnimation,
    SemanticSkeleton,
    axis_angle_matrix,
    blend_skinning_at_seams,
    make_test_animation,
)

# ---------------------------------------------------------------------------
# 配置（经验权重）
# ---------------------------------------------------------------------------


@dataclass
class SkinOptimizeConfig:
    """联合优化超参。"""

    beta: float = 0.1
    max_influences: int = 4

    # 主能量：初始热力图 / 初始分块贴合（测地热力图拟合类比）
    w_heat: float = 12.0
    w_part_stick: float = 5.0
    w_partition_live: float = 3.0

    # GRPO 几何（论文相对权重；R_vk 低时略提高）
    w_vj: float = 5.0
    w_vk: float = 8.0
    w_sc: float = 2.0
    w_mo: float = 4.0

    # R_mo：给定姿态下边拉伸（论文约 5 个姿态采样，s≈2）
    mo_max_poses: int = 5
    mo_scale: float = 2.0
    mo_eps: float = 1e-6
    joint_fd_eps_ratio: float = 2e-3
    mo_allow_random_fallback: bool = False

    # 关节
    w_joint_reg: float = 1.0
    w_bone_length: float = 1.5
    joint_max_iter: int = 60
    joint_lr: float = 0.028
    joint_move_ratio: float = 0.14
    fix_root_joints: bool = True

    # 蒙皮：smooth = 以建骨 u0（接缝带）为锚，网格扩散自然多骨；不强行塞第二骨
    weight_mode: str = "smooth"
    smooth_sigma_ratio: float = 0.32
    smooth_distance_power: float = 1.15
    smooth_u0_blend: float = 1.0
    smooth_mesh_iters: int = 10
    smooth_mesh_step: float = 0.18
    min_influence_bones: int = 0
    min_influence_weight: float = 0.12
    smooth_dominant_cap: float = 1.0
    blend_radius_ratio: float = 0.0
    smooth_uniform_gamma: float = 0.90
    smooth_continuity_alpha: float = 0.85
    smooth_continuity_passes: int = 1
    smooth_final_diffuse_iters: int = 0
    smooth_u0_anchor: float = 0.22

    # 分块：体积比/表面积比/凹陷深度 + Agent 能量（默认）；direct=旧 argmax
    optimize_partition: bool = False
    partition_refine_iters: int = 0
    partition_mode: str = "metric"
    use_agent_partition_energy: bool = True
    boundary_partition_iters: int = 12
    boundary_dijkstra_passes: int = 2
    boundary_label_hops: int = 2
    # 动画形变能量：边界顶点改绑到相邻骨时，按给定动画姿态下其关联边的拉伸变化
    # 加权，让初始分块边界朝「VLM 动画下形变最小」方向移动（0=纯几何，不启用）。
    w_partition_motion: float = 2.0

    # 硬权重退火：进入四大指标前，先松后紧拉普拉斯扩散
    anneal_pre_diffuse: bool = True
    anneal_stages: int = 4
    anneal_loose_step: float = 0.26
    anneal_tight_step: float = 0.07
    anneal_anchor_start: float = 0.04
    anneal_anchor_end: float = 0.32

    outer_cycles: int = 1
    dirichlet_anchor: float = 12.0
    dirichlet_smooth: bool = False
    report_dirichlet_energy: bool = False

    voxel_resolution: int = 96
    bone_samples: int = 9
    vj_alpha: float = 0.05
    softmin_tau: float = 0.04
    vk_sigma_ratio: float = 0.05
    use_cotangent_laplacian: bool = True
    verbose: bool = True

    optimize_joints: bool = False

    # ---- 加速（见 ``SkinOptimizeConfig.fast()`` / ``fast_mode``）----
    fast_mode: bool = False
    mo_edge_max: int = 8000
    mo_fd_interval: int = 3
    mo_fd_pose_bones_only: bool = True
    dirichlet_cache: bool = True
    partition_energy_heat_only: bool = False
    joint_log_interval: int = 8
    use_voxel_vk: bool = True

    write_partition_report: bool = True
    partition_report_image_size: int = 640
    partition_report_panel_cols: int = 4

    @classmethod
    def fast(cls) -> "SkinOptimizeConfig":
        """较快预设：约 3–8× 加速，精度略降。适合迭代调试。"""
        return cls(
            fast_mode=True,
            weight_mode="smooth",
            smooth_mesh_iters=6,
            smooth_mesh_step=0.15,
            smooth_continuity_alpha=0.6,
            smooth_continuity_passes=1,
            smooth_u0_anchor=0.25,
            outer_cycles=1,
            optimize_partition=False,
            partition_refine_iters=0,
            boundary_partition_iters=0,
            use_agent_partition_energy=False,
            anneal_pre_diffuse=False,
            optimize_joints=False,
            joint_max_iter=16,
            report_dirichlet_energy=False,
            mo_max_poses=3,
            mo_edge_max=5000,
            mo_fd_interval=4,
            mo_fd_pose_bones_only=True,
            dirichlet_cache=True,
            partition_energy_heat_only=True,
            voxel_resolution=64,
            bone_samples=5,
            use_cotangent_laplacian=False,
            joint_log_interval=8,
            use_voxel_vk=True,
            write_partition_report=False,
            w_partition_motion=0.0,
        )


_BONE_PALETTE: List[Tuple[int, int, int]] = [
    (230, 60, 40), (46, 158, 94), (52, 120, 220), (242, 168, 43),
    (150, 90, 200), (40, 190, 200), (220, 90, 160), (120, 130, 60),
    (200, 120, 60), (90, 200, 120), (120, 100, 220), (200, 200, 60),
]


@dataclass
class SkinRewardReport:
    R_vj: float
    R_vk: float
    R_sc: float
    R_z: float
    R_m: float
    R_mo: float
    mo_stretch: float
    laplacian_energy: float
    heat_mse: float
    partition_change_ratio: float
    loss_total: float
    report_html: Optional[str] = None
    partition_before_png: Optional[str] = None
    partition_after_png: Optional[str] = None
    partition_boundary_png: Optional[str] = None

    def to_dict(self) -> Dict[str, Union[float, str, None]]:
        d: Dict[str, Union[float, str, None]] = {
            "R_vj": float(self.R_vj),
            "R_vk": float(self.R_vk),
            "R_sc": float(self.R_sc),
            "R_z": float(self.R_z),
            "R_m": float(self.R_m),
            "R_mo": float(self.R_mo),
            "mo_stretch": float(self.mo_stretch),
            "laplacian_energy": float(self.laplacian_energy),
            "heat_mse": float(self.heat_mse),
            "partition_change_ratio": float(self.partition_change_ratio),
            "loss_total": float(self.loss_total),
        }
        for key in (
            "report_html",
            "partition_before_png",
            "partition_after_png",
            "partition_boundary_png",
        ):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d


# ---------------------------------------------------------------------------
# 拉普拉斯 / Dirichlet 能量平滑
# ---------------------------------------------------------------------------


def _mesh_edges(faces: np.ndarray, num_v: int) -> np.ndarray:
    e = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            e.add((u, v))
    if not e:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(e), dtype=np.int64)


def _cotangent_weights(
    vertices: np.ndarray, faces: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    w = np.zeros(len(edges), dtype=np.float64)
    edge_to_idx = {tuple(e): i for i, e in enumerate(edges)}
    for tri in faces:
        ids = [int(tri[0]), int(tri[1]), int(tri[2])]
        pts = vertices[ids]
        for k in range(3):
            i, j = ids[k], ids[(k + 1) % 3]
            ekey = (i, j) if i < j else (j, i)
            if ekey not in edge_to_idx:
                continue
            ei = edge_to_idx[ekey]
            v1 = pts[(k + 1) % 3] - pts[k]
            v2 = pts[(k + 2) % 3] - pts[k]
            sin_a = np.linalg.norm(np.cross(v1, v2))
            if sin_a < 1e-12:
                continue
            w[ei] += max(float(np.dot(v1, v2) / sin_a), 0.0)
    w[w < 1e-12] = 1e-12
    return w


def build_mesh_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    cotangent: bool = True,
) -> sparse.csr_matrix:
    num_v = len(vertices)
    edges = _mesh_edges(faces, num_v)
    if len(edges) == 0:
        return sparse.csr_matrix((num_v, num_v))
    ew = (
        _cotangent_weights(vertices, faces, edges)
        if cotangent
        else np.ones(len(edges), dtype=np.float64)
    )
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([ew, ew])
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(num_v, num_v))
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return sparse.diags(deg) - adj


class _ImplicitSmoothCache:
    """缓存 ``(I + α L)`` 分解，用于权重空间连续性平滑。"""

    def __init__(self, lap: sparse.csr_matrix, alpha: float) -> None:
        self.alpha = float(alpha)
        n = lap.shape[0]
        A = sparse.eye(n, format="csc") + self.alpha * lap.tocsc()
        self._solve = splinalg.factorized(A)

    def smooth_columns(self, w: np.ndarray) -> np.ndarray:
        return np.asarray(self._solve(np.asarray(w, dtype=np.float64)), dtype=np.float64)


class _DirichletSmoothCache:
    """缓存 ``(L + μ I)`` 分解，各骨列共用。"""

    def __init__(self, lap: sparse.csr_matrix, anchor: float) -> None:
        self.anchor = float(anchor)
        n = lap.shape[0]
        A = lap + self.anchor * sparse.eye(n, format="csc")
        self._solve = splinalg.factorized(A.tocsc())

    def smooth_columns(self, u0: np.ndarray) -> np.ndarray:
        rhs = self.anchor * np.asarray(u0, dtype=np.float64)
        return np.asarray(self._solve(rhs), dtype=np.float64)


def dirichlet_smooth_skin_weights(
    lap: sparse.csr_matrix,
    u0: np.ndarray,
    *,
    anchor: float = 12.0,
    cache: Optional[_DirichletSmoothCache] = None,
) -> np.ndarray:
    """Dirichlet 能量 + 贴合 ``u0``：``min ½ wᵀLw + (μ/2)‖w-u0‖²``，逐骨列求解。"""
    if cache is not None:
        w = cache.smooth_columns(u0)
    else:
        n = lap.shape[0]
        mu = float(anchor)
        A = lap + mu * sparse.eye(n, format="csc")
        factor = splinalg.factorized(A.tocsc())
        w = np.asarray(factor(mu * u0), dtype=np.float64)
    w = np.maximum(w, 0.0)
    row = w.sum(axis=1, keepdims=True)
    row[row < 1e-12] = 1.0
    return w / row


def dirichlet_energy(
    lap: sparse.csr_matrix,
    weights: np.ndarray,
) -> float:
    """``½ Σ_b w_bᵀ L w_b``。"""
    lw = lap @ weights
    return 0.5 * float(np.sum(weights * lw))


def _point_segment_distance(
    points: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
) -> np.ndarray:
    """(V,) 到线段 head–tail 的欧氏距离。"""
    ab = tail - head
    ap = points - head
    denom = float(np.dot(ab, ab)) + 1e-12
    t = np.clip(ap @ ab / denom, 0.0, 1.0)
    closest = head + t[:, np.newaxis] * ab
    return np.linalg.norm(points - closest, axis=1)


def _vertex_bone_distances(
    skeleton: SemanticSkeleton,
) -> Tuple[np.ndarray, np.ndarray, float]:
    verts = skeleton.vertices
    joints = _joint_positions(skeleton)
    num_v = len(verts)
    num_b = skeleton.num_bones
    dist = np.full((num_v, num_b), np.inf, dtype=np.float64)
    for b in skeleton.bones:
        bi = b.index
        if b.parent < 0:
            dist[:, bi] = np.linalg.norm(verts - joints[bi], axis=1)
        else:
            dist[:, bi] = _point_segment_distance(
                verts, joints[b.parent], joints[bi],
            )
    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    return dist, verts, bbox_diag


def _bone_neighbor_lists(skeleton: SemanticSkeleton) -> List[List[int]]:
    n = skeleton.num_bones
    adj: List[set] = [set() for _ in range(n)]
    for b in skeleton.bones:
        if b.parent >= 0:
            adj[b.index].add(b.parent)
            adj[b.parent].add(b.index)
        for c in b.children:
            adj[b.index].add(c)
    for j in skeleton.joints:
        adj[j.parent_bone].add(j.child_bone)
        adj[j.child_bone].add(j.parent_bone)
    return [sorted(s) for s in adj]


def _enforce_min_influences(
    w: np.ndarray,
    dist: np.ndarray,
    skeleton: SemanticSkeleton,
    *,
    min_bones: int,
    min_weight: float,
) -> np.ndarray:
    """（默认关闭）仅在 min_bones>=2 时强制相邻骨最低权重。"""
    if min_bones < 2:
        return np.maximum(w, 0.0)
    w = np.maximum(w, 0.0).copy()
    num_v, num_b = w.shape
    neighbors = _bone_neighbor_lists(skeleton)
    min_bones = max(1, min(min_bones, num_b))
    min_weight = float(np.clip(min_weight, 0.05, 0.45))

    for v in range(num_v):
        if (w[v] >= min_weight).sum() >= min_bones:
            continue
        dom = int(np.argmax(w[v]))
        nbrs = [b for b in neighbors[dom] if b != dom]
        if not nbrs:
            nbrs = list(np.argsort(dist[v])[: min_bones])
        nbrs = sorted(nbrs, key=lambda b: dist[v, b])[: max(min_bones - 1, 0)]
        w[v, dom] = max(w[v, dom], 1.0 - min_weight * len(nbrs))
        for b in nbrs:
            w[v, b] = max(w[v, b], min_weight)
    row = w.sum(axis=1, keepdims=True)
    w /= np.maximum(row, 1e-12)
    return w


def _normalize_skin_rows(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    row = w.sum(axis=1, keepdims=True)
    return w / np.maximum(row, 1e-12)


def weight_edge_discontinuity(
    weights: np.ndarray,
    faces: np.ndarray,
) -> float:
    """网格边两侧权重差的平均 L2 范数（越小越连续）。"""
    num_v = len(weights)
    edges = _mesh_edges(faces, num_v)
    if len(edges) == 0:
        return 0.0
    diff = weights[edges[:, 0]] - weights[edges[:, 1]]
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _implicit_continuity_smooth(
    lap: sparse.csr_matrix,
    w: np.ndarray,
    alpha: float,
    *,
    cache: Optional[_ImplicitSmoothCache] = None,
    passes: int = 1,
) -> np.ndarray:
    """``(I + α L)^{-1} W``：在贴合当前权重的同时沿网格强平滑。"""
    if passes <= 0:
        return w
    if cache is None:
        cache = _ImplicitSmoothCache(lap, alpha)
    out = np.asarray(w, dtype=np.float64)
    for _ in range(passes):
        out = cache.smooth_columns(out)
        out = _normalize_skin_rows(out)
    return out


def _diffuse_weights_on_mesh(
    w: np.ndarray,
    lap: sparse.csr_matrix,
    *,
    iters: int,
    step: float,
) -> np.ndarray:
    """在网格上扩散权重列，使相邻顶点影响骨一致变化。"""
    w = np.maximum(w, 0.0).copy()
    step = float(np.clip(step, 0.05, 0.55))
    for _ in range(max(iters, 0)):
        w = w - step * (lap @ w)
        w = np.maximum(w, 0.0)
        row = w.sum(axis=1, keepdims=True)
        w /= np.maximum(row, 1e-12)
    return w


def smooth_combined_skin_weights(
    skeleton: SemanticSkeleton,
    u0: np.ndarray,
    cfg: SkinOptimizeConfig,
    *,
    lap: Optional[sparse.csr_matrix] = None,
) -> np.ndarray:
    """以建骨蒙皮 u0（含接缝过渡）为锚，沿网格热扩散；多骨由扩散自然产生，不硬塞。"""
    u0p = project_skin_weights(u0, cfg.max_influences)
    w = np.maximum(u0p, 0.0).copy()

    if lap is None:
        lap = build_mesh_laplacian(
            skeleton.vertices,
            skeleton.faces,
            cotangent=cfg.use_cotangent_laplacian,
        )

    step = float(np.clip(cfg.smooth_mesh_step, 0.06, 0.28))
    if cfg.smooth_mesh_iters > 0:
        w = _diffuse_weights_on_mesh(
            w, lap, iters=cfg.smooth_mesh_iters, step=step,
        )

    if cfg.smooth_continuity_passes > 0 and cfg.smooth_continuity_alpha > 0:
        w = _implicit_continuity_smooth(
            lap, w, cfg.smooth_continuity_alpha, passes=cfg.smooth_continuity_passes,
        )

    if cfg.smooth_final_diffuse_iters > 0:
        w = _diffuse_weights_on_mesh(
            w, lap, iters=cfg.smooth_final_diffuse_iters, step=step,
        )

    gamma = float(np.clip(cfg.smooth_uniform_gamma, 0.8, 1.0))
    if gamma < 0.998:
        nz = w > 1e-10
        w[nz] = w[nz] ** gamma
        w = _normalize_skin_rows(w)

    anchor = float(np.clip(cfg.smooth_u0_anchor, 0.0, 0.5))
    if anchor > 0:
        w = (1.0 - anchor) * w + anchor * u0p
        w = _normalize_skin_rows(w)

    return project_skin_weights(w, cfg.max_influences)


def smooth_distance_skin_weights(
    skeleton: SemanticSkeleton,
    u0: np.ndarray,
    cfg: SkinOptimizeConfig,
    *,
    lap: Optional[sparse.csr_matrix] = None,
) -> np.ndarray:
    return smooth_combined_skin_weights(skeleton, u0, cfg, lap=lap)


def _compute_skin_weights(
    skeleton: SemanticSkeleton,
    u0: np.ndarray,
    cfg: SkinOptimizeConfig,
    *,
    lap: Optional[sparse.csr_matrix] = None,
    dcache: Optional[_DirichletSmoothCache] = None,
) -> np.ndarray:
    mode = (cfg.weight_mode or "smooth").lower()
    if mode == "u0":
        return project_skin_weights(u0, cfg.max_influences)
    if mode == "dirichlet" or cfg.dirichlet_smooth:
        if lap is None:
            raise ValueError("dirichlet 模式需要拉普拉斯矩阵")
        w = dirichlet_smooth_skin_weights(
            lap, u0, anchor=cfg.dirichlet_anchor, cache=dcache,
        )
        return project_skin_weights(w, cfg.max_influences)
    return smooth_combined_skin_weights(skeleton, u0, cfg, lap=lap)


def _cap_dominant_weights(
    weights: np.ndarray,
    cap: float,
) -> np.ndarray:
    """限制单骨最大权重；cap>=1 时关闭。"""
    cap = float(cap)
    if cap >= 0.999:
        return np.maximum(weights, 0.0)
    cap = float(np.clip(cap, 0.35, 0.95))
    w = np.maximum(weights, 0.0).copy()
    dom = w.max(axis=1)
    over = dom > cap
    if not np.any(over):
        return w
    rows = np.where(over)[0]
    cols = w.argmax(axis=1)
    w[rows, cols[rows]] = cap
    rest = 1.0 - cap
    other = w[rows].copy()
    other[np.arange(len(rows)), cols[rows]] = 0.0
    osum = other.sum(axis=1, keepdims=True)
    scale = np.where(osum > 1e-12, rest / osum, 0.0)
    w[rows] = other * scale
    w[rows, cols[rows]] = cap
    return w


def skin_weight_summary(
    weights: np.ndarray,
    *,
    thr: float = 0.14,
) -> Dict[str, float]:
    w = np.asarray(weights, dtype=np.float64)
    nz = (w > thr).sum(axis=1)
    dom = w.max(axis=1)
    return {
        "mean_bones": float(nz.mean()),
        "min_bones": float(nz.min()),
        "max_bones": float(nz.max()),
        "mean_dominant": float(dom.mean()),
        "frac_dominant_gt_0.85": float((dom > 0.85).mean()),
        "frac_ge2_bones": float((nz >= 2).mean()),
        "frac_ge3_bones": float((nz >= 3).mean()),
    }


def project_skin_weights(
    weights: np.ndarray,
    max_bones: int = 4,
) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    num_v, num_b = w.shape
    k = min(max_bones, num_b)
    if k < num_b:
        top = np.argpartition(-w, k, axis=1)[:, :k]
        mask = np.zeros_like(w)
        rows = np.arange(num_v)[:, None]
        mask[rows, top] = 1.0
        w *= mask
    row = w.sum(axis=1, keepdims=True)
    row[row < 1e-12] = 1.0
    return w / row


def laplacian_residual_norm(
    lap: sparse.csr_matrix,
    weights: np.ndarray,
) -> float:
    """``‖L W‖`` 均方根（监控用，非优化目标）。"""
    lw = lap @ weights
    return float(np.sqrt(np.mean(lw ** 2)))


# ---------------------------------------------------------------------------
# 分块：体积比 / 表面积比 / 凹陷深度 + Agent 能量边界松弛
# ---------------------------------------------------------------------------


def _vertex_adjacency(edges: np.ndarray, num_v: int) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_v)]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def _signed_mean_curvature(
    vertices: np.ndarray,
    faces: np.ndarray,
    lap: sparse.csr_matrix,
) -> Tuple[np.ndarray, np.ndarray]:
    """离散平均曲率：``H_vec = L @ V``，符号相对顶点法向（凹为负）。"""
    h_vec = np.asarray(lap @ vertices, dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
    signed = np.sum(h_vec * vn, axis=1)
    groove = np.maximum(-signed, 0.0)
    return signed, groove


def _boundary_vertex_mask(part: np.ndarray, edges: np.ndarray) -> np.ndarray:
    on_bnd = np.zeros(len(part), dtype=bool)
    for u, v in edges:
        if part[u] != part[v]:
            on_bnd[u] = True
            on_bnd[v] = True
    return on_bnd


def _hard_weights_from_partition(
    part: np.ndarray,
    num_bones: int,
) -> np.ndarray:
    num_v = len(part)
    w = np.zeros((num_v, num_bones), dtype=np.float64)
    valid = (part >= 0) & (part < num_bones)
    w[valid, part[valid]] = 1.0
    return w


def _u0_from_partition(
    skeleton: SemanticSkeleton,
    part: np.ndarray,
    cfg: SkinOptimizeConfig,
) -> np.ndarray:
    """由硬分块生成 u0（可选接缝过渡）。"""
    w = _hard_weights_from_partition(part, skeleton.num_bones)
    br = float(cfg.blend_radius_ratio)
    if br > 0 and skeleton.joints:
        verts = skeleton.vertices
        bbox = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        w = blend_skinning_at_seams(
            verts, w, skeleton.joints, max(bbox * br, 1e-9),
        )
    return project_skin_weights(w, cfg.max_influences)


@dataclass
class PartitionGeometryMetrics:
    """硬分块三类几何指标（按骨段）。"""

    volume_ratio: np.ndarray
    area_ratio: np.ndarray
    max_concavity_depth: np.ndarray

    def to_dict(self, bone_names: Sequence[str]) -> Dict[str, object]:
        rows = []
        for i, name in enumerate(bone_names):
            rows.append({
                "name": name,
                "volume_ratio": float(self.volume_ratio[i]),
                "area_ratio": float(self.area_ratio[i]),
                "max_concavity_depth": float(self.max_concavity_depth[i]),
            })
        return {"bones": rows}


@dataclass
class PartitionEnergySpec:
    """Agent（或默认回退）给定的分块边界能量参数。"""

    w_volume: float = 1.0
    w_area: float = 0.8
    w_concavity: float = 2.5
    w_stick: float = 2.5
    w_smooth: float = 1.2
    volume_ratio_target: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64),
    )
    area_ratio_target: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64),
    )
    concavity_pull: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64),
    )
    seam_w: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64),
    )
    source: str = "default"
    bone_notes: Dict[int, str] = field(default_factory=dict)

    def to_dict(self, bone_names: Sequence[str]) -> Dict[str, object]:
        bones = []
        for i, name in enumerate(bone_names):
            bones.append({
                "name": name,
                "volume_ratio_target": float(self.volume_ratio_target[i]),
                "area_ratio_target": float(self.area_ratio_target[i]),
                "concavity_pull": float(self.concavity_pull[i]),
                "notes": self.bone_notes.get(i, ""),
            })
        seams = []
        n = len(bone_names)
        for a in range(n):
            for b in range(a + 1, n):
                w = float(self.seam_w[a, b])
                if w > 1e-9:
                    seams.append({
                        "bone_a": bone_names[a],
                        "bone_b": bone_names[b],
                        "w_concavity": w,
                    })
        return {
            "source": self.source,
            "global": {
                "w_volume": self.w_volume,
                "w_area": self.w_area,
                "w_concavity": self.w_concavity,
                "w_stick": self.w_stick,
                "w_smooth": self.w_smooth,
            },
            "bones": bones,
            "seams": seams,
        }


def _vertex_mass_weights(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """顶点质量权重（面积均分），总和归一为网格体积估计。"""
    num_v = len(vertices)
    vm = np.zeros(num_v, dtype=np.float64)
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        area = 0.5 * float(
            np.linalg.norm(np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a]))
        )
        share = area / 3.0
        vm[a] += share
        vm[b] += share
        vm[c] += share
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.is_volume:
        total_vol = abs(float(mesh.volume))
    else:
        total_vol = abs(float(mesh.convex_hull.volume))
    s = float(vm.sum())
    if s < 1e-12:
        return np.full(num_v, total_vol / max(num_v, 1), dtype=np.float64)
    return vm * (total_vol / s)


def _normalize_groove_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    lap: sparse.csr_matrix,
) -> np.ndarray:
    """凹槽深度 [0,1]：负平均曲率相对外法向，按 bbox 尺度归一。"""
    _, groove = _signed_mean_curvature(vertices, faces, lap)
    bbox = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    scale = max(bbox * 0.02, groove.max(), 1e-9)
    return np.clip(groove / scale, 0.0, 1.0)


def compute_partition_geometry_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    part: np.ndarray,
    num_bones: int,
    groove: np.ndarray,
    edges: np.ndarray,
) -> PartitionGeometryMetrics:
    """计算体积比、表面积比、最大凹陷深度。"""
    vm = _vertex_mass_weights(vertices, faces)
    vol = np.zeros(num_bones, dtype=np.float64)
    for v in range(len(part)):
        b = int(part[v])
        if 0 <= b < num_bones:
            vol[b] += vm[v]
    volume_ratio = vol / max(vol.sum(), 1e-12)

    bnd_area = np.zeros(num_bones, dtype=np.float64)
    for u, v in edges:
        if part[u] != part[v]:
            el = float(np.linalg.norm(vertices[u] - vertices[v]))
            bnd_area[int(part[u])] += el * 0.5
            bnd_area[int(part[v])] += el * 0.5
    area_ratio = bnd_area / max(bnd_area.sum(), 1e-12)

    max_conc = np.zeros(num_bones, dtype=np.float64)
    bnd = _boundary_vertex_mask(part, edges)
    for v in np.where(bnd)[0]:
        b = int(part[v])
        if 0 <= b < num_bones:
            max_conc[b] = max(max_conc[b], float(groove[v]))

    return PartitionGeometryMetrics(volume_ratio, area_ratio, max_conc)


def _ratio_sq_penalty(ratio: np.ndarray, target: np.ndarray) -> float:
    n = min(len(ratio), len(target))
    if n == 0:
        return 0.0
    d = ratio[:n] - target[:n]
    return float(np.sum(d * d))


class _PartitionMetricState:
    """可增量更新的分块指标状态（边界顶点标号切换）。"""

    def __init__(
        self,
        vertices: np.ndarray,
        part: np.ndarray,
        vertex_mass: np.ndarray,
        groove: np.ndarray,
        neighbors: List[List[int]],
        energy_spec: PartitionEnergySpec,
        *,
        pose_mats: Optional[np.ndarray] = None,
        w_motion: float = 0.0,
    ) -> None:
        self.vertices = vertices
        self.part = part
        self.vertex_mass = vertex_mass
        self.groove = groove
        self.neighbors = neighbors
        self.spec = energy_spec
        self.num_bones = len(energy_spec.volume_ratio_target)
        self.vol = np.zeros(self.num_bones, dtype=np.float64)
        self.bnd_len = np.zeros(self.num_bones, dtype=np.float64)
        self._rebuild_vol()
        self._rebuild_bnd_len()
        # 动画形变项：(P,B,4,4) 各姿态各骨蒙皮矩阵 + 齐次顶点；w_motion<=0 时关闭
        self.w_motion = float(w_motion)
        self.pose_mats = (
            pose_mats
            if (pose_mats is not None and len(pose_mats) > 0 and self.w_motion > 0.0)
            else None
        )
        self.motion_on = self.pose_mats is not None
        self.vh = (
            np.hstack([vertices, np.ones((len(vertices), 1))])
            if self.motion_on
            else None
        )

    def _rebuild_vol(self) -> None:
        self.vol.fill(0.0)
        for v in range(len(self.part)):
            b = int(self.part[v])
            if 0 <= b < self.num_bones:
                self.vol[b] += self.vertex_mass[v]

    def _edge_on_boundary(self, u: int, v: int) -> bool:
        return int(self.part[u]) != int(self.part[v])

    def _rebuild_bnd_len(self) -> None:
        self.bnd_len.fill(0.0)
        for v in range(len(self.part)):
            for n in self.neighbors[v]:
                if n < v:
                    continue
                if self._edge_on_boundary(v, n):
                    el = float(np.linalg.norm(self.vertices[v] - self.vertices[n]))
                    self.bnd_len[int(self.part[v])] += el * 0.5
                    self.bnd_len[int(self.part[n])] += el * 0.5

    def metrics(self, edges: np.ndarray) -> PartitionGeometryMetrics:
        return compute_partition_geometry_metrics(
            self.vertices,
            np.zeros((0, 3), dtype=np.int64),
            self.part,
            self.num_bones,
            self.groove,
            edges,
        )

    def vol_ratio(self) -> np.ndarray:
        return self.vol / max(self.vol.sum(), 1e-12)

    def area_ratio(self) -> np.ndarray:
        return self.bnd_len / max(self.bnd_len.sum(), 1e-12)

    def global_metric_energy(self) -> float:
        sp = self.spec
        vr = self.vol_ratio()
        ar = self.area_ratio()
        e = sp.w_volume * _ratio_sq_penalty(vr, sp.volume_ratio_target)
        e += sp.w_area * _ratio_sq_penalty(ar, sp.area_ratio_target)
        return float(e)

    def _local_motion_penalty(self, v: int, label: int) -> float:
        """顶点 v 绑到骨 ``label`` 时，其关联边在各动画姿态下的平均超伸惩罚。

        近似把顶点视为刚绑到 ``label`` 骨：对每个姿态用该骨蒙皮矩阵变形 v，与邻
        点（按各自当前标号变形）比较边长，取 ``max(l_def/l_rest - 1, 0)``，对边求
        和、对姿态取平均。改标号只改 v 的变形，故是 O(deg×P) 的局部量。
        """
        if not self.motion_on:
            return 0.0
        nbrs = self.neighbors[v]
        if not nbrs:
            return 0.0
        vd_v = (self.pose_mats[:, label] @ self.vh[v])[:, :3]  # (P,3)
        pen = 0.0
        for n in nbrs:
            bn = int(self.part[n])
            vd_n = (self.pose_mats[:, bn] @ self.vh[n])[:, :3]  # (P,3)
            rest = float(np.linalg.norm(self.vertices[v] - self.vertices[n]))
            if rest < 1e-12:
                continue
            ratio = np.linalg.norm(vd_v - vd_n, axis=1) / rest
            pen += float(np.mean(np.maximum(ratio - 1.0, 0.0)))
        return pen

    def _local_concavity_bonus(self, v: int, label: int) -> float:
        g = float(self.groove[v])
        if g < 1e-9:
            return 0.0
        bonus = 0.0
        for n in self.neighbors[v]:
            nb = int(self.part[n])
            if nb != label:
                bonus += float(self.spec.seam_w[label, nb]) * g
        pull = (
            float(self.spec.concavity_pull[label])
            if label < len(self.spec.concavity_pull)
            else 1.0
        )
        return pull * bonus

    def relabel_delta_energy(
        self,
        v: int,
        new_b: int,
        part0: np.ndarray,
    ) -> float:
        """将边界顶点 ``v`` 从当前标号改为 ``new_b`` 的能量变化（负=更优）。"""
        old = int(self.part[v])
        if new_b == old:
            return 0.0

        vr = self.vol_ratio()
        ar = self.area_ratio()
        vol_def = self.spec.volume_ratio_target - vr
        area_def = self.spec.area_ratio_target - ar
        # 骨段缺口驱动：标号流向“体积/面积仍不足”的骨段（O(1) 尺度，非单点 1e-7）
        dE_vol = -self.spec.w_volume * float(vol_def[new_b] - vol_def[old])
        dE_area = -self.spec.w_area * float(area_def[new_b] - area_def[old])

        nbrs = self.part[self.neighbors[v]] if self.neighbors[v] else np.array([], dtype=np.int64)
        smooth_old = float(np.mean(nbrs != old)) if nbrs.size else 0.0
        smooth_new = float(np.mean(nbrs != new_b)) if nbrs.size else 0.0
        dE_smooth = self.spec.w_smooth * (smooth_new - smooth_old)

        dE_stick = self.spec.w_stick * (
            float(new_b != int(part0[v])) - float(old != int(part0[v]))
        )

        dE_conc = -(
            self._local_concavity_bonus(v, new_b)
            - self._local_concavity_bonus(v, old)
        )
        dE_conc *= self.spec.w_concavity

        dE_motion = 0.0
        if self.motion_on:
            dE_motion = self.w_motion * (
                self._local_motion_penalty(v, new_b)
                - self._local_motion_penalty(v, old)
            )

        return dE_vol + dE_area + dE_smooth + dE_stick + dE_conc + dE_motion

    def _inc_bnd_edge(self, u: int, v: int, sign: float) -> None:
        if int(self.part[u]) == int(self.part[v]):
            return
        el = float(np.linalg.norm(self.vertices[u] - self.vertices[v])) * 0.5
        self.bnd_len[int(self.part[u])] += sign * el
        self.bnd_len[int(self.part[v])] += sign * el

    def apply_relabel(self, v: int, new_b: int) -> None:
        old = int(self.part[v])
        if new_b == old:
            return
        m = float(self.vertex_mass[v])
        self.vol[old] -= m
        self.vol[new_b] += m
        for n in self.neighbors[v]:
            self._inc_bnd_edge(v, n, -1.0)
        self.part[v] = int(new_b)
        for n in self.neighbors[v]:
            self._inc_bnd_edge(v, n, +1.0)


def _boundary_label_candidates(
    v: int,
    part: np.ndarray,
    neighbors: List[List[int]],
    hops: int,
) -> set:
    """边界顶点可选标号：自身 + k 环邻域内出现过的骨段。"""
    hops = max(int(hops), 1)
    cand: set = set()
    frontier = {v}
    seen = {v}
    for _ in range(hops):
        nxt: set = set()
        for u in frontier:
            cand.add(int(part[u]))
            for n in neighbors[u]:
                if n not in seen:
                    seen.add(n)
                    nxt.add(n)
        frontier = nxt
    return cand


def _skeleton_seam_pairs(skeleton: SemanticSkeleton) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    for j in skeleton.joints:
        a, b = int(j.parent_bone), int(j.child_bone)
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((skeleton.bones[a].name, skeleton.bones[b].name))
    return pairs


def default_partition_energy_spec(
    skeleton: SemanticSkeleton,
    metrics: PartitionGeometryMetrics,
) -> PartitionEnergySpec:
    """无 Agent 时：以建骨初始指标为目标，接缝默认凹陷吸引。"""
    n = skeleton.num_bones
    seam_w = np.zeros((n, n), dtype=np.float64)
    for j in skeleton.joints:
        a, b = int(j.parent_bone), int(j.child_bone)
        seam_w[a, b] = seam_w[b, a] = 2.0
    pull = np.ones(n, dtype=np.float64)
    for b in skeleton.bones:
        if b.parent < 0:
            pull[b.index] = 0.5
    return PartitionEnergySpec(
        w_volume=1.0,
        w_area=0.8,
        w_concavity=2.5,
        w_stick=0.8,
        w_smooth=1.0,
        volume_ratio_target=metrics.volume_ratio.copy(),
        area_ratio_target=metrics.area_ratio.copy(),
        concavity_pull=pull,
        seam_w=seam_w,
        source="default",
    )


def partition_energy_spec_from_agent_dict(
    payload: Dict[str, object],
    skeleton: SemanticSkeleton,
    metrics: PartitionGeometryMetrics,
) -> PartitionEnergySpec:
    """把 Agent JSON 解析为 ``PartitionEnergySpec``。"""
    n = skeleton.num_bones
    name_to_idx = {b.name: b.index for b in skeleton.bones}
    g = payload.get("global") if isinstance(payload.get("global"), dict) else {}
    vol_t = metrics.volume_ratio.copy()
    area_t = metrics.area_ratio.copy()
    pull = np.ones(n, dtype=np.float64)
    notes: Dict[int, str] = {}
    bones = payload.get("bones")
    if isinstance(bones, list):
        for row in bones:
            if not isinstance(row, dict):
                continue
            nm = str(row.get("name", "")).strip()
            if nm not in name_to_idx:
                continue
            i = name_to_idx[nm]
            if "volume_ratio_target" in row:
                vol_t[i] = float(row["volume_ratio_target"])
            if "area_ratio_target" in row:
                area_t[i] = float(row["area_ratio_target"])
            if "concavity_pull" in row:
                pull[i] = float(row["concavity_pull"])
            if row.get("notes"):
                notes[i] = str(row["notes"])
    seam_w = np.zeros((n, n), dtype=np.float64)
    seams = payload.get("seams")
    if isinstance(seams, list):
        for row in seams:
            if not isinstance(row, dict):
                continue
            a = str(row.get("bone_a", "")).strip()
            b = str(row.get("bone_b", "")).strip()
            if a not in name_to_idx or b not in name_to_idx:
                continue
            ia, ib = name_to_idx[a], name_to_idx[b]
            w = float(row.get("w_concavity", 2.0))
            seam_w[ia, ib] = seam_w[ib, ia] = max(seam_w[ia, ib], w)
    for j in skeleton.joints:
        ia, ib = int(j.parent_bone), int(j.child_bone)
        if seam_w[ia, ib] < 1e-9:
            seam_w[ia, ib] = seam_w[ib, ia] = 1.5
    return PartitionEnergySpec(
        w_volume=float(g.get("w_volume", 1.0)),
        w_area=float(g.get("w_area", 0.8)),
        w_concavity=float(g.get("w_concavity", 2.5)),
        w_stick=float(g.get("w_stick", 2.5)),
        w_smooth=float(g.get("w_smooth", 1.2)),
        volume_ratio_target=vol_t,
        area_ratio_target=area_t,
        concavity_pull=pull,
        seam_w=seam_w,
        source="agent",
        bone_notes=notes,
    )


def save_partition_energy_spec(
    spec: PartitionEnergySpec,
    skeleton: SemanticSkeleton,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [b.name for b in skeleton.bones]
    path.write_text(
        json.dumps(spec.to_dict(names), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_partition_energy_spec(
    path: Path,
    skeleton: SemanticSkeleton,
    metrics: PartitionGeometryMetrics,
) -> PartitionEnergySpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "global" in payload or "bones" in payload:
        return partition_energy_spec_from_agent_dict(payload, skeleton, metrics)
    return default_partition_energy_spec(skeleton, metrics)


def resolve_partition_energy_spec(
    skeleton: SemanticSkeleton,
    part0: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    lap: sparse.csr_matrix,
    edges: np.ndarray,
    cfg: SkinOptimizeConfig,
    *,
    report_dir: Optional[Path] = None,
    vlm_client: Optional[object] = None,
    image_path: Optional[Path] = None,
    spec_override: Optional[PartitionEnergySpec] = None,
) -> Tuple[PartitionEnergySpec, PartitionGeometryMetrics]:
    """解析 Agent 能量：缓存文件 > 显式传入 > VLM > 默认回退。"""
    groove = _normalize_groove_depth(vertices, faces, lap)
    metrics0 = compute_partition_geometry_metrics(
        vertices, faces, part0, skeleton.num_bones, groove, edges,
    )
    if spec_override is not None:
        return spec_override, metrics0

    cache = Path(report_dir) / "partition_energy_spec.json" if report_dir else None
    if cache is not None and cache.is_file():
        spec = load_partition_energy_spec(cache, skeleton, metrics0)
        spec.source = "cache"
        return spec, metrics0

    if cfg.use_agent_partition_energy and vlm_client is not None:
        from vlm.partition_energy_designer import design_partition_energy

        parents = [b.parent for b in skeleton.bones]
        names = [b.name for b in skeleton.bones]
        design = design_partition_energy(
            vlm_client,
            names,
            parents=parents,
            seam_pairs=_skeleton_seam_pairs(skeleton),
            volume_ratio=metrics0.volume_ratio.tolist(),
            area_ratio=metrics0.area_ratio.tolist(),
            max_concavity_depth=metrics0.max_concavity_depth.tolist(),
            image_path=image_path,
        )
        if design.ok:
            spec = partition_energy_spec_from_agent_dict(
                design.spec, skeleton, metrics0,
            )
            if cache is not None:
                save_partition_energy_spec(spec, skeleton, cache)
                (Path(report_dir) / "partition_energy_design.json").write_text(
                    json.dumps(design.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            if cfg.verbose:
                print(
                    f"[part-agent] 能量设计完成 ({design.elapsed_seconds:.1f}s)  "
                    f"w_vol={spec.w_volume:.2f} w_conc={spec.w_concavity:.2f}"
                )
            return spec, metrics0
        if cfg.verbose:
            print(f"[part-agent] 回退默认能量: {design.error}")

    spec = default_partition_energy_spec(skeleton, metrics0)
    if cache is not None:
        save_partition_energy_spec(spec, skeleton, cache)
    return spec, metrics0


def _try_boundary_relabel(
    state: _PartitionMetricState,
    v: int,
    part0: np.ndarray,
    label_hops: int,
) -> bool:
    """若存在严格降能标号则 relabel，返回是否发生变更。"""
    old = int(state.part[v])
    cand = _boundary_label_candidates(v, state.part, state.neighbors, label_hops)
    best_l = old
    best_dE = 0.0
    for L in cand:
        if not (0 <= int(L) < state.num_bones):
            continue
        dE = state.relabel_delta_energy(v, int(L), part0)
        if dE < best_dE - 1e-12:
            best_dE, best_l = dE, int(L)
    if best_l != old:
        state.apply_relabel(v, best_l)
        return True
    return False


def _relax_boundary_metric_greedy(
    state: _PartitionMetricState,
    part0: np.ndarray,
    edges: np.ndarray,
    groove: np.ndarray,
    label_hops: int,
) -> int:
    n_chg = 0
    bnd = _boundary_vertex_mask(state.part, edges)
    order = np.where(bnd)[0]
    if order.size:
        order = order[np.argsort(-groove[order])]
    for v in order:
        if _try_boundary_relabel(state, int(v), part0, label_hops):
            n_chg += 1
    return n_chg


def _relax_boundary_metric_dijkstra(
    state: _PartitionMetricState,
    part0: np.ndarray,
    edges: np.ndarray,
    groove: np.ndarray,
    label_hops: int,
) -> int:
    import heapq

    n_chg = 0
    bnd = _boundary_vertex_mask(state.part, edges)
    seeds = np.where(bnd & (groove > 1e-12))[0]
    if seeds.size == 0:
        seeds = np.where(bnd)[0]
    heap: List[Tuple[float, int]] = []
    seen: set = set()
    for v in seeds:
        heapq.heappush(heap, (-float(groove[v]), int(v)))
    while heap:
        _, v = heapq.heappop(heap)
        if v in seen:
            continue
        seen.add(v)
        bnd = _boundary_vertex_mask(state.part, edges)
        if not bnd[v]:
            continue
        if _try_boundary_relabel(state, v, part0, label_hops):
            n_chg += 1
        bnd = _boundary_vertex_mask(state.part, edges)
        for n in state.neighbors[v]:
            if n not in seen and bnd[n]:
                heapq.heappush(heap, (-float(groove[n]), int(n)))
    return n_chg


def relax_partition_boundary(
    vertices: np.ndarray,
    faces: np.ndarray,
    part: np.ndarray,
    part0: np.ndarray,
    lap: sparse.csr_matrix,
    cfg: SkinOptimizeConfig,
    energy_spec: PartitionEnergySpec,
    *,
    edges: Optional[np.ndarray] = None,
    neighbors: Optional[List[List[int]]] = None,
    pose_mats: Optional[np.ndarray] = None,
    w_motion: float = 0.0,
) -> np.ndarray:
    """
    指标驱动边界松弛：体积比 / 表面积比 / 凹陷深度 + Agent 能量权重
    （+ 可选的动画形变能量 ``pose_mats``/``w_motion``）。

    仅更新边界顶点标号，使缝口对齐语义凹槽、比例接近 Agent 目标，且在给定动画
    姿态下边拉伸更小。
    """
    num_v = len(part)
    if edges is None:
        edges = _mesh_edges(faces, num_v)
    if neighbors is None:
        neighbors = _vertex_adjacency(edges, num_v)
    groove = _normalize_groove_depth(vertices, faces, lap)
    vm = _vertex_mass_weights(vertices, faces)
    state = _PartitionMetricState(
        vertices, part.copy(), vm, groove, neighbors, energy_spec,
        pose_mats=pose_mats, w_motion=w_motion,
    )

    iters = max(int(cfg.boundary_partition_iters), 0)
    dj_pass = max(int(cfg.boundary_dijkstra_passes), 0)
    hops = int(cfg.boundary_label_hops)
    part_in = part.copy()
    for it in range(iters):
        bnd = _boundary_vertex_mask(state.part, edges)
        if not np.any(bnd):
            break
        n_chg = 0
        for _ in range(dj_pass):
            n_chg += _relax_boundary_metric_dijkstra(
                state, part0, edges, groove, hops,
            )
        n_chg += _relax_boundary_metric_greedy(
            state, part0, edges, groove, hops,
        )
        if cfg.verbose:
            print(
                f"[part-bnd]  iter {it + 1}/{iters}  "
                f"边界顶点 {int(bnd.sum())}  relabel={n_chg}  "
                f"Δpart={partition_change_ratio(state.part, part_in):.4f}"
            )
        if n_chg == 0:
            break
    return state.part


def anneal_diffuse_hard_weights(
    u0_hard: np.ndarray,
    lap: sparse.csr_matrix,
    cfg: SkinOptimizeConfig,
) -> np.ndarray:
    """先松后紧：多阶段网格扩散 + 递增贴合硬锚，软化硬分块再交给调和优化。"""
    stages = max(int(cfg.anneal_stages), 1)
    u_anchor = np.maximum(np.asarray(u0_hard, dtype=np.float64), 0.0)
    w = u_anchor.copy()
    lo_step = float(cfg.anneal_loose_step)
    hi_step = float(cfg.anneal_tight_step)
    a0 = float(cfg.anneal_anchor_start)
    a1 = float(cfg.anneal_anchor_end)
    for s in range(stages):
        t = (s + 1) / stages
        step = lo_step + (hi_step - lo_step) * t
        anchor = a0 + (a1 - a0) * t
        iters = max(2, int(round(2 + 4 * (1.0 - t))))
        w = _diffuse_weights_on_mesh(w, lap, iters=iters, step=step)
        w = (1.0 - anchor) * w + anchor * u_anchor
        w = _normalize_skin_rows(w)
    return project_skin_weights(w, cfg.max_influences)


def refine_partition(
    part: np.ndarray,
    part0: np.ndarray,
    u0: np.ndarray,
    w_live: np.ndarray,
    cfg: SkinOptimizeConfig,
) -> np.ndarray:
    """旧版：全顶点 argmax 硬分块（``partition_mode=direct`` 时启用）。"""
    num_b = u0.shape[1]
    new_part = part.copy()
    for v in range(len(part)):
        scores = cfg.w_heat * u0[v]
        if cfg.w_partition_live > 0:
            scores = scores + cfg.w_partition_live * w_live[v]
        scores = np.maximum(scores, 0.0)
        if 0 <= int(part0[v]) < num_b:
            scores[int(part0[v])] += cfg.w_part_stick
        new_part[v] = int(np.argmax(scores))
    return new_part


def refine_partition_step(
    skeleton: SemanticSkeleton,
    part: np.ndarray,
    part0: np.ndarray,
    u0: np.ndarray,
    w_live: np.ndarray,
    lap: sparse.csr_matrix,
    cfg: SkinOptimizeConfig,
    energy_spec: PartitionEnergySpec,
    *,
    edges: Optional[np.ndarray] = None,
    neighbors: Optional[List[List[int]]] = None,
    pose_mats: Optional[np.ndarray] = None,
    w_motion: float = 0.0,
) -> np.ndarray:
    """单步分块更新：默认指标+Agent 边界松弛，否则 direct argmax。"""
    mode = (cfg.partition_mode or "metric").lower()
    if mode in ("boundary", "metric"):
        mode = "metric"
    if mode == "direct":
        return refine_partition(part, part0, u0, w_live, cfg)
    verts, faces = skeleton.vertices, skeleton.faces
    one_iter = replace(cfg, boundary_partition_iters=1, boundary_dijkstra_passes=1)
    return relax_partition_boundary(
        verts, faces, part, part0, lap, one_iter, energy_spec,
        edges=edges, neighbors=neighbors,
        pose_mats=pose_mats, w_motion=w_motion,
    )


def partition_change_ratio(part: np.ndarray, part0: np.ndarray) -> float:
    return float(np.mean(part != part0))


# ---------------------------------------------------------------------------
# 体素 / 关节 / GRPO
# ---------------------------------------------------------------------------


def _voxel_centers(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
) -> Tuple[np.ndarray, float]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    extent = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    pitch = max(extent / max(resolution, 8), 1e-9)
    try:
        vg = mesh.voxelized(pitch=pitch)
        centers = np.asarray(vg.points, dtype=np.float64)
        if centers.ndim == 1:
            centers = centers.reshape(0, 3)
        return centers, pitch
    except Exception:
        return np.empty((0, 3)), pitch


def _joint_positions(skeleton: SemanticSkeleton) -> np.ndarray:
    return np.stack(
        [np.asarray(b.joint_position, dtype=np.float64) for b in skeleton.bones],
        axis=0,
    )


def _apply_joint_positions(skeleton: SemanticSkeleton, joints: np.ndarray) -> None:
    for i, b in enumerate(skeleton.bones):
        b.joint_position = joints[i].copy()
    for j in skeleton.joints:
        j.position = joints[j.child_bone].copy()


@dataclass(frozen=True)
class _BoneSample:
    head_idx: int
    tail_idx: int
    t: float


def _bone_samples_from_joints(
    joints: np.ndarray,
    skeleton: SemanticSkeleton,
    samples_per_bone: int,
) -> Tuple[np.ndarray, List[_BoneSample]]:
    meta: List[_BoneSample] = []
    s = max(int(samples_per_bone), 0)
    for b in skeleton.bones:
        p = b.parent
        if p < 0:
            for i in range(s + 1):
                meta.append(_BoneSample(b.index, b.index, 0.0))
        else:
            for i in range(s + 1):
                meta.append(_BoneSample(p, b.index, i / max(s, 1)))
    if not meta:
        return np.zeros((0, 3)), meta
    pts = np.zeros((len(meta), 3), dtype=np.float64)
    for i, m in enumerate(meta):
        h, t = joints[m.head_idx], joints[m.tail_idx]
        pts[i] = (1.0 - m.t) * h + m.t * t
    return pts, meta


def _joint_grad_from_point_grads(
    grad_pts: np.ndarray,
    meta: Sequence[_BoneSample],
    num_bones: int,
) -> np.ndarray:
    g = np.zeros((num_bones, 3), dtype=np.float64)
    for gi, m in zip(grad_pts, meta):
        if m.head_idx == m.tail_idx:
            g[m.tail_idx] += gi
        else:
            g[m.head_idx] += gi * m.t
            g[m.tail_idx] += gi * (1.0 - m.t)
    return g


def _bone_rest_lengths(joints: np.ndarray, skeleton: SemanticSkeleton) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for b in skeleton.bones:
        if b.parent < 0:
            continue
        out[b.index] = max(
            float(np.linalg.norm(joints[b.index] - joints[b.parent])), 1e-9
        )
    return out


def reward_volumetric_joint_coverage(
    voxel_centers: np.ndarray,
    joints: np.ndarray,
    alpha: float = 0.05,
) -> float:
    if voxel_centers.size == 0 or joints.size == 0:
        return 0.0
    d = np.linalg.norm(
        voxel_centers[:, None, :] - joints[None, :, :], axis=2
    ).min(axis=1)
    return float(np.exp(-alpha * d).mean())


def reward_bone_mesh_containment(
    vertices: np.ndarray,
    faces: np.ndarray,
    bone_points: np.ndarray,
) -> float:
    if bone_points.size == 0:
        return 0.0
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        return float(np.asarray(mesh.contains(bone_points), dtype=np.float64).mean())
    except Exception:
        voxel, pitch = _voxel_centers(vertices, faces, 64)
        if voxel.size == 0:
            return 0.0
        d, _ = cKDTree(voxel).query(bone_points, k=1)
        return float((d < pitch * 0.87).mean())


def reward_skinning_coverage_sparsity(
    weights: np.ndarray,
    beta: float = 0.1,
    max_bones: int = 4,
) -> Tuple[float, float, float]:
    w = np.asarray(weights, dtype=np.float64)
    r_z = float(np.all(w < beta, axis=1).mean())
    r_m = float((np.sum(w > beta, axis=1) > max_bones).mean())
    return 1.0 - 0.5 * r_z - 0.5 * r_m, r_z, r_m


def _heat_mse(w: np.ndarray, u0: np.ndarray) -> float:
    return float(np.mean((w - u0) ** 2))


# ---------------------------------------------------------------------------
# R_mo：Deformation Smoothness（LBS 边拉伸）
# ---------------------------------------------------------------------------


def _mesh_edge_rest_lengths(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """(E,2) 边索引与静止边长。"""
    edges = _mesh_edges(faces, len(vertices))
    if len(edges) == 0:
        return edges, np.zeros(0, dtype=np.float64)
    ei, ej = edges[:, 0], edges[:, 1]
    rest = np.linalg.norm(vertices[ei] - vertices[ej], axis=1)
    return edges, np.maximum(rest, 1e-12)


def poses_from_animations(
    animations: Sequence[RigAnimation],
    max_poses: int = 5,
) -> List[Dict[int, float]]:
    """从给定 ``RigAnimation`` 关键帧收集姿态 ``{bone_index: angle_deg}``。"""
    raw: List[Dict[int, float]] = []
    for anim in animations:
        for kf in sorted(anim.keyframes, key=lambda k: k.t):
            if not kf.angles:
                continue
            raw.append({int(b): float(a) for b, a in kf.angles.items()})
    if not raw:
        return []
    if len(raw) <= max(max_poses, 1):
        return raw
    idx = np.linspace(0, len(raw) - 1, max(max_poses, 1), dtype=int)
    return [raw[int(i)] for i in idx]


def _pose_bone_skin_matrices(
    skeleton: SemanticSkeleton,
    poses: Sequence[Dict[int, float]],
) -> np.ndarray:
    """各动画姿态下每根骨的 LBS 蒙皮矩阵，返回 (P, B, 4, 4)。

    用于分块边界松弛时**局部**评估「某顶点改绑到相邻骨后，其关联边在各动画
    姿态下的拉伸如何变化」，从而让边界朝「动画形变最小」的方向移动。
    关节在分块阶段固定，故可一次性预计算后复用。
    """
    n_bones = skeleton.num_bones
    if not poses:
        return np.zeros((0, n_bones, 4, 4), dtype=np.float64)
    mats: List[np.ndarray] = []
    for ang in poses:
        rmap: Dict[int, np.ndarray] = {}
        for bi, a in ang.items():
            if abs(float(a)) < 1e-9:
                continue
            rmap[int(bi)] = axis_angle_matrix(skeleton.bone_axis(int(bi)), float(a))
        mats.append(skeleton.bone_skin_matrices(rmap))
    return np.asarray(mats, dtype=np.float64)


def resolve_mo_poses(
    skeleton: SemanticSkeleton,
    cfg: SkinOptimizeConfig,
    *,
    animations: Optional[Sequence[RigAnimation]] = None,
    mo_poses: Optional[Sequence[Dict[int, float]]] = None,
    test_angle_deg: Optional[float] = None,
) -> List[Dict[int, float]]:
    """
    解析 R_mo 用的给定姿态（优先级：显式 mo_poses > animations > 检验动画）。

    不做随机采样，除非 ``mo_allow_random_fallback=True``。
    """
    if mo_poses is not None:
        out = [
            {int(b): float(a) for b, a in p.items()}
            for p in mo_poses
            if p
        ]
        if out:
            return out

    if animations:
        poses = poses_from_animations(animations, cfg.mo_max_poses)
        if poses:
            return poses

    if test_angle_deg is not None:
        test_anim = make_test_animation(skeleton, angle_deg=float(test_angle_deg))
        poses = poses_from_animations([test_anim], cfg.mo_max_poses)
        if poses:
            return poses

    if cfg.mo_allow_random_fallback:
        return _sample_mo_poses_random(skeleton, cfg)

    raise ValueError(
        "R_mo 需要给定骨骼运动姿态：请传入 animations（RigAnimation 列表）、"
        "mo_poses，或 test_angle_deg（检验弯曲角）。"
    )


def _sample_mo_poses_random(
    skeleton: SemanticSkeleton,
    cfg: SkinOptimizeConfig,
    *,
    seed: int = 0,
) -> List[Dict[int, float]]:
    """仅作兜底：随机姿态（默认关闭）。"""
    rng = np.random.default_rng(seed)
    poses: List[Dict[int, float]] = []
    candidates = [b.index for b in skeleton.bones if b.parent >= 0 and b.face_ids]
    for _ in range(max(cfg.mo_max_poses, 1)):
        ang: Dict[int, float] = {}
        if candidates:
            k = min(len(candidates), max(3, len(candidates) // 3))
            pick = rng.choice(candidates, size=k, replace=False)
            for bi in pick:
                ang[int(bi)] = float(rng.uniform(-22.0, 22.0))
        poses.append(ang)
    return poses


def _subsample_mo_edges(
    edges: np.ndarray,
    rest_lens: np.ndarray,
    max_edges: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_edges <= 0 or len(edges) <= max_edges:
        return edges, rest_lens
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(edges), size=max_edges, replace=False)
    return edges[idx], rest_lens[idx]


def _bones_affecting_poses(poses: Sequence[Dict[int, float]]) -> np.ndarray:
    s: set[int] = set()
    for p in poses:
        s.update(int(b) for b in p.keys())
    return np.array(sorted(s), dtype=np.int64)


def mo_stretch_penalty(
    skeleton: SemanticSkeleton,
    weights: np.ndarray,
    poses: Sequence[Dict[int, float]],
    edges: np.ndarray,
    rest_lens: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    论文 ``E_p max_e max(1, l_deformed/l_rest)`` 的离散近似（对姿态取平均）。
    返回 ``penalty >= 0``，静止无拉伸时为 0。
    """
    if len(edges) == 0 or not poses:
        return 0.0
    old_w = skeleton.weights
    skeleton.weights = np.asarray(weights, dtype=np.float64)
    ei, ej = edges[:, 0], edges[:, 1]
    acc = 0.0
    try:
        for ang in poses:
            vd = skeleton.pose_from_angles(dict(ang))
            def_len = np.linalg.norm(vd[ei] - vd[ej], axis=1)
            ratio = def_len / (rest_lens + eps)
            acc += float(np.max(np.maximum(ratio, 1.0) - 1.0))
        return acc / len(poses)
    finally:
        skeleton.weights = old_w


def reward_deformation_smoothness(
    stretch_penalty: float,
    mo_scale: float,
) -> float:
    """``R_mo = (1 + s * penalty)^{-1}``，``penalty=0`` 时 ``R_mo=1``。"""
    return 1.0 / (1.0 + mo_scale * max(0.0, stretch_penalty))


def _mo_loss_and_joint_grad(
    skeleton: SemanticSkeleton,
    weights: np.ndarray,
    joints: np.ndarray,
    j0: np.ndarray,
    poses: Sequence[Dict[int, float]],
    edges: np.ndarray,
    rest_lens: np.ndarray,
    movable: np.ndarray,
    cfg: SkinOptimizeConfig,
    bbox_diag: float,
) -> Tuple[float, np.ndarray]:
    """``loss_mo = 1 - R_mo`` 及对关节的有限差分梯度（可仅对姿态相关骨做 FD）。"""
    _apply_joint_positions(skeleton, joints)
    pen0 = mo_stretch_penalty(
        skeleton, weights, poses, edges, rest_lens, cfg.mo_eps
    )
    r0 = reward_deformation_smoothness(pen0, cfg.mo_scale)
    loss = 1.0 - r0
    dloss_dpen = cfg.mo_scale / ((1.0 + cfg.mo_scale * pen0) ** 2)

    grad = np.zeros_like(joints)
    eps = max(bbox_diag * cfg.joint_fd_eps_ratio, 1e-6)
    if cfg.mo_fd_pose_bones_only:
        fd_bones = _bones_affecting_poses(poses)
        fd_bones = fd_bones[np.isin(fd_bones, np.where(movable)[0])]
    else:
        fd_bones = np.where(movable)[0]
    for b in fd_bones:
        for d in range(3):
            jp = joints.copy()
            jp[b, d] += eps
            _apply_joint_positions(skeleton, jp)
            pen_p = mo_stretch_penalty(
                skeleton, weights, poses, edges, rest_lens, cfg.mo_eps
            )
            grad[b, d] = dloss_dpen * (pen_p - pen0) / eps
    _apply_joint_positions(skeleton, joints)
    return loss, grad


def _soft_sc_loss(w: np.ndarray, beta: float, max_bones: int) -> float:
    wmax = w.max(axis=1)
    l_cov = float(np.mean(np.maximum(0.0, beta - wmax) ** 2))
    sig = 1.0 / (1.0 + np.exp(-10.0 * (w - beta)))
    excess = np.maximum(0.0, sig.sum(axis=1) - float(max_bones))
    return l_cov + float(np.mean(excess ** 2))


def compute_grpo_rewards(
    skeleton: SemanticSkeleton,
    weights: Optional[np.ndarray] = None,
    *,
    part0: Optional[np.ndarray] = None,
    u0: Optional[np.ndarray] = None,
    mo_poses: Optional[Sequence[Dict[int, float]]] = None,
    mo_edges: Optional[np.ndarray] = None,
    mo_rest_lens: Optional[np.ndarray] = None,
    vox: Optional[np.ndarray] = None,
    cfg: Optional[SkinOptimizeConfig] = None,
) -> SkinRewardReport:
    cfg = cfg or SkinOptimizeConfig()
    w = skeleton.weights if weights is None else weights
    part = skeleton.vertex_bone
    verts, faces = skeleton.vertices, skeleton.faces
    joints = _joint_positions(skeleton)
    if vox is None:
        vox, _ = _voxel_centers(verts, faces, cfg.voxel_resolution)
    bone_pts, _ = _bone_samples_from_joints(joints, skeleton, cfg.bone_samples)

    r_vj = reward_volumetric_joint_coverage(vox, joints, cfg.vj_alpha)
    if cfg.use_voxel_vk and vox.size > 0:
        pitch_est = float(np.linalg.norm(verts.max(0) - verts.min(0)))
        pitch_est = max(pitch_est / max(cfg.voxel_resolution, 8), 1e-9)
        r_vk = float(
            (cKDTree(vox).query(bone_pts, k=1)[0] < pitch_est * 0.87).mean()
        ) if bone_pts.size else 0.0
    else:
        r_vk = reward_bone_mesh_containment(verts, faces, bone_pts)
    r_sc, r_z, r_m = reward_skinning_coverage_sparsity(
        w, cfg.beta, cfg.max_influences
    )

    if mo_edges is None or mo_rest_lens is None:
        mo_edges, mo_rest_lens = _mesh_edge_rest_lengths(verts, faces)
    if mo_poses is None:
        mo_poses = resolve_mo_poses(skeleton, cfg)
    mo_pen = mo_stretch_penalty(
        skeleton, w, mo_poses, mo_edges, mo_rest_lens, cfg.mo_eps
    )
    r_mo = reward_deformation_smoothness(mo_pen, cfg.mo_scale)

    if cfg.report_dirichlet_energy:
        lap = build_mesh_laplacian(
            verts, faces, cotangent=cfg.use_cotangent_laplacian
        )
        lap_e = dirichlet_energy(lap, w)
    else:
        lap_e = 0.0
    hm = _heat_mse(w, u0) if u0 is not None else 0.0
    pcr = (
        partition_change_ratio(part, part0)
        if part0 is not None
        else 0.0
    )
    loss = (
        cfg.w_heat * hm
        + cfg.w_vj * (1.0 - r_vj)
        + cfg.w_vk * (1.0 - r_vk)
        + cfg.w_sc * (1.0 - r_sc)
        + cfg.w_mo * (1.0 - r_mo)
    )
    return SkinRewardReport(
        R_vj=r_vj, R_vk=r_vk, R_sc=r_sc, R_z=r_z, R_m=r_m,
        R_mo=r_mo, mo_stretch=mo_pen,
        laplacian_energy=lap_e, heat_mse=hm,
        partition_change_ratio=pcr, loss_total=loss,
    )


def _vj_soft_loss_grad(
    voxel_centers: np.ndarray,
    joints: np.ndarray,
    alpha: float,
    tau: float,
) -> Tuple[float, np.ndarray]:
    if voxel_centers.size == 0:
        return 0.0, np.zeros_like(joints)
    diff = voxel_centers[:, None, :] - joints[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    logits = -dist / max(tau, 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    sw = np.exp(logits)
    sw /= sw.sum(axis=1, keepdims=True) + 1e-12
    d_soft = (sw * dist).sum(axis=1)
    kern = np.exp(-alpha * d_soft)
    loss = -float(kern.mean())
    coeff = -alpha * kern / max(len(d_soft), 1)
    g_j = np.zeros_like(joints)
    for j in range(joints.shape[0]):
        d_ij = dist[:, j]
        dkern_dd = coeff * sw[:, j]
        valid = d_ij > 1e-10
        if not np.any(valid):
            continue
        direction = (joints[j][None, :] - voxel_centers[valid]) / d_ij[valid, None]
        g_j[j] = (dkern_dd[valid, None] * direction).sum(axis=0)
    return loss, g_j


def _vk_soft_loss_grad(
    voxel_centers: np.ndarray,
    bone_pts: np.ndarray,
    meta: Sequence[_BoneSample],
    sigma: float,
    num_bones: int,
) -> Tuple[float, np.ndarray]:
    if bone_pts.size == 0 or voxel_centers.size == 0:
        return 0.0, np.zeros((num_bones, 3))
    s2 = max(sigma * sigma, 1e-16)
    diff = bone_pts[:, None, :] - voxel_centers[None, :, :]
    d2 = (diff ** 2).sum(axis=2)
    kern = np.exp(-0.5 * d2 / s2)
    score = np.tanh(kern.sum(axis=1))
    loss = -float(score.mean())
    g_pts = np.zeros_like(bone_pts)
    for i in range(len(bone_pts)):
        ki = kern[i]
        if ki.sum() < 1e-12:
            continue
        tanh_s = 1.0 - score[i] ** 2
        g_pts[i] = (tanh_s / len(score)) * (
            (ki[:, None] * (bone_pts[i] - voxel_centers) / s2).sum(axis=0)
        )
    return loss, _joint_grad_from_point_grads(g_pts, meta, num_bones)


def _clamp_joints(
    joints: np.ndarray,
    j0: np.ndarray,
    movable: np.ndarray,
    max_delta: float,
) -> np.ndarray:
    out = joints.copy()
    delta = out - j0
    delta[~movable] = 0.0
    norm = np.linalg.norm(delta, axis=1)
    scale = np.ones_like(norm)
    over = norm > max_delta
    scale[over] = max_delta / norm[over]
    out = j0 + delta * scale[:, None]
    out[~movable] = j0[~movable]
    return out


def _optimize_joints(
    skeleton: SemanticSkeleton,
    j0: np.ndarray,
    w: np.ndarray,
    vox: np.ndarray,
    pitch: float,
    cfg: SkinOptimizeConfig,
    bbox_diag: float,
    mo_poses: Sequence[Dict[int, float]],
    mo_edges: np.ndarray,
    mo_rest_lens: np.ndarray,
) -> None:
    num_b = skeleton.num_bones
    movable = np.ones(num_b, dtype=bool)
    if cfg.fix_root_joints:
        for r in skeleton.roots:
            movable[r] = False
    rest_len = _bone_rest_lengths(j0, skeleton)
    max_delta = bbox_diag * cfg.joint_move_ratio
    vk_sigma = max(pitch, bbox_diag * cfg.vk_sigma_ratio)
    meta: List[_BoneSample] = []
    for b in skeleton.bones:
        p, s = b.parent, cfg.bone_samples
        if p < 0:
            for i in range(s + 1):
                meta.append(_BoneSample(b.index, b.index, 0.0))
        else:
            for i in range(s + 1):
                meta.append(_BoneSample(p, b.index, i / max(s, 1)))

    joints = j0.copy()
    for it in range(cfg.joint_max_iter):
        bone_pts, _ = _bone_samples_from_joints(joints, skeleton, cfg.bone_samples)
        lv, gv = _vj_soft_loss_grad(vox, joints, cfg.vj_alpha, cfg.softmin_tau)
        lk, gk = _vk_soft_loss_grad(vox, bone_pts, meta, vk_sigma, num_b)
        g_reg = 2.0 * (joints - j0) / max(num_b, 1)
        g_len = np.zeros_like(joints)
        l_len = 0.0
        for b in skeleton.bones:
            if b.parent < 0:
                continue
            vec = joints[b.index] - joints[b.parent]
            length = float(np.linalg.norm(vec)) + 1e-12
            err = length - rest_len[b.index]
            l_len += err * err
            if length > 1e-9:
                u = vec / length
                g_len[b.index] += 2.0 * err * u / max(num_b, 1)
                g_len[b.parent] -= 2.0 * err * u / max(num_b, 1)
        gm = np.zeros_like(joints)
        if cfg.mo_fd_interval <= 1 or (it % cfg.mo_fd_interval) == 0:
            _lm, gm = _mo_loss_and_joint_grad(
                skeleton, w, joints, j0, mo_poses, mo_edges, mo_rest_lens,
                movable, cfg, bbox_diag,
            )
        grad = (
            cfg.w_vj * gv
            + cfg.w_vk * gk
            + cfg.w_joint_reg * g_reg
            + cfg.w_bone_length * g_len
            + cfg.w_mo * gm
        )
        grad[~movable] = 0.0
        joints = _clamp_joints(joints - cfg.joint_lr * grad, j0, movable, max_delta)
        _apply_joint_positions(skeleton, joints)
        log_it = (
            it == 0
            or (it + 1) % max(cfg.joint_log_interval, 1) == 0
            or it + 1 == cfg.joint_max_iter
        )
        if cfg.verbose and log_it:
            pen = mo_stretch_penalty(
                skeleton, w, mo_poses, mo_edges, mo_rest_lens, cfg.mo_eps
            )
            r_mo = reward_deformation_smoothness(pen, cfg.mo_scale)
            print(
                f"[joint-opt] {it + 1}/{cfg.joint_max_iter}  "
                f"R_mo={r_mo:.4f}  stretch={pen:.4f}"
            )


# ---------------------------------------------------------------------------
# 硬分块可视化与 HTML 报告
# ---------------------------------------------------------------------------


def partition_face_colors(
    faces: np.ndarray,
    part: np.ndarray,
    num_bones: int,
) -> np.ndarray:
    """由顶点硬标号 ``part`` 给每个三角面上色（面内众数骨）。"""
    pal = np.asarray(_BONE_PALETTE, dtype=np.uint8)
    n_faces = len(faces)
    colors = np.full((n_faces, 3), 210, dtype=np.uint8)
    for fi in range(n_faces):
        tri = faces[fi]
        labels = part[tri]
        valid = (labels >= 0) & (labels < num_bones)
        if not np.any(valid):
            continue
        labels = labels[valid]
        dom = int(np.bincount(labels.astype(np.int64)).argmax())
        colors[fi] = pal[dom % len(pal)]
    return colors


def render_partition_panel(
    skeleton: SemanticSkeleton,
    part: np.ndarray,
    out_path: Path,
    title: str,
    cfg: SkinOptimizeConfig,
) -> Optional[Path]:
    """12 视角拼图：按硬分块（顶点骨标号）对面片上色。"""
    try:
        from rendering.render_component import (
            _compose_panel,
            _render_views_with_color_table,
        )
    except ImportError as exc:
        if cfg.verbose:
            print(f"[report] 跳过分块渲染（缺少依赖）: {exc}")
        return None
    verts = skeleton.vertices
    faces = skeleton.faces
    fc = partition_face_colors(faces, part, skeleton.num_bones)
    sz = int(cfg.partition_report_image_size)
    views = _render_views_with_color_table(
        verts, faces, fc, image_size=(sz, sz),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _compose_panel(
        views,
        output_path=out_path,
        title=title,
        cols=int(cfg.partition_report_panel_cols),
    )


def _partition_legend_html(skeleton: SemanticSkeleton) -> str:
    rows = []
    for b in skeleton.bones:
        rgb = _BONE_PALETTE[b.index % len(_BONE_PALETTE)]
        swatch = (
            f'<span class="swatch" style="background:rgb({rgb[0]},{rgb[1]},{rgb[2]})">'
            f"</span>"
        )
        rows.append(
            f"<li>{swatch} <b>{html_escape(b.name)}</b> "
            f"<span class='muted'>骨 #{b.index}</span></li>"
        )
    return "<ul class='legend'>" + "".join(rows) + "</ul>"


def _partition_metrics_html(
    skeleton: SemanticSkeleton,
    metrics: PartitionGeometryMetrics,
    *,
    title: str,
) -> str:
    rows = []
    for b in skeleton.bones:
        i = b.index
        rows.append(
            f"<tr><td>{html_escape(b.name)}</td>"
            f"<td>{metrics.volume_ratio[i]:.4f}</td>"
            f"<td>{metrics.area_ratio[i]:.4f}</td>"
            f"<td>{metrics.max_concavity_depth[i]:.4f}</td></tr>"
        )
    return (
        f"<h3>{html_escape(title)}</h3>"
        "<table class='metrics'><tr><th>骨段</th><th>体积比</th>"
        "<th>表面积比</th><th>最大凹陷深度</th></tr>"
        + "".join(rows) + "</table>"
    )


def write_skin_optimize_report(
    report_dir: Path,
    skeleton: SemanticSkeleton,
    part0: np.ndarray,
    part_final: np.ndarray,
    metrics: SkinRewardReport,
    cfg: SkinOptimizeConfig,
    *,
    part_boundary: Optional[np.ndarray] = None,
    geom_before: Optional[PartitionGeometryMetrics] = None,
    geom_after: Optional[PartitionGeometryMetrics] = None,
    energy_spec: Optional[PartitionEnergySpec] = None,
) -> SkinRewardReport:
    """渲染硬分块对比图并写入 ``skin_optimize_report.html``。"""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    p_before = report_dir / "partition_before.png"
    p_after = report_dir / "partition_after.png"
    p_bnd: Optional[Path] = None
    if part_boundary is not None and not np.array_equal(part_boundary, part0):
        p_bnd = report_dir / "partition_boundary.png"

    render_partition_panel(
        skeleton, part0, p_before,
        "硬分块 · 优化前（建骨）", cfg,
    )
    render_partition_panel(
        skeleton, part_final, p_after,
        "硬分块 · 优化后（指标驱动边界松弛）", cfg,
    )
    if p_bnd is not None:
        render_partition_panel(
            skeleton, part_boundary, p_bnd,
            "硬分块 · 边界松弛后（蒙皮前）", cfg,
        )

    bone_names = [b.name for b in skeleton.bones]
    snap: Dict[str, object] = {
        "partition_change_ratio": partition_change_ratio(part_final, part0),
        "num_vertices": int(len(part0)),
        "num_bones": int(skeleton.num_bones),
        "bone_names": bone_names,
        "boundary_partition_iters": int(cfg.boundary_partition_iters),
        "partition_mode": str(cfg.partition_mode),
    }
    if geom_before is not None:
        snap["metrics_before"] = geom_before.to_dict(bone_names)
    if geom_after is not None:
        snap["metrics_after"] = geom_after.to_dict(bone_names)
    if energy_spec is not None:
        snap["energy_spec"] = energy_spec.to_dict(bone_names)
    (report_dir / "partition_snapshot.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def _rel(p: Optional[Path]) -> str:
        if p is None or not p.is_file():
            return ""
        try:
            return p.relative_to(report_dir).as_posix()
        except ValueError:
            return p.name

    rb, ra, rbd = _rel(p_before), _rel(p_after), _rel(p_bnd)
    m = metrics
    img_block = []
    if rb:
        img_block.append(
            f'<figure><img src="{html_escape(rb)}" alt="before" />'
            f"<figcaption>优化前（建骨硬分块）</figcaption></figure>"
        )
    if rbd:
        img_block.append(
            f'<figure><img src="{html_escape(rbd)}" alt="boundary" />'
            f"<figcaption>曲率边界松弛后</figcaption></figure>"
        )
    if ra:
        img_block.append(
            f'<figure><img src="{html_escape(ra)}" alt="after" />'
            f"<figcaption>最终硬分块</figcaption></figure>"
        )

    html_path = report_dir / "skin_optimize_report.html"
    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>蒙皮优化报告</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1a1a1a; }}
  h1 {{ font-size: 1.35rem; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  figure {{ margin: 0; max-width: 48%; min-width: 320px; }}
  figure img {{ width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
  figcaption {{ font-size: 0.85rem; color: #555; margin-top: 6px; }}
  table.metrics {{ border-collapse: collapse; margin: 16px 0; }}
  table.metrics td, table.metrics th {{
    border: 1px solid #ddd; padding: 6px 12px; text-align: left;
  }}
  .muted {{ color: #666; font-size: 0.9rem; }}
  .legend {{ list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 8px 20px; }}
  .legend li {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block;
    border: 1px solid rgba(0,0,0,.15); }}
</style>
</head>
<body>
<h1>蒙皮 / 硬分块优化报告</h1>
<p class="muted">流程：Agent 能量 + 体积比/表面积比/凹陷深度边界松弛 → 硬权重退火 → 调和蒙皮（{html_escape(cfg.weight_mode)}）</p>
<p class="muted">能量来源：{html_escape(energy_spec.source if energy_spec else "n/a")}</p>
<div class="grid">{"".join(img_block)}</div>
<h2>分块几何指标</h2>
{(_partition_metrics_html(skeleton, geom_before, title="优化前") if geom_before is not None else "")}
{(_partition_metrics_html(skeleton, geom_after, title="优化后") if geom_after is not None else "")}
<h2>蒙皮指标</h2>
<table class="metrics">
  <tr><th>Δpart（相对建骨）</th><td>{m.partition_change_ratio:.4f}</td></tr>
  <tr><th>heat_mse</th><td>{m.heat_mse:.6f}</td></tr>
  <tr><th>R_vj</th><td>{m.R_vj:.4f}</td></tr>
  <tr><th>R_vk</th><td>{m.R_vk:.4f}</td></tr>
  <tr><th>R_sc</th><td>{m.R_sc:.4f}</td></tr>
  <tr><th>R_mo</th><td>{m.R_mo:.4f}</td></tr>
  <tr><th>mo_stretch</th><td>{m.mo_stretch:.4f}</td></tr>
  <tr><th>loss_total</th><td>{m.loss_total:.4f}</td></tr>
</table>
<h2>骨段图例</h2>
{_partition_legend_html(skeleton)}
</body>
</html>
"""
    html_path.write_text(body, encoding="utf-8")

    return replace(
        metrics,
        report_html=str(html_path.resolve()),
        partition_before_png=str(p_before.resolve()) if p_before.is_file() else None,
        partition_after_png=str(p_after.resolve()) if p_after.is_file() else None,
        partition_boundary_png=(
            str(p_bnd.resolve()) if p_bnd is not None and p_bnd.is_file() else None
        ),
    )


def _total_energy(
    w: np.ndarray,
    u0: np.ndarray,
    part: np.ndarray,
    part0: np.ndarray,
    skeleton: SemanticSkeleton,
    cfg: SkinOptimizeConfig,
    mo_poses: Sequence[Dict[int, float]],
    mo_edges: np.ndarray,
    mo_rest_lens: np.ndarray,
    *,
    vox: Optional[np.ndarray] = None,
    heat_only: bool = False,
) -> float:
    e_part = partition_change_ratio(part, part0)
    hm = _heat_mse(w, u0)
    if heat_only:
        return cfg.w_heat * hm + cfg.w_part_stick * e_part
    rep = compute_grpo_rewards(
        skeleton, w, part0=part0, u0=u0,
        mo_poses=mo_poses, mo_edges=mo_edges, mo_rest_lens=mo_rest_lens,
        vox=vox, cfg=cfg,
    )
    return (
        cfg.w_heat * hm
        + cfg.w_part_stick * e_part
        + cfg.w_vj * (1.0 - rep.R_vj)
        + cfg.w_vk * (1.0 - rep.R_vk)
        + cfg.w_sc * (1.0 - rep.R_sc)
        + cfg.w_mo * (1.0 - rep.R_mo)
    )


def optimize_semantic_skin(
    skeleton: SemanticSkeleton,
    cfg: Optional[SkinOptimizeConfig] = None,
    *,
    animations: Optional[Sequence[RigAnimation]] = None,
    mo_poses: Optional[Sequence[Dict[int, float]]] = None,
    test_angle_deg: Optional[float] = None,
    report_dir: Optional[Path] = None,
    partition_energy_spec: Optional[PartitionEnergySpec] = None,
    vlm_client: Optional[object] = None,
    partition_image_path: Optional[Path] = None,
) -> SkinRewardReport:
    """
    更新蒙皮权重并可选微调关节。

    1. Agent 给定分块能量；体积比/表面积比/凹陷深度边界松弛
    2. 硬权重退火扩散得 u0 → 3. 以 u0 为锚的调和蒙皮 + 四大 GRPO 指标。
    ``animations`` / ``mo_poses`` 仅在与 ``optimize_joints=True`` 时用于 R_mo。
    """
    cfg = cfg or SkinOptimizeConfig()
    if cfg.fast_mode:
        cfg = replace(SkinOptimizeConfig.fast(), verbose=cfg.verbose)
    verts, faces = skeleton.vertices, skeleton.faces
    part0 = skeleton.vertex_bone.copy().astype(np.int64)
    part = part0.copy()
    j0 = _joint_positions(skeleton)

    lap: Optional[sparse.csr_matrix] = None
    dcache: Optional[_DirichletSmoothCache] = None
    mode = (cfg.weight_mode or "smooth").lower()
    use_lap = (
        cfg.dirichlet_smooth
        or mode in ("smooth", "dirichlet")
        or cfg.boundary_partition_iters > 0
        or cfg.anneal_pre_diffuse
        or (cfg.optimize_partition and cfg.partition_refine_iters > 0)
    )
    edges: Optional[np.ndarray] = None
    neighbors: Optional[List[List[int]]] = None
    if use_lap:
        lap = build_mesh_laplacian(
            verts, faces, cotangent=cfg.use_cotangent_laplacian
        )
        if cfg.dirichlet_cache:
            dcache = _DirichletSmoothCache(lap, cfg.dirichlet_anchor)
        edges = _mesh_edges(faces, len(verts))
        neighbors = _vertex_adjacency(edges, len(verts))

    energy_spec: Optional[PartitionEnergySpec] = None
    geom_before: Optional[PartitionGeometryMetrics] = None
    geom_after: Optional[PartitionGeometryMetrics] = None
    if lap is not None and edges is not None:
        energy_spec, geom_before = resolve_partition_energy_spec(
            skeleton, part0, verts, faces, lap, edges, cfg,
            report_dir=report_dir,
            vlm_client=vlm_client,
            image_path=partition_image_path,
            spec_override=partition_energy_spec,
        )

    part_after_boundary: Optional[np.ndarray] = None
    if (
        cfg.boundary_partition_iters > 0
        and lap is not None
        and energy_spec is not None
        and edges is not None
    ):
        part = relax_partition_boundary(
            verts, faces, part, part0, lap, cfg, energy_spec,
            edges=edges, neighbors=neighbors,
        )
        part_after_boundary = part.copy()
        groove = _normalize_groove_depth(verts, faces, lap)
        geom_after = compute_partition_geometry_metrics(
            verts, faces, part, skeleton.num_bones, groove, edges,
        )
        if cfg.verbose:
            print(
                f"[part-metric] 边界松弛  iters={cfg.boundary_partition_iters}  "
                f"能量={energy_spec.source}  "
                f"Δpart={partition_change_ratio(part, part0):.3f}"
            )

    u0_hard = _u0_from_partition(skeleton, part, cfg)
    if cfg.anneal_pre_diffuse and lap is not None and cfg.anneal_stages > 0:
        u0 = anneal_diffuse_hard_weights(u0_hard, lap, cfg)
        if cfg.verbose:
            print(
                f"[anneal] 硬权重退火扩散  stages={cfg.anneal_stages}  "
                f"anchor {cfg.anneal_anchor_start:.2f}→{cfg.anneal_anchor_end:.2f}"
            )
    else:
        u0 = u0_hard
    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    mo_edges, mo_rest_lens = _mesh_edge_rest_lengths(verts, faces)
    mo_edges, mo_rest_lens = _subsample_mo_edges(
        mo_edges, mo_rest_lens, cfg.mo_edge_max,
    )
    vox = np.zeros((0, 3), dtype=np.float64)
    pitch = max(bbox_diag / max(cfg.voxel_resolution, 8), 1e-9)
    if cfg.optimize_joints:
        vox, pitch = _voxel_centers(verts, faces, cfg.voxel_resolution)
    if animations or mo_poses or test_angle_deg is not None:
        mo_poses = resolve_mo_poses(
            skeleton, cfg,
            animations=animations,
            mo_poses=mo_poses,
            test_angle_deg=test_angle_deg,
        )
    else:
        mo_poses = []
    if cfg.verbose:
        names = []
        if animations:
            names = [a.name for a in animations]
        print(
            f"[opt]    蒙皮={cfg.weight_mode}"
            + (f"  R_mo姿态={len(mo_poses)}" if cfg.optimize_joints else "")
            + (f"（动画: {names}）" if names and cfg.optimize_joints else "")
            + (f"  边采样 {len(mo_edges)}" if cfg.mo_edge_max > 0 else "")
            + ("  [fast]" if cfg.fast_mode else "")
        )

    w_build = skeleton.weights.copy()
    w = _compute_skin_weights(skeleton, u0, cfg, lap=lap, dcache=dcache)
    skeleton.weights = w
    w_delta = float(np.mean(np.abs(w - w_build)))
    if cfg.verbose:
        s0 = skin_weight_summary(w_build, thr=cfg.min_influence_weight)
        s1 = skin_weight_summary(w, thr=cfg.min_influence_weight)
        print(
            f"[opt]    建骨→平滑  |Δw|_mean={w_delta:.4f}  "
            f"≥2骨 {s0['frac_ge2_bones']:.1%}→{s1['frac_ge2_bones']:.1%}  "
            f"≥3骨 {s0['frac_ge3_bones']:.1%}→{s1['frac_ge3_bones']:.1%}  "
            f"主骨>0.85 {s0['frac_dominant_gt_0.85']:.1%}→{s1['frac_dominant_gt_0.85']:.1%}"
        )
        if (cfg.weight_mode or "smooth").lower() == "smooth":
            ej0 = weight_edge_discontinuity(w_build, faces)
            ej1 = weight_edge_discontinuity(w, faces)
            print(
                f"[opt]    显著骨(>{cfg.min_influence_weight}): "
                f"mean={s1['mean_bones']:.2f}  cap={cfg.smooth_dominant_cap}  "
                f"边权重跳变 {ej0:.4f}→{ej1:.4f} (越小越连续)"
            )
    before = compute_grpo_rewards(
        skeleton, w, part0=part0, u0=u0,
        mo_poses=mo_poses, mo_edges=mo_edges, mo_rest_lens=mo_rest_lens,
        vox=vox, cfg=cfg,
    )
    if cfg.verbose:
        msg = (
            f"[opt]    指标  heat_mse={before.heat_mse:.5f}  "
            f"R_mo={before.R_mo:.4f}  stretch={before.mo_stretch:.4f}  "
            f"(姿态数={len(mo_poses)})"
        )
        if cfg.report_dirichlet_energy:
            msg += f"  dirichlet={before.laplacian_energy:.4f}"
        print(msg)

    if cfg.optimize_joints:
        for cy in range(max(1, cfg.outer_cycles)):
            if cfg.verbose:
                print(f"[opt]    === 外轮 {cy + 1}/{cfg.outer_cycles} ===")
            w = _compute_skin_weights(skeleton, u0, cfg, lap=lap, dcache=dcache)
            skeleton.weights = w
            _optimize_joints(
                skeleton, j0, w, vox, pitch, cfg, bbox_diag,
                mo_poses, mo_edges, mo_rest_lens,
            )

    if cfg.optimize_partition and cfg.partition_refine_iters > 0:
        if lap is None:
            lap = build_mesh_laplacian(
                verts, faces, cotangent=cfg.use_cotangent_laplacian,
            )
            if cfg.dirichlet_cache and dcache is None:
                dcache = _DirichletSmoothCache(lap, cfg.dirichlet_anchor)
        if edges is None:
            edges = _mesh_edges(faces, len(verts))
            neighbors = _vertex_adjacency(edges, len(verts))
        if energy_spec is None and lap is not None and edges is not None:
            energy_spec, geom_before = resolve_partition_energy_spec(
                skeleton, part0, verts, faces, lap, edges, cfg,
                report_dir=report_dir,
                vlm_client=vlm_client,
                image_path=partition_image_path,
                spec_override=partition_energy_spec,
            )
        # 预计算动画姿态下各骨蒙皮矩阵（关节在分块阶段固定，可复用）
        part_pose_mats: Optional[np.ndarray] = None
        if energy_spec is not None and cfg.w_partition_motion > 0.0 and mo_poses:
            part_pose_mats = _pose_bone_skin_matrices(skeleton, mo_poses)
            if cfg.verbose:
                print(
                    f"[part-opt] 动画形变项启用: 姿态数={len(mo_poses)} "
                    f"w_motion={cfg.w_partition_motion}"
                )
        if energy_spec is not None and lap is not None:
            for cy in range(max(1, cfg.outer_cycles)):
                if cfg.verbose:
                    pm = (cfg.partition_mode or "metric").lower()
                    print(f"[opt]    === 分块轮 {cy + 1} ({pm}) ===")
                heat_only = cfg.partition_energy_heat_only
                for it in range(cfg.partition_refine_iters):
                    skeleton.weights = w
                    part_new = refine_partition_step(
                        skeleton, part, part0, u0, w, lap, cfg, energy_spec,
                        edges=edges, neighbors=neighbors,
                        pose_mats=part_pose_mats,
                        w_motion=cfg.w_partition_motion,
                    )
                    u0_try = u0
                    if cfg.anneal_pre_diffuse and cfg.anneal_stages > 0:
                        u0_try = anneal_diffuse_hard_weights(
                            _u0_from_partition(skeleton, part_new, cfg), lap, cfg,
                        )
                    e_old = _total_energy(
                        w, u0, part, part0, skeleton, cfg,
                        mo_poses, mo_edges, mo_rest_lens, vox=vox, heat_only=heat_only,
                    )
                    w_new = _compute_skin_weights(
                        skeleton, u0_try, cfg, lap=lap, dcache=dcache,
                    )
                    e_new = _total_energy(
                        w_new, u0_try, part_new, part0, skeleton, cfg,
                        mo_poses, mo_edges, mo_rest_lens, vox=vox, heat_only=heat_only,
                    )
                    if e_new <= e_old:
                        part, w, u0 = part_new, w_new, u0_try
                    else:
                        w = _compute_skin_weights(
                            skeleton, u0, cfg, lap=lap, dcache=dcache,
                        )

                    if cfg.verbose and (
                        it == 0
                        or (it + 1) % 4 == 0
                        or it + 1 == cfg.partition_refine_iters
                    ):
                        rep = compute_grpo_rewards(
                            skeleton, w, part0=part0, u0=u0,
                            mo_poses=mo_poses, mo_edges=mo_edges, mo_rest_lens=mo_rest_lens,
                            vox=vox, cfg=cfg,
                        )
                        print(
                            f"[part-opt] {it + 1}/{cfg.partition_refine_iters}  "
                            f"heat_mse={rep.heat_mse:.5f}  dirichlet={rep.laplacian_energy:.4f}  "
                            f"Δpart={rep.partition_change_ratio:.3f}  "
                            f"R_sc={rep.R_sc:.4f}  R_mo={rep.R_mo:.4f}"
                        )

    if lap is not None and edges is not None:
        groove = _normalize_groove_depth(verts, faces, lap)
        geom_after = compute_partition_geometry_metrics(
            verts, faces, part, skeleton.num_bones, groove, edges,
        )

    skeleton.weights = w
    skeleton.vertex_bone = part.astype(np.int64)
    after = compute_grpo_rewards(
        skeleton, w, part0=part0, u0=u0,
        mo_poses=mo_poses, mo_edges=mo_edges, mo_rest_lens=mo_rest_lens,
        vox=vox, cfg=cfg,
    )
    after = replace(
        after,
        partition_change_ratio=partition_change_ratio(part, part0),
    )

    if cfg.write_partition_report and report_dir is not None:
        after = write_skin_optimize_report(
            Path(report_dir),
            skeleton,
            part0,
            part,
            after,
            cfg,
            part_boundary=part_after_boundary,
            geom_before=geom_before,
            geom_after=geom_after,
            energy_spec=energy_spec,
        )
        if cfg.verbose and after.report_html:
            print(f"[report] 硬分块报告: {after.report_html}")

    if cfg.verbose:
        s1 = skin_weight_summary(w, thr=cfg.min_influence_weight)
        print(
            f"[opt]    完成  |Δw|_mean={w_delta:.4f}  "
            f"≥2骨={s1['frac_ge2_bones']:.1%}  ≥3骨={s1['frac_ge3_bones']:.1%}  "
            f"stretch {before.mo_stretch:.4f}→{after.mo_stretch:.4f}  "
            f"Δpart={after.partition_change_ratio:.3f}"
        )
        if not cfg.optimize_joints and w_delta < 1e-5:
            print(
                "[opt]    注意: 权重相对建骨几乎未变，请检查 weight_mode 或加大 smooth_sigma_ratio"
            )

    return after
