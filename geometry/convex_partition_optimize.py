"""
基于三维凸包凹度的面片边界松弛。

在已有 face_labels 上，通过边界面的贪婪交换降低各分块与其凸包的偏离：

    E(S) = max_{v in S} dist(v, ∂CH(S))

整体目标为 sum_k E(S_k)。仅移动边界三角面，保持分块数量不变。
"""

from __future__ import annotations

import heapq
import json
import time
from collections import deque
from dataclasses import dataclass, field
from html import escape as html_escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import ConvexHull
except ImportError as exc:  # pragma: no cover
    raise ImportError("convex_partition_optimize 需要 scipy") from exc

from .partition.common import build_face_adjacency_csr


@dataclass
class ConvexPartitionOptConfig:
    """凸包边界优化超参。"""

    enabled: bool = True
    max_swaps: int = 800
    min_block_faces: int = 20
    # 优化前把同语义不连通区域拆成独立连通块（见 semantic_rig.connected_components_per_part）
    split_connected_components: bool = True
    min_component_faces: int = 20
    # 优化后不再把碎块并回语义标签（避免"删掉"独立连通分块）
    enforce_connectivity: bool = False
    connectivity_max_iter: int = 8
    image_size: int = 512
    panel_cols: int = 4
    verbose: bool = True
    # 若给定，仅允许在 label_group 相同的分块之间交换边界面
    label_group: Optional[np.ndarray] = None
    # True 时保持输入 label 编号（语义部位索引），不压缩为 0..K-1
    preserve_label_ids: bool = False
    num_labels_hint: int = 0
    stage_title: str = "分块"
    # ``planar``：相邻连通块平面截面 + handle_loop（默认）
    # ``greedy_swap``：旧版边界三角面贪婪交换
    boundary_method: str = "planar"
    max_pair_passes: int = 3


@dataclass
class ComponentSplitResult:
    """同语义不连通区域拆分结果（优化在连通块 id 上进行）。"""

    comp_labels: np.ndarray
    comp_part: List[int]
    component_names: List[str]
    num_components: int


@dataclass
class PartitionConcavityMetrics:
    """各分块凸包凹度指标。"""

    energies: np.ndarray
    face_counts: np.ndarray

    @property
    def total_energy(self) -> float:
        return float(self.energies.sum())

    @property
    def active_total_energy(self) -> float:
        """仅对有面的分块求和（日志用，排除空 label 槽位）。"""
        mask = self.face_counts > 0
        return float(self.energies[mask].sum()) if np.any(mask) else 0.0

    def per_label_dict(self) -> Dict[int, Dict[str, float]]:
        out: Dict[int, Dict[str, float]] = {}
        for lab in range(len(self.energies)):
            out[int(lab)] = {
                "concavity_energy": float(self.energies[lab]),
                "num_faces": int(self.face_counts[lab]),
            }
        return out

    def per_part_list(
        self,
        semantic_parts: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        names = list(semantic_parts) if semantic_parts else []
        for lab in range(len(self.energies)):
            if int(self.face_counts[lab]) <= 0:
                continue
            rows.append({
                "index": int(lab),
                "name": (
                    names[lab] if lab < len(names) else str(lab)
                ),
                "num_faces": int(self.face_counts[lab]),
                "concavity_energy": float(self.energies[lab]),
            })
        return rows


@dataclass
class ConvexPartitionOptResult:
    """优化结果。"""

    face_labels: np.ndarray
    metrics_before: PartitionConcavityMetrics
    metrics_after: PartitionConcavityMetrics
    num_swaps: int = 0
    num_faces_changed: int = 0
    num_connectivity_relabels: int = 0
    elapsed_seconds: float = 0.0
    history: List[Dict[str, object]] = field(default_factory=list)
    before_png: Optional[Path] = None
    after_png: Optional[Path] = None
    report_html: Optional[Path] = None
    log_json: Optional[Path] = None
    changed_face_ratio: float = 0.0

    @property
    def energy_delta(self) -> float:
        return (
            self.metrics_after.active_total_energy
            - self.metrics_before.active_total_energy
        )


def _partition_sample_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
) -> np.ndarray:
    """分块采样点：顶点 + 三角面心（开放曲面分块也能捕捉凹坑）。"""
    if face_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    tri = np.asarray(faces[face_ids], dtype=np.int64)
    vtx = np.asarray(vertices[np.unique(tri.reshape(-1))], dtype=np.float64)
    v0 = vertices[tri[:, 0]]
    v1 = vertices[tri[:, 1]]
    v2 = vertices[tri[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    return np.unique(np.vstack([vtx, centroids]), axis=0)


def _max_concavity_depth_scipy(pts: np.ndarray) -> float:
    """
    ACD 风格凹度：落在三维凸包内部的点到凸包表面的最大穿透深度。

    对非水密开放曲面分块，用点云凸包而非 mesh.volume；比 trimesh
    closest_point 更稳定（后者对凸包内点常返回 0）。
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 4:
        return 0.0
    if pts.shape[0] == 4 and np.linalg.matrix_rank(pts - pts[0]) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 0.0

    eq = hull.equations
    norms = np.linalg.norm(eq[:, :3], axis=1)
    norms = np.maximum(norms, 1e-15)
    signed = (pts @ eq[:, :3].T + eq[:, 3]) / norms
    bbox = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    tol = max(1e-9, 1e-7 * bbox)

    inside = np.all(signed <= tol, axis=1)
    if not np.any(inside):
        return 0.0
    # 凸包内点到表面的距离 = 各半空间面的最小穿透深度
    facet_dist = np.where(signed < -tol, -signed, np.inf)
    depth_per_point = np.min(facet_dist, axis=1)
    depth_per_point[~inside] = 0.0
    depth_per_point[~np.isfinite(depth_per_point)] = 0.0
    return float(depth_per_point.max())


def partition_concavity_energy(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
) -> float:
    """单分块凹度：采样点到其三维凸包表面的最大穿透深度。"""
    pts = _partition_sample_points(vertices, faces, face_ids)
    return _max_concavity_depth_scipy(pts)


def partition_concavity_detail(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
) -> Dict[str, float]:
    """返回凹度及诊断信息（报告/日志用）。"""
    pts = _partition_sample_points(vertices, faces, face_ids)
    depth = _max_concavity_depth_scipy(pts)
    bbox = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) if len(pts) else 0.0
    return {
        "concavity_energy": depth,
        "concavity_ratio": depth / max(bbox, 1e-12),
        "num_sample_points": float(len(pts)),
        "bbox_diagonal": bbox,
    }


def compute_partition_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    num_labels: int,
) -> PartitionConcavityMetrics:
    energies = np.zeros(num_labels, dtype=np.float64)
    counts = np.zeros(num_labels, dtype=np.int64)
    for lab in range(num_labels):
        mask = face_labels == lab
        counts[lab] = int(mask.sum())
        if counts[lab] > 0:
            energies[lab] = partition_concavity_energy(
                vertices, faces, np.flatnonzero(mask),
            )
    return PartitionConcavityMetrics(energies, counts)


def split_semantic_into_connected_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    semantic_parts: Sequence[str],
    *,
    min_component_faces: int = 20,
) -> ComponentSplitResult:
    """
    把每个语义部位按共享边切成几何连通块（与 ``semantic_rig`` 建骨同口径）。

    小于 ``min_component_faces`` 的碎块会并入共享边最多的邻接块（去噪，不删语义）。
    """
    from collections import Counter

    from .semantic_rig import connected_components_per_part

    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    comp_of_face, comp_part = connected_components_per_part(
        vertices,
        faces,
        fpi,
        min_component_faces=max(int(min_component_faces), 1),
    )
    comp_labels = comp_of_face.astype(np.int64, copy=True)
    names: List[str] = []
    totals = Counter(comp_part)
    per_sem_seen: Counter = Counter()
    for c, sem_idx in enumerate(comp_part):
        base = (
            semantic_parts[sem_idx]
            if 0 <= sem_idx < len(semantic_parts)
            else f"part_{sem_idx}"
        )
        if totals[sem_idx] > 1:
            names.append(f"{base}#{per_sem_seen[sem_idx]}")
            per_sem_seen[sem_idx] += 1
        else:
            names.append(base)
    valid = comp_labels[comp_labels >= 0]
    num_components = int(valid.max()) + 1 if valid.size else 0
    return ComponentSplitResult(
        comp_labels=comp_labels,
        comp_part=list(comp_part),
        component_names=names,
        num_components=num_components,
    )


def map_component_labels_to_semantic(
    comp_labels: np.ndarray,
    comp_part: Sequence[int],
    fallback_semantic: np.ndarray,
) -> np.ndarray:
    """连通块 label → 语义部位 index（未标注面沿用 fallback）。"""
    out = np.asarray(fallback_semantic, dtype=np.int64).copy()
    for f in range(len(comp_labels)):
        c = int(comp_labels[f])
        if 0 <= c < len(comp_part):
            out[f] = int(comp_part[c])
    return out


def _num_labels_for(face_labels: np.ndarray, *, hint: int = 0) -> int:
    valid = np.asarray(face_labels, dtype=np.int64)
    valid = valid[valid >= 0]
    if valid.size == 0:
        return max(int(hint), 0)
    return max(int(valid.max()) + 1, int(hint))


def compute_partition_metrics_detailed(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    num_labels: int,
    *,
    semantic_parts: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """各分块凹度 + 诊断字段。"""
    names = list(semantic_parts) if semantic_parts else []
    rows: List[Dict[str, object]] = []
    for lab in range(num_labels):
        mask = face_labels == lab
        n_faces = int(mask.sum())
        if n_faces <= 0:
            continue
        detail = partition_concavity_detail(
            vertices, faces, np.flatnonzero(mask),
        )
        rows.append({
            "index": int(lab),
            "name": names[lab] if lab < len(names) else str(lab),
            "num_faces": n_faces,
            "concavity_energy": float(detail["concavity_energy"]),
            "concavity_ratio": float(detail["concavity_ratio"]),
            "num_sample_points": int(detail["num_sample_points"]),
            "bbox_diagonal": float(detail["bbox_diagonal"]),
        })
    return rows


def _metrics_table_html(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    num_labels: int,
    semantic_parts: Optional[Sequence[str]],
) -> str:
    details = compute_partition_metrics_detailed(
        vertices,
        faces,
        np.asarray(face_labels, dtype=np.int64),
        num_labels,
        semantic_parts=semantic_parts,
    )
    if not details:
        return "<tr><td colspan='5' class='muted'>无有效分块</td></tr>"
    rows = []
    for d in details:
        name = html_escape(str(d["name"]))
        depth = float(d["concavity_energy"])
        ratio = float(d["concavity_ratio"])
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td>{int(d['num_faces'])}</td>"
            f"<td>{depth:.6g}</td>"
            f"<td>{100 * ratio:.3f}%</td>"
            f"<td>{int(d['num_sample_points'])}</td></tr>"
        )
    return "".join(rows)


def _label_face_sets(face_labels: np.ndarray, num_labels: int) -> List[set]:
    sets: List[set] = [set() for _ in range(num_labels)]
    for f, lab in enumerate(face_labels):
        if 0 <= int(lab) < num_labels:
            sets[int(lab)].add(int(f))
    return sets


def _is_connected_face_set(
    seed: int,
    face_set: set,
    neighbors: List[List[int]],
) -> bool:
    if not face_set:
        return True
    if seed not in face_set:
        return False
    seen = {seed}
    q: deque = deque([seed])
    while q:
        f = q.popleft()
        for nb in neighbors[f]:
            if nb in face_set and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(face_set)


def _swap_stays_connected(
    face_id: int,
    old_label: int,
    new_label: int,
    label_sets: List[set],
    neighbors: List[List[int]],
    min_block_faces: int,
) -> bool:
    old_set = label_sets[old_label]
    if face_id not in old_set:
        return False
    remaining = set(old_set)
    remaining.discard(face_id)
    if len(remaining) < min_block_faces:
        return False
    if not remaining:
        return True
    seed = next(iter(remaining))
    return _is_connected_face_set(seed, remaining, neighbors)


def _neighbor_labels(
    face_id: int,
    face_labels: np.ndarray,
    neighbors: List[List[int]],
) -> List[int]:
    own = int(face_labels[face_id])
    if own < 0:
        return []
    nbrs: set = set()
    for nb in neighbors[face_id]:
        lab = int(face_labels[nb])
        if lab >= 0 and lab != own:
            nbrs.add(lab)
    return sorted(nbrs)


def _relabel_delta_energy(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    label_sets: List[set],
    energies: np.ndarray,
    face_id: int,
    new_label: int,
) -> float:
    old_label = int(face_labels[face_id])
    if new_label == old_label:
        return 0.0
    old_set = label_sets[old_label]
    new_set = label_sets[new_label]
    e_old_before = float(energies[old_label])
    e_new_before = float(energies[new_label])
    old_after_ids = np.array(sorted(old_set - {face_id}), dtype=np.int64)
    new_after_ids = np.array(sorted(new_set | {face_id}), dtype=np.int64)
    e_old_after = partition_concavity_energy(vertices, faces, old_after_ids)
    e_new_after = partition_concavity_energy(vertices, faces, new_after_ids)
    return (e_old_after + e_new_after) - (e_old_before + e_new_before)


def optimize_face_partition_convex_hull(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    cfg: Optional[ConvexPartitionOptConfig] = None,
) -> ConvexPartitionOptResult:
    """
    凸包边界优化入口。

    ``boundary_method=planar``（默认）：相邻连通分块平面截面 + handle_loop。
    ``boundary_method=handle_loop``：仅 handle_loop / 凹缝拉直，不做平面与凸包能量优化。
    ``boundary_method=greedy_swap``：旧版边界三角面贪婪交换。
    """
    cfg = cfg or ConvexPartitionOptConfig()
    method = str(cfg.boundary_method or "planar").strip().lower()
    if method in ("planar", "planar_boundary", "plane", "handle_loop", "handle-loop", "handleloop"):
        from .planar_boundary_optimize import (
            PlanarBoundaryOptConfig,
            optimize_face_partition_planar_boundary,
        )

        handle_loop_only = method in ("handle_loop", "handle-loop", "handleloop")
        pcfg = PlanarBoundaryOptConfig(
            max_pair_passes=max(int(cfg.max_pair_passes), 1),
            min_block_faces=max(int(cfg.min_block_faces), 1),
            verbose=cfg.verbose,
            handle_loop_only=handle_loop_only,
        )
        return optimize_face_partition_planar_boundary(
            vertices, faces, face_labels, cfg, planar_cfg=pcfg,
        )
    return _optimize_face_partition_greedy_swap(
        vertices, faces, face_labels, cfg,
    )


def _optimize_face_partition_greedy_swap(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    cfg: Optional[ConvexPartitionOptConfig] = None,
) -> ConvexPartitionOptResult:
    """
    基于优先队列的边界三角面交换，降低各分块凸包凹度之和。

    返回优化后的 face_labels（与输入等长，label 值域不变）。
    """
    t0 = time.perf_counter()
    cfg = cfg or ConvexPartitionOptConfig()
    labels = np.asarray(face_labels, dtype=np.int64).copy()
    n_faces = len(labels)
    if n_faces == 0:
        empty = PartitionConcavityMetrics(
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )
        return ConvexPartitionOptResult(
            labels, empty, empty, elapsed_seconds=0.0,
        )

    unique = sorted({int(x) for x in labels if int(x) >= 0})
    if len(unique) < 2:
        metrics = compute_partition_metrics(
            vertices, faces, labels, max(len(unique), 1),
        )
        return ConvexPartitionOptResult(
            np.asarray(face_labels, dtype=np.int64).copy(),
            metrics,
            metrics,
            elapsed_seconds=time.perf_counter() - t0,
        )
    if cfg.preserve_label_ids:
        num_labels = max(int(max(unique)) + 1, int(cfg.num_labels_hint))
        label_map = {i: i for i in unique}
    else:
        label_map = {old: i for i, old in enumerate(unique)}
        num_labels = len(unique)
        for i in range(n_faces):
            if int(labels[i]) >= 0:
                labels[i] = label_map[int(labels[i])]

    metrics_before = compute_partition_metrics(
        vertices, faces, labels, num_labels,
    )
    label_sets = _label_face_sets(labels, num_labels)
    energies = metrics_before.energies.copy()

    m = int(len(faces))
    adj = build_face_adjacency_csr(faces)
    neighbors: List[List[int]] = [[] for _ in range(m)]
    for f in range(m):
        row = adj.getrow(f)
        neighbors[f] = row.indices.tolist()

    history: List[Dict[str, object]] = []
    heap: List[Tuple[float, int, int, int]] = []
    counter = 0
    max_swaps = max(int(cfg.max_swaps), 0)
    min_faces = max(int(cfg.min_block_faces), 1)

    label_group = (
        np.asarray(cfg.label_group, dtype=np.int64)
        if cfg.label_group is not None
        else None
    )

    def _swap_group_ok(old_label: int, new_label: int) -> bool:
        if label_group is None:
            return True
        if old_label < 0 or new_label < 0:
            return False
        if old_label >= len(label_group) or new_label >= len(label_group):
            return False
        return int(label_group[old_label]) == int(label_group[new_label])

    def push_boundary_candidates(face_id: int) -> None:
        nonlocal counter
        if int(labels[face_id]) < 0:
            return
        old_lab = int(labels[face_id])
        for new_lab in _neighbor_labels(face_id, labels, neighbors):
            if not _swap_group_ok(old_lab, new_lab):
                continue
            if not _swap_stays_connected(
                face_id, old_lab, new_lab,
                label_sets, neighbors, min_faces,
            ):
                continue
            dE = _relabel_delta_energy(
                vertices, faces, labels, label_sets, energies,
                face_id, new_lab,
            )
            if dE < -1e-9:
                heapq.heappush(heap, (dE, counter, face_id, new_lab))
                counter += 1

    def is_boundary(face_id: int) -> bool:
        own = int(labels[face_id])
        if own < 0:
            return False
        return any(
            int(labels[nb]) >= 0 and int(labels[nb]) != own
            for nb in neighbors[face_id]
        )

    for f in range(n_faces):
        if is_boundary(f):
            push_boundary_candidates(f)

    num_swaps = 0
    while heap and num_swaps < max_swaps:
        dE_est, _, face_id, new_label = heapq.heappop(heap)
        old_label = int(labels[face_id])
        if old_label == new_label:
            continue
        if not _swap_group_ok(old_label, new_label):
            continue
        if not _swap_stays_connected(
            face_id, old_label, new_label,
            label_sets, neighbors, min_faces,
        ):
            continue
        dE = _relabel_delta_energy(
            vertices, faces, labels, label_sets, energies,
            face_id, new_label,
        )
        if dE >= -1e-9:
            continue

        old_set = label_sets[old_label]
        new_set = label_sets[new_label]
        old_after_ids = np.array(sorted(old_set - {face_id}), dtype=np.int64)
        new_after_ids = np.array(sorted(new_set | {face_id}), dtype=np.int64)
        e_old_after = partition_concavity_energy(vertices, faces, old_after_ids)
        e_new_after = partition_concavity_energy(vertices, faces, new_after_ids)

        old_set.discard(face_id)
        new_set.add(face_id)
        labels[face_id] = int(new_label)
        energies[old_label] = e_old_after
        energies[new_label] = e_new_after
        num_swaps += 1
        history.append({
            "swap": num_swaps,
            "face_id": int(face_id),
            "from_label": int(old_label),
            "to_label": int(new_label),
            "delta_energy": float(dE),
            "total_energy": float(energies.sum()),
        })
        if cfg.verbose and (num_swaps <= 5 or num_swaps % 50 == 0):
            print(
                f"[convex-opt] swap {num_swaps}: face {face_id} "
                f"{old_label}→{new_label}  ΔE={dE:.6f}  "
                f"E_total={energies.sum():.6f}"
            )

        affected = {face_id}
        for nb in neighbors[face_id]:
            if is_boundary(nb) or any(
                int(labels[n2]) != int(labels[nb])
                for n2 in neighbors[nb]
            ):
                affected.add(nb)
            for n2 in neighbors[nb]:
                if is_boundary(n2):
                    affected.add(n2)
        for f in affected:
            if is_boundary(f):
                push_boundary_candidates(f)

    labels_pre_connectivity = labels.copy()
    if cfg.enforce_connectivity:
        from .partition.common import enforce_per_label_face_connectivity

        labels = enforce_per_label_face_connectivity(
            faces, labels, max_iter=cfg.connectivity_max_iter,
        )
        label_sets = _label_face_sets(labels, num_labels)

    if cfg.preserve_label_ids:
        labels_out = labels.copy()
    else:
        inv_map = {i: old for old, i in label_map.items()}
        labels_out = np.asarray(face_labels, dtype=np.int64).copy()
        for i in range(n_faces):
            lab = int(labels[i])
            if lab >= 0:
                labels_out[i] = inv_map[lab]

    metrics_after = compute_partition_metrics(
        vertices, faces, labels, num_labels,
    )

    labels_in = np.asarray(face_labels, dtype=np.int64)
    compare_mask = labels_in >= 0
    num_connectivity_relabels = int(
        np.sum((labels != labels_pre_connectivity) & compare_mask)
    ) if cfg.enforce_connectivity else 0
    num_faces_changed = int(np.sum(labels_out != labels_in))
    if compare_mask.any():
        changed_ratio = float(
            np.mean(labels_out[compare_mask] != labels_in[compare_mask])
        )
    else:
        changed_ratio = float(np.mean(labels_out != labels_in))

    result = ConvexPartitionOptResult(
        face_labels=labels_out,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        num_swaps=num_swaps,
        num_faces_changed=num_faces_changed,
        num_connectivity_relabels=num_connectivity_relabels,
        elapsed_seconds=time.perf_counter() - t0,
        history=history,
        changed_face_ratio=changed_ratio,
    )
    if cfg.verbose:
        print(
            f"[convex-opt] 完成: swaps={num_swaps}  "
            f"connectivity_relabels={num_connectivity_relabels}  "
            f"faces_changed={num_faces_changed}  "
            f"E_active {metrics_before.active_total_energy:.6f} → "
            f"{metrics_after.active_total_energy:.6f}  "
            f"(Δ={result.energy_delta:.6f})  "
            f"changed_faces={100*changed_ratio:.2f}%  "
            f"{result.elapsed_seconds:.1f}s"
        )
    return result


def _render_partition_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    out_path: Path,
    title: str,
    cfg: ConvexPartitionOptConfig,
    *,
    semantic_parts: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    try:
        if semantic_parts:
            from rendering.render_component import render_full_coverage_panel
            return render_full_coverage_panel(
                vertices,
                faces,
                list(face_labels),
                list(semantic_parts),
                out_path,
                title=title,
                image_size=cfg.image_size,
                cols=cfg.panel_cols,
            )
        from rendering.render_component import render_initial_segmentation_panel
        return render_initial_segmentation_panel(
            vertices,
            faces,
            list(face_labels),
            out_path,
            title=title,
            image_size=cfg.image_size,
            cols=cfg.panel_cols,
        )
    except ImportError:
        return None
    except Exception:
        return None


def _enrich_history(
    history: Sequence[Dict[str, object]],
    label_names: Optional[Sequence[str]],
) -> List[Dict[str, object]]:
    names = list(label_names) if label_names else []

    def _name(idx: object) -> str:
        i = int(idx)
        return names[i] if 0 <= i < len(names) else str(i)

    out: List[Dict[str, object]] = []
    for row in history:
        item = dict(row)
        if "from_label" in item:
            item["from_name"] = _name(item["from_label"])
        if "to_label" in item:
            item["to_name"] = _name(item["to_label"])
        out.append(item)
    return out


def write_convex_optimize_report(
    output_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels_before: Sequence[int],
    opt_result: ConvexPartitionOptResult,
    cfg: ConvexPartitionOptConfig,
    *,
    semantic_parts: Optional[Sequence[str]] = None,
    display_labels_before: Optional[Sequence[int]] = None,
    display_labels_after: Optional[Sequence[int]] = None,
    component_names: Optional[Sequence[str]] = None,
    comp_labels_split: Optional[Sequence[int]] = None,
    split_meta: Optional[Dict[str, object]] = None,
) -> ConvexPartitionOptResult:
    """写 PNG 对比图、JSON 日志与 ``convex_optimize_report.html``。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_png = output_dir / "convex_optimize_before.png"
    after_png = output_dir / "convex_optimize_after.png"
    split_png = output_dir / "convex_optimize_split.png"
    stage = cfg.stage_title or "分块"
    labels_for_before_png = (
        list(display_labels_before)
        if display_labels_before is not None
        else list(face_labels_before)
    )
    labels_for_after_png = (
        list(display_labels_after)
        if display_labels_after is not None
        else list(opt_result.face_labels)
    )
    opt_result.before_png = _render_partition_panel(
        vertices, faces, labels_for_before_png, before_png,
        f"凸包优化前 · 语义部位", cfg, semantic_parts=semantic_parts,
    )
    opt_result.after_png = _render_partition_panel(
        vertices, faces, labels_for_after_png, after_png,
        f"凸包优化后 · 语义部位", cfg, semantic_parts=semantic_parts,
    )
    split_png_path: Optional[Path] = None
    if comp_labels_split is not None:
        split_png_path = _render_partition_panel(
            vertices, faces, list(comp_labels_split), split_png,
            f"连通块拆分（优化单元）· {len(component_names or [])} 块",
            cfg, semantic_parts=None,
        )

    log_path = output_dir / "convex_optimize_log.json"
    log_payload = {
        "stage": cfg.stage_title,
        "num_swaps": opt_result.num_swaps,
        "num_faces_changed": opt_result.num_faces_changed,
        "num_connectivity_relabels": opt_result.num_connectivity_relabels,
        "elapsed_seconds": opt_result.elapsed_seconds,
        "energy_before_active": opt_result.metrics_before.active_total_energy,
        "energy_after_active": opt_result.metrics_after.active_total_energy,
        "energy_delta_active": opt_result.energy_delta,
        "changed_face_ratio": opt_result.changed_face_ratio,
        "concavity_method": (
            "boundary_handle_loop"
            if str(cfg.boundary_method or "planar").lower()
            in ("handle_loop", "handle-loop", "handleloop")
            else (
                "planar_boundary_handle_loop"
                if str(cfg.boundary_method or "planar").lower()
                not in ("greedy_swap", "greedy", "swap")
                else "scipy_convex_hull_greedy_swap"
            )
        ),
        "component_split": split_meta or {},
        "components_before": compute_partition_metrics_detailed(
            vertices, faces,
            np.asarray(face_labels_before, dtype=np.int64),
            len(opt_result.metrics_before.energies),
            semantic_parts=component_names,
        ),
        "components_after": compute_partition_metrics_detailed(
            vertices, faces,
            np.asarray(opt_result.face_labels, dtype=np.int64),
            len(opt_result.metrics_after.energies),
            semantic_parts=component_names,
        ),
        "history": _enrich_history(
            opt_result.history[-200:],
            component_names or semantic_parts,
        ),
    }
    log_path.write_text(
        json.dumps(log_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    opt_result.log_json = log_path

    n_labels = len(opt_result.metrics_before.energies)
    metric_names = list(component_names) if component_names else list(semantic_parts or [])
    metrics_before_html = _metrics_table_html(
        vertices, faces, face_labels_before, n_labels, metric_names,
    )
    metrics_after_html = _metrics_table_html(
        vertices, faces, opt_result.face_labels, n_labels, metric_names,
    )

    split_meta_html = ""
    if split_meta:
        split_meta_html = (
            f"<p class='muted'>优化前拆分：{split_meta.get('num_components', 0)} 个连通块"
            f"（同语义不连通区域独立优化，优化后仍映射回原语义，不合并删除）。</p>"
        )
    split_img_html = ""
    if split_png_path is not None:
        split_img_html = (
            f'<figure style="margin:12px 0">'
            f'<img src="{html_escape(split_png.name)}" alt="split" '
            f'style="max-width:480px;border:1px solid #d2d2d7;border-radius:8px"/>'
            f'<figcaption class="muted">连通块拆分（每个随机色 = 一个凸包优化单元）</figcaption>'
            f"</figure>"
        )

    before_rel = before_png.name if opt_result.before_png else ""
    after_rel = after_png.name if opt_result.after_png else ""
    img_row = ""
    if before_rel and after_rel:
        img_row = f"""
<div class="img-row">
  <figure><img src="{html_escape(before_rel)}" alt="before"/><figcaption>优化前</figcaption></figure>
  <figure><img src="{html_escape(after_rel)}" alt="after"/><figcaption>优化后</figcaption></figure>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>凸包边界优化报告</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft Yahei", sans-serif;
         background:#f5f5f7; color:#1d1d1f; margin:0; padding:24px; }}
  h1 {{ font-size:22px; margin:0 0 12px 0; }}
  h2 {{ font-size:16px; margin:20px 0 8px 0; }}
  .cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:12px 0; }}
  .card {{ background:#fff; border:1px solid #d2d2d7; border-radius:10px;
           padding:12px 16px; min-width:160px; }}
  .card b {{ display:block; color:#6e6e73; font-size:12px; margin-bottom:4px; }}
  .card span {{ font-size:18px; font-weight:600; }}
  table {{ border-collapse:collapse; background:#fff; width:100%;
            border:1px solid #d2d2d7; border-radius:8px; overflow:hidden; }}
  th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #eee; }}
  th {{ background:#fafafa; font-size:13px; }}
  .img-row {{ display:flex; gap:16px; flex-wrap:wrap; margin:16px 0; }}
  .img-row figure {{ margin:0; background:#fff; border:1px solid #d2d2d7;
                     border-radius:10px; padding:10px; }}
  .img-row img {{ max-width:480px; height:auto; display:block; }}
  figcaption {{ color:#6e6e73; font-size:12px; margin-top:6px; text-align:center; }}
  .muted {{ color:#6e6e73; font-size:13px; }}
  .good {{ color:#0a6c2c; }}
</style>
</head>
<body>
<h1>凸包边界优化（{html_escape(stage)}）</h1>
<p class="muted">开放曲面分块（非水密）使用点云凸包穿透深度衡量凹度：顶点+面心采样，
凹度 = 落在三维凸包内部的点到凸包表面的最大距离；concavity% 相对分块包围盒对角线。
优化在连通块上进行，最终仍写回语义部位着色。</p>
{split_meta_html}
{split_img_html}

<div class="cards">
  <div class="card"><b>边界调整步数</b><span>{opt_result.num_swaps}</span></div>
  <div class="card"><b>变更面数</b><span>{opt_result.num_faces_changed}</span></div>
  <div class="card"><b>连通性修正</b><span>{opt_result.num_connectivity_relabels}</span></div>
  <div class="card"><b>活跃部位凹度能量</b>
    <span>{opt_result.metrics_before.active_total_energy:.4f} → {opt_result.metrics_after.active_total_energy:.4f}</span></div>
  <div class="card"><b>ΔE</b>
    <span class="{'good' if opt_result.energy_delta < 0 else ''}">{opt_result.energy_delta:+.6f}</span></div>
  <div class="card"><b>变更面比例</b><span>{100*opt_result.changed_face_ratio:.2f}%</span></div>
  <div class="card"><b>耗时</b><span>{opt_result.elapsed_seconds:.1f}s</span></div>
</div>

{img_row}

<h2>各连通块凹度（优化前）</h2>
<table><tr><th>连通块</th><th>faces</th><th>concavity</th><th>concavity%</th><th>采样点</th></tr>
{metrics_before_html}</table>

<h2>各连通块凹度（优化后）</h2>
<table><tr><th>连通块</th><th>faces</th><th>concavity</th><th>concavity%</th><th>采样点</th></tr>
{metrics_after_html}</table>

<p class="muted">详细交换日志见 <code>convex_optimize_log.json</code>（最近 {min(200, len(opt_result.history))} 步）。</p>
</body>
</html>"""

    report_path = output_dir / "convex_optimize_report.html"
    report_path.write_text(html, encoding="utf-8")
    opt_result.report_html = report_path
    return opt_result


def blocks_to_face_labels(
    n_faces: int,
    blocks: Sequence[object],
) -> np.ndarray:
    """把 SemanticBlock 列表转为稠密 face_labels（按 block 枚举顺序）。"""
    labels = np.full(n_faces, -1, dtype=np.int64)
    for idx, block in enumerate(blocks):
        for fid in block.face_ids:
            labels[int(fid)] = idx
    return labels


def face_labels_to_blocks(
    face_labels: Sequence[int],
    *,
    id_prefix: str = "C",
    min_block_size: int = 1,
):
    """把 face_labels 转回 SemanticBlock 列表（复用 blocks.build_initial_blocks）。"""
    from .blocks import build_initial_blocks

    return build_initial_blocks(
        face_labels, min_block_size=min_block_size, id_prefix=id_prefix,
    )
