"""
相邻连通分块对的平面截面 + handle_loop 边界优化。

流程（每个相邻分块对 A|B）：
1. 理想状态下不连通的分块（如四肢四块）各自视作独立优化单元（输入为连通块 label）。
2. 取交界带顶点拟合平面 n·x + d = 0。
3. 沿法向/小角度旋转微调平面，使 E(A)+E(B) 凸包凹度下降（有限差分近似梯度）。
4. 按微调平面在交界带重划截面。
5. 在截面带用 handle_loop / 凹缝最短路拉直分界线。
6. 仅当两侧分块仍连通且交界拓扑圈数不变（同胚）时接受改动。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .convex_partition_optimize import (
    ConvexPartitionOptConfig,
    ConvexPartitionOptResult,
    PartitionConcavityMetrics,
    _is_connected_face_set,
    _label_face_sets,
    compute_partition_metrics,
    partition_concavity_energy,
)
from .partition.common import build_face_adjacency_csr, face_centroids_and_normals
from .topology_repair.concave_seam import find_concave_seam_between_seeds
from .topology_repair.handle_loop import find_handle_loop_cycle
from .topology_repair.local_region import expand_face_rings, semantic_boundary_faces


@dataclass
class PlanarBoundaryOptConfig:
    """平面截面边界优化参数。"""

    max_pair_passes: int = 3
    plane_line_search_steps: int = 13
    plane_offset_span: float = 0.15
    plane_normal_step_deg: float = 4.0
    plane_normal_iters: int = 4
    boundary_band_rings: int = 2
    min_block_faces: int = 8
    verbose: bool = True
    # True：仅沿 handle_loop / 凹缝拉直交界，不做平面拟合与凸包能量优化
    handle_loop_only: bool = False


@dataclass
class PairOptRecord:
    part_a: int
    part_b: int
    plane_normal: List[float]
    plane_offset: float
    energy_before: float
    energy_after: float
    boundary_loops_before: int
    boundary_loops_after: int
    faces_changed: int
    accepted: bool
    notes: str = ""


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """PCA 拟合平面，返回单位法向 n 与 d（n·x+d=0）。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        c = pts.mean(axis=0) if len(pts) else np.zeros(3)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), float(-c[2])
    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    normal = _normalize(vh[-1])
    d = float(-np.dot(normal, center))
    return normal, d


def _rotate_normal(
    normal: np.ndarray,
    axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    axis = _normalize(axis)
    n = _normalize(normal)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return _normalize(n * c + np.cross(axis, n) * s + axis * np.dot(axis, n) * (1.0 - c))


def _boundary_vertex_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_faces: Sequence[int],
) -> np.ndarray:
    if not boundary_faces:
        return np.zeros((0, 3), dtype=np.float64)
    tri = np.asarray(faces[np.asarray(boundary_faces, dtype=np.int64)], dtype=np.int64)
    vids = np.unique(tri.reshape(-1))
    return np.asarray(vertices[vids], dtype=np.float64)


def _boundary_loop_count(
    faces: np.ndarray,
    labels: np.ndarray,
    part_a: int,
    part_b: int,
) -> int:
    """A|B 交界边在网格上的连通分量数（同胚检查）。"""
    fpi = np.asarray(labels, dtype=np.int64)
    m = int(len(faces))
    adj = build_face_adjacency_csr(faces)
    indptr, indices = adj.indptr, adj.indices

    edge_to_sides: Dict[Tuple[int, int], Set[int]] = {}
    for fi in range(m):
        la = int(fpi[fi])
        if la not in (part_a, part_b):
            continue
        tri = faces[fi]
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(min(u, v)), int(max(u, v))
            edge_to_sides.setdefault((a, b), set()).add(la)

    cross_edges: List[Tuple[int, int]] = []
    for (a, b), sides in edge_to_sides.items():
        if part_a in sides and part_b in sides:
            cross_edges.append((a, b))

    if not cross_edges:
        return 0

    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in cross_edges:
        union(a, b)

    roots = {find(a) for a, _ in cross_edges}
    roots.update(find(b) for _, b in cross_edges)
    return len(roots)


def _component_connected(
    label: int,
    label_sets: List[set],
    neighbors: List[List[int]],
    min_faces: int,
) -> bool:
    s = label_sets[label]
    if len(s) < min_faces:
        return False
    if not s:
        return True
    seed = next(iter(s))
    return _is_connected_face_set(seed, s, neighbors)


def _pair_energy(
    vertices: np.ndarray,
    faces: np.ndarray,
    label_sets: List[set],
    lab_a: int,
    lab_b: int,
) -> float:
    ids_a = np.array(sorted(label_sets[lab_a]), dtype=np.int64)
    ids_b = np.array(sorted(label_sets[lab_b]), dtype=np.int64)
    if ids_a.size == 0 or ids_b.size == 0:
        return 0.0
    return partition_concavity_energy(vertices, faces, ids_a) + partition_concavity_energy(
        vertices, faces, ids_b
    )


def _assign_band_by_plane(
    labels: np.ndarray,
    band_faces: Sequence[int],
    centroids: np.ndarray,
    normal: np.ndarray,
    offset: float,
    lab_a: int,
    lab_b: int,
) -> np.ndarray:
    out = labels.copy()
    n = _normalize(normal)
    for fi in band_faces:
        sd = float(np.dot(n, centroids[int(fi)]) + offset)
        out[int(fi)] = lab_a if sd >= 0.0 else lab_b
    return out


def _optimize_plane_params(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    label_sets: List[set],
    lab_a: int,
    lab_b: int,
    band_faces: Sequence[int],
    centroids: np.ndarray,
    boundary_pts: np.ndarray,
    cfg: PlanarBoundaryOptConfig,
) -> Tuple[np.ndarray, float, float]:
    """微调平面，使 E(A)+E(B) 最小。"""
    normal, offset = _fit_plane(boundary_pts)
    bbox = float(np.linalg.norm(boundary_pts.max(axis=0) - boundary_pts.min(axis=0)))
    if bbox < 1e-12:
        bbox = 1.0

    best_normal = normal.copy()
    best_offset = offset
    best_e = _pair_energy(vertices, faces, label_sets, lab_a, lab_b)

    span = max(float(cfg.plane_offset_span) * bbox, 1e-6)
    steps = max(int(cfg.plane_line_search_steps), 3)

    for _ in range(max(int(cfg.plane_normal_iters), 1)):
        for d in np.linspace(-span, span, steps):
            trial = _assign_band_by_plane(
                labels, band_faces, centroids,
                best_normal, best_offset + float(d), lab_a, lab_b,
            )
            sets = _label_face_sets(trial, len(label_sets))
            e = _pair_energy(vertices, faces, sets, lab_a, lab_b)
            if e + 1e-12 < best_e:
                best_e = e
                best_offset = best_offset + float(d)

        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, best_normal))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis1 = _normalize(np.cross(best_normal, ref))
        axis2 = _normalize(np.cross(best_normal, axis1))
        step = np.deg2rad(float(cfg.plane_normal_step_deg))
        for ax in (axis1, axis2):
            for sign in (-1.0, 1.0):
                n2 = _rotate_normal(best_normal, ax, sign * step)
                trial = _assign_band_by_plane(
                    labels, band_faces, centroids,
                    n2, best_offset, lab_a, lab_b,
                )
                sets = _label_face_sets(trial, len(label_sets))
                e = _pair_energy(vertices, faces, sets, lab_a, lab_b)
                if e + 1e-12 < best_e:
                    best_e = e
                    best_normal = n2

    return best_normal, best_offset, best_e


def _handle_loop_refine_band(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    lab_a: int,
    lab_b: int,
    band_faces: Sequence[int],
    neighbors: List[List[int]],
    label_sets: List[set],
    cfg: PlanarBoundaryOptConfig,
) -> Tuple[np.ndarray, int]:
    """在截面带沿 handle_loop / 凹缝路线微调。"""
    region = sorted(set(int(f) for f in band_faces))
    if len(region) < 4:
        return labels, 0

    global_to_local = {g: i for i, g in enumerate(region)}
    seeds_a_global = [int(f) for f in region if int(labels[int(f)]) == lab_a]
    seeds_b_global = [int(f) for f in region if int(labels[int(f)]) == lab_b]
    if not seeds_a_global or not seeds_b_global:
        return labels, 0

    from .partition.common import extract_submesh

    sub_v, sub_f, _ = extract_submesh(vertices, faces, region)
    seeds_a_local = [global_to_local[int(f)] for f in seeds_a_global[:8]]
    seeds_b_local = [global_to_local[int(f)] for f in seeds_b_global[:8]]

    loop_global, _ = find_handle_loop_cycle(
        vertices,
        faces,
        region,
        seeds_a_global[: min(8, len(seeds_a_global))],
        seeds_b_global[: min(8, len(seeds_b_global))],
    )
    seam_local, _ = find_concave_seam_between_seeds(
        sub_v,
        sub_f,
        seeds_a_local,
        seeds_b_local,
    )

    path_local: Set[int] = set(seam_local) if seam_local else set()
    if loop_global:
        path_local.update(
            global_to_local[int(g)]
            for g in loop_global
            if int(g) in global_to_local
        )
    if not path_local:
        return labels, 0

    out = labels.copy()
    changed = 0
    min_faces = max(int(cfg.min_block_faces), 1)
    core_a = set(seeds_a_local)
    core_b = set(seeds_b_local)

    for li in path_local:
        gfi = region[int(li)]
        if int(out[gfi]) not in (lab_a, lab_b):
            continue
        dist_a = min((abs(li - sa) for sa in core_a), default=999)
        dist_b = min((abs(li - sb) for sb in core_b), default=999)
        new_lab = lab_a if dist_a <= dist_b else lab_b
        if int(out[gfi]) == new_lab:
            continue
        old_lab = int(out[gfi])
        trial = out.copy()
        trial[gfi] = new_lab
        sets = _label_face_sets(trial, len(label_sets))
        if not _component_connected(old_lab, sets, neighbors, min_faces):
            continue
        if not _component_connected(new_lab, sets, neighbors, min_faces):
            continue
        out[gfi] = new_lab
        changed += 1

    return out, changed


def _adjacent_label_pairs(
    labels: np.ndarray,
    neighbors: List[List[int]],
) -> List[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    for fi, nbrs in enumerate(neighbors):
        la = int(labels[fi])
        if la < 0:
            continue
        for nb in nbrs:
            lb = int(labels[int(nb)])
            if lb < 0 or la == lb:
                continue
            pairs.add((min(la, lb), max(la, lb)))
    return sorted(pairs)


def _optimize_one_pair(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    lab_a: int,
    lab_b: int,
    neighbors: List[List[int]],
    centroids: np.ndarray,
    cfg: PlanarBoundaryOptConfig,
) -> Tuple[np.ndarray, PairOptRecord]:
    boundary = semantic_boundary_faces(faces, labels, lab_a, lab_b)
    if not boundary:
        return labels, PairOptRecord(
            lab_a, lab_b, [0, 0, 1], 0.0, 0.0, 0.0, 0, 0, 0, False, "no boundary"
        )

    loops_before = _boundary_loop_count(faces, labels, lab_a, lab_b)
    max_lab = max(int(labels.max()) + 1, lab_a + 1, lab_b + 1)
    label_sets = _label_face_sets(labels, max_lab)

    e_before = _pair_energy(vertices, faces, label_sets, lab_a, lab_b)
    band = sorted(
        expand_face_rings(
            faces,
            boundary,
            rings=max(int(cfg.boundary_band_rings), 1),
        )
    )
    band = [f for f in band if int(labels[f]) in (lab_a, lab_b)]
    if len(band) < 2:
        return labels, PairOptRecord(
            lab_a, lab_b, [0, 0, 1], 0.0, e_before, e_before,
            loops_before, loops_before, 0, False, "band too small",
        )

    bpts = _boundary_vertex_points(vertices, faces, boundary)
    normal, offset, _ = _optimize_plane_params(
        vertices, faces, labels, label_sets,
        lab_a, lab_b, band, centroids, bpts, cfg,
    )

    trial = _assign_band_by_plane(
        labels, band, centroids, normal, offset, lab_a, lab_b,
    )
    trial, _ = _handle_loop_refine_band(
        vertices, faces, trial, lab_a, lab_b, band,
        neighbors, label_sets, cfg,
    )

    loops_after = _boundary_loop_count(faces, trial, lab_a, lab_b)
    sets_after = _label_face_sets(trial, max_lab)
    min_faces = max(int(cfg.min_block_faces), 1)
    connected = (
        _component_connected(lab_a, sets_after, neighbors, min_faces)
        and _component_connected(lab_b, sets_after, neighbors, min_faces)
    )
    topo_ok = loops_after == loops_before
    e_after = _pair_energy(vertices, faces, sets_after, lab_a, lab_b)
    changed = int(np.sum(trial != labels))

    accept = connected and topo_ok and (e_after <= e_before + 1e-9)
    notes: List[str] = []
    if not connected:
        notes.append("connectivity")
    if not topo_ok:
        notes.append(f"topology {loops_before}->{loops_after}")
    if e_after > e_before + 1e-9:
        notes.append("energy increased")

    rec = PairOptRecord(
        part_a=lab_a,
        part_b=lab_b,
        plane_normal=[float(x) for x in normal.tolist()],
        plane_offset=float(offset),
        energy_before=float(e_before),
        energy_after=float(e_after),
        boundary_loops_before=int(loops_before),
        boundary_loops_after=int(loops_after),
        faces_changed=changed,
        accepted=accept,
        notes=", ".join(notes) if notes else "ok",
    )
    if accept and changed > 0:
        return trial, rec
    return labels, rec


def _optimize_one_pair_handle_loop_only(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    lab_a: int,
    lab_b: int,
    neighbors: List[List[int]],
    cfg: PlanarBoundaryOptConfig,
) -> Tuple[np.ndarray, PairOptRecord]:
    """仅沿 handle_loop / 凹缝微调交界，不要求凸包能量下降。"""
    boundary = semantic_boundary_faces(faces, labels, lab_a, lab_b)
    if not boundary:
        return labels, PairOptRecord(
            lab_a, lab_b, [0, 0, 1], 0.0, 0.0, 0.0, 0, 0, 0, False, "no boundary"
        )

    loops_before = _boundary_loop_count(faces, labels, lab_a, lab_b)
    max_lab = max(int(labels.max()) + 1, lab_a + 1, lab_b + 1)
    label_sets = _label_face_sets(labels, max_lab)

    e_before = _pair_energy(vertices, faces, label_sets, lab_a, lab_b)
    band = sorted(
        expand_face_rings(
            faces,
            boundary,
            rings=max(int(cfg.boundary_band_rings), 1),
        )
    )
    band = [f for f in band if int(labels[f]) in (lab_a, lab_b)]
    if len(band) < 2:
        return labels, PairOptRecord(
            lab_a, lab_b, [0, 0, 1], 0.0, e_before, e_before,
            loops_before, loops_before, 0, False, "band too small",
        )

    trial, _ = _handle_loop_refine_band(
        vertices, faces, labels, lab_a, lab_b, band,
        neighbors, label_sets, cfg,
    )

    loops_after = _boundary_loop_count(faces, trial, lab_a, lab_b)
    sets_after = _label_face_sets(trial, max_lab)
    min_faces = max(int(cfg.min_block_faces), 1)
    connected = (
        _component_connected(lab_a, sets_after, neighbors, min_faces)
        and _component_connected(lab_b, sets_after, neighbors, min_faces)
    )
    topo_ok = loops_after == loops_before
    e_after = _pair_energy(vertices, faces, sets_after, lab_a, lab_b)
    changed = int(np.sum(trial != labels))

    accept = connected and topo_ok and changed > 0
    notes: List[str] = []
    if not connected:
        notes.append("connectivity")
    if not topo_ok:
        notes.append(f"topology {loops_before}->{loops_after}")
    if changed <= 0:
        notes.append("no change")

    rec = PairOptRecord(
        part_a=lab_a,
        part_b=lab_b,
        plane_normal=[0.0, 0.0, 0.0],
        plane_offset=0.0,
        energy_before=float(e_before),
        energy_after=float(e_after),
        boundary_loops_before=int(loops_before),
        boundary_loops_after=int(loops_after),
        faces_changed=changed,
        accepted=accept,
        notes=", ".join(notes) if notes else "ok",
    )
    if accept:
        return trial, rec
    return labels, rec


def optimize_face_partition_planar_boundary(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    cfg: Optional[ConvexPartitionOptConfig] = None,
    planar_cfg: Optional[PlanarBoundaryOptConfig] = None,
) -> ConvexPartitionOptResult:
    """对每个相邻连通分块对做边界优化（平面 + handle_loop，或仅 handle_loop）。"""
    t0 = time.perf_counter()
    cfg = cfg or ConvexPartitionOptConfig()
    handle_loop_only = str(cfg.boundary_method or "planar").strip().lower() in (
        "handle_loop",
        "handle-loop",
        "handleloop",
    )
    pcfg = planar_cfg or PlanarBoundaryOptConfig(
        min_block_faces=max(int(cfg.min_block_faces), 1),
        verbose=cfg.verbose,
        handle_loop_only=handle_loop_only,
    )
    if planar_cfg is not None:
        pcfg.handle_loop_only = bool(planar_cfg.handle_loop_only or handle_loop_only)

    labels = np.asarray(face_labels, dtype=np.int64).copy()
    n_faces = len(labels)
    if n_faces == 0:
        empty = PartitionConcavityMetrics(
            np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64),
        )
        return ConvexPartitionOptResult(labels, empty, empty, elapsed_seconds=0.0)

    unique = sorted({int(x) for x in labels if int(x) >= 0})
    if len(unique) < 2:
        metrics = compute_partition_metrics(vertices, faces, labels, max(len(unique), 1))
        return ConvexPartitionOptResult(
            labels.copy(), metrics, metrics, elapsed_seconds=time.perf_counter() - t0,
        )

    label_map = {old: i for i, old in enumerate(unique)}
    num_labels = len(unique)
    for i in range(n_faces):
        if int(labels[i]) >= 0:
            labels[i] = label_map[int(labels[i])]

    metrics_before = compute_partition_metrics(vertices, faces, labels, num_labels)
    centroids, _ = face_centroids_and_normals(vertices, faces)
    adj = build_face_adjacency_csr(faces)
    neighbors: List[List[int]] = [
        adj.getrow(f).indices.tolist() for f in range(int(len(faces)))
    ]

    history: List[Dict[str, object]] = []
    total_changed = 0

    for pass_idx in range(max(int(pcfg.max_pair_passes), 1)):
        pairs = _adjacent_label_pairs(labels, neighbors)
        pass_changed = 0
        for lab_a, lab_b in pairs:
            if pcfg.handle_loop_only:
                new_labels, rec = _optimize_one_pair_handle_loop_only(
                    vertices, faces, labels, lab_a, lab_b,
                    neighbors, pcfg,
                )
            else:
                new_labels, rec = _optimize_one_pair(
                    vertices, faces, labels, lab_a, lab_b,
                    neighbors, centroids, pcfg,
                )
            history.append({
                "pass": pass_idx + 1,
                "part_a": lab_a,
                "part_b": lab_b,
                "accepted": rec.accepted,
                "faces_changed": rec.faces_changed,
                "energy_before": rec.energy_before,
                "energy_after": rec.energy_after,
                "boundary_loops_before": rec.boundary_loops_before,
                "boundary_loops_after": rec.boundary_loops_after,
                "plane_normal": rec.plane_normal,
                "plane_offset": rec.plane_offset,
                "notes": rec.notes,
            })
            if rec.accepted and rec.faces_changed > 0:
                labels = new_labels
                pass_changed += rec.faces_changed
                if pcfg.verbose:
                    tag = "handle-loop" if pcfg.handle_loop_only else "planar-boundary"
                    print(
                        f"[{tag}] pass {pass_idx+1} pair {lab_a}|{lab_b}: "
                        f"Δfaces={rec.faces_changed}  "
                        f"E {rec.energy_before:.5f}→{rec.energy_after:.5f}  "
                        f"loops={rec.boundary_loops_before}"
                    )
        total_changed += pass_changed
        if pass_changed == 0:
            break

    inv_map = {i: old for old, i in label_map.items()}
    labels_out = np.asarray(face_labels, dtype=np.int64).copy()
    for i in range(n_faces):
        lab = int(labels[i])
        if lab >= 0:
            labels_out[i] = inv_map[lab]

    metrics_after = compute_partition_metrics(vertices, faces, labels, num_labels)
    labels_orig = np.asarray(face_labels, dtype=np.int64)
    compare = labels_orig >= 0
    changed_ratio = (
        float(np.mean(labels_out[compare] != labels_orig[compare]))
        if compare.any()
        else float(np.mean(labels_out != labels_orig))
    )

    result = ConvexPartitionOptResult(
        face_labels=labels_out,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        num_swaps=total_changed,
        num_faces_changed=int(np.sum(labels_out != labels_orig)),
        elapsed_seconds=time.perf_counter() - t0,
        history=history,
        changed_face_ratio=changed_ratio,
    )
    if pcfg.verbose:
        tag = "handle-loop" if pcfg.handle_loop_only else "planar-boundary"
        print(
            f"[{tag}] 完成: pairs={len(history)}  "
            f"faces_changed={result.num_faces_changed}  "
            f"E_active {metrics_before.active_total_energy:.6f} → "
            f"{metrics_after.active_total_energy:.6f}  "
            f"(Δ={result.energy_delta:.6f})  "
            f"{result.elapsed_seconds:.1f}s"
        )
    return result


__all__ = [
    "PairOptRecord",
    "PlanarBoundaryOptConfig",
    "optimize_face_partition_planar_boundary",
]
