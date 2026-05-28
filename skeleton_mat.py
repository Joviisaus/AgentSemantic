"""
从蒙皮骨架近似生成 SEG-MAT / QMAT 所需的 initial.ma。

骨架关节 → MAT 球 (v x y z r)，骨骼父子 → 边，兄弟关节 → 三角面。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mesh_distortion_analyzer import SkeletonRig


def _skin_joint_parent(joint_nodes: Sequence[int], node_parents: Sequence[int], j: int) -> int:
    node_to_joint = {node_id: idx for idx, node_id in enumerate(joint_nodes)}
    parent_node = node_parents[joint_nodes[j]]
    while parent_node >= 0 and parent_node not in node_to_joint:
        parent_node = node_parents[parent_node]
    return node_to_joint[parent_node] if parent_node >= 0 else -1


def _estimate_joint_radii(
    joint_positions: np.ndarray,
    pose_vertices: np.ndarray,
    weights: np.ndarray,
    joints: np.ndarray,
    bbox_diag: float,
) -> np.ndarray:
    n = len(joint_positions)
    radii = np.full(n, max(bbox_diag * 0.02, 1e-4), dtype=np.float64)
    min_r = max(bbox_diag * 0.005, 1e-4)
    max_r = max(bbox_diag * 0.15, min_r * 2)

    for j in range(n):
        mask = np.zeros(len(pose_vertices), dtype=bool)
        w_j = np.zeros(len(pose_vertices), dtype=np.float64)
        for k in range(min(4, joints.shape[1])):
            sel = joints[:, k] == j
            mask |= sel
            w_j[sel] = np.maximum(w_j[sel], weights[sel, k])
        if not np.any(mask):
            continue
        pts = pose_vertices[mask]
        w = w_j[mask]
        w = w / max(float(w.sum()), 1e-12)
        center = joint_positions[j]
        dists = np.linalg.norm(pts - center, axis=1)
        r = float(np.sum(w * dists))
        if r <= 0:
            r = float(np.percentile(dists, 60))
        radii[j] = np.clip(r * 0.45, min_r, max_r)

    return radii


def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _ensure_edge(edge_set: set[Tuple[int, int]], a: int, b: int) -> None:
    edge_set.add(_edge_key(a, b))


def _has_triangle_edges(edge_set: set[Tuple[int, int]], a: int, b: int, c: int) -> bool:
    return (
        _edge_key(a, b) in edge_set
        and _edge_key(a, c) in edge_set
        and _edge_key(b, c) in edge_set
    )


def _add_face(
    edge_set: set[Tuple[int, int]],
    faces: List[Tuple[int, int, int]],
    a: int,
    b: int,
    c: int,
) -> None:
    """QMAT 要求 f 的三条边均存在，否则加载时会崩溃。"""
    _ensure_edge(edge_set, a, b)
    _ensure_edge(edge_set, a, c)
    _ensure_edge(edge_set, b, c)
    if _has_triangle_edges(edge_set, a, b, c):
        faces.append((a, b, c))


def _build_skeleton_graph(
    joint_nodes: Sequence[int],
    node_parents: Sequence[int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    n = len(joint_nodes)
    edge_set: set[Tuple[int, int]] = set()
    children: Dict[int, List[int]] = defaultdict(list)

    for j in range(n):
        p = _skin_joint_parent(joint_nodes, node_parents, j)
        if p >= 0:
            _ensure_edge(edge_set, p, j)
            children[p].append(j)

    faces: List[Tuple[int, int, int]] = []
    for p, chs in children.items():
        if len(chs) >= 2:
            for i in range(len(chs) - 1):
                _add_face(edge_set, faces, p, chs[i], chs[i + 1])
        elif len(chs) == 1:
            gp = _skin_joint_parent(joint_nodes, node_parents, p)
            if gp >= 0:
                _add_face(edge_set, faces, gp, p, chs[0])

    edges = sorted(edge_set)
    return edges, faces


def validate_ma_topology(path: Path) -> bool:
    """QMAT 要求每个 f 的三条边都存在，否则加载时会崩溃。"""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split()
            if len(header) < 3:
                return False
            nv, ne, nf = int(header[0]), int(header[1]), int(header[2])
            for _ in range(nv + ne):
                if not f.readline():
                    return False
            edge_set: set[Tuple[int, int]] = set()
            f.seek(0)
            f.readline()
            for _ in range(nv):
                f.readline()
            for _ in range(ne):
                parts = f.readline().strip().split()
                if len(parts) < 3:
                    return False
                a, b = int(parts[1]), int(parts[2])
                edge_set.add(_edge_key(a, b))
            for _ in range(nf):
                parts = f.readline().strip().split()
                if len(parts) < 4:
                    return False
                a, b, c = int(parts[1]), int(parts[2]), int(parts[3])
                if not _has_triangle_edges(edge_set, a, b, c):
                    return False
        return True
    except (OSError, ValueError):
        return False


def write_ma_file(
    path: Path,
    positions: np.ndarray,
    radii: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    faces: Sequence[Tuple[int, int, int]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(positions)} {len(edges)} {len(faces)}\n")
        for (x, y, z), r in zip(positions, radii):
            f.write(f"v {x:.15f} {y:.15f} {z:.15f} {r:.15f}\n")
        for a, b in edges:
            f.write(f"e {a} {b}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")
    return path


def export_skeleton_initial_ma(
    rig: "SkeletonRig",
    joint_positions: np.ndarray,
    pose_vertices: np.ndarray,
    output_path: Path,
) -> Path:
    """将 medoid 姿态下的骨架关节写入 initial.ma。"""
    positions = np.asarray(joint_positions, dtype=np.float64)
    if len(positions) != len(rig.joint_nodes):
        raise ValueError("关节数量与骨架不匹配")

    verts = np.asarray(pose_vertices, dtype=np.float64)
    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) if len(verts) else 1.0
    radii = _estimate_joint_radii(positions, verts, rig.weights, rig.joints, bbox_diag)
    edges, faces = _build_skeleton_graph(rig.joint_nodes, rig.node_parents)
    return write_ma_file(output_path, positions, radii, edges, faces)
