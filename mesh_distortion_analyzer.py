"""
Mesh 动画畸变分析器

支持 GLTF/GLB（含蒙皮动画）、FBX（via ufbx，纯 Python/C，无需 Blender/GUI）、以及 OBJ/PLY/STL 等静态格式。
对所有动作的所有帧做加权平均得到参考形状，再逐动作计算每个顶点的：
  - 角度畸变（1-ring 内各角相对参考形状的变化）
  - 1-邻域面积畸变（相邻三角形面积之和相对参考形状的变化）
取全局最大畸变作为热力图标量；运行本脚本会自动导出 JSON、渲染各动作 12 视角热力图（output/<模型>/heatmaps/）、
在 medoid 姿态上运行 SEG-MAT，并在 output/<模型>/medoid_pose/ 生成同视角 composite.png；
SEG-MAT 分块 12 视角图在 output/<模型>/segmentation/；谱聚类分块图在 output/<模型>/spectral_clustering/。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
import yaml
from pygltflib import GLTF2


# ---------------------------------------------------------------------------
# VLM 配置
# ---------------------------------------------------------------------------


def _resolve_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([^}]+)\}", value.strip())
    if match:
        return os.environ.get(match.group(1), "")
    return value


def load_vlm_config(path: str | Path) -> dict:
    """加载 VLM API 配置文件，并展开 ${ENV_VAR} 占位符。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for provider in cfg.get("providers", {}).values():
        for key, val in list(provider.items()):
            provider[key] = _resolve_env(val)
    return cfg


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class AnimationClip:
    name: str
    duration: float
    frame_times: np.ndarray  # (F,)
    frame_vertices: np.ndarray  # (F, V, 3)
    weight: float = 1.0


@dataclass
class SkeletonRig:
    """GLTF 蒙皮骨架，用于在指定动画时刻求关节位置并生成 initial.ma。"""

    joint_nodes: List[int]
    node_parents: List[int]
    node_roots: List[int]
    root_node_idx: int
    weights: np.ndarray  # (V, 4)
    joints: np.ndarray  # (V, 4)
    bind_locals: List[np.ndarray]
    node_translations: List[np.ndarray]
    node_rotations: List[np.ndarray]
    node_scales: List[np.ndarray]
    animation_channels: Dict[str, Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]]


@dataclass
class MeshData:
    vertices: np.ndarray  # (V, 3) rest / bind pose
    faces: np.ndarray  # (T, 3) int
    animations: List[AnimationClip] = field(default_factory=list)
    skeleton: Optional[SkeletonRig] = None


@dataclass
class DistortionResult:
    reference_vertices: np.ndarray
    weighted_average_vertices: np.ndarray
    angular_distortion: np.ndarray  # (V,) max over anims & frames
    area_distortion: np.ndarray  # (V,)
    combined_distortion: np.ndarray  # (V,) 全动画各帧取 max 后归一化（分析用）
    render_frame_angular: np.ndarray = field(default_factory=lambda: np.array([]))
    render_frame_area: np.ndarray = field(default_factory=lambda: np.array([]))
    render_frame_combined: np.ndarray = field(default_factory=lambda: np.array([]))
    per_animation_angular: Dict[str, np.ndarray] = field(default_factory=dict)
    per_animation_area: Dict[str, np.ndarray] = field(default_factory=dict)
    render_vertices: Optional[np.ndarray] = None
    render_anim_name: str = ""
    render_frame_index: int = -1
    render_time: float = 0.0
    render_medoid_score: float = 0.0


# ---------------------------------------------------------------------------
# 网格拓扑工具
# ---------------------------------------------------------------------------


@dataclass
class MeshTopology:
    """预计算的网格拓扑，畸变分析时复用。"""

    faces: np.ndarray
    vertex_faces: List[List[int]]
    corner_faces: List[np.ndarray]
    corner_local: List[np.ndarray]
    ref_ring_area: np.ndarray
    ref_corner_angles: np.ndarray  # (F, 3)
    sorted_ref_corner_angles: List[np.ndarray]


def build_mesh_topology(ref_vertices: np.ndarray, faces: np.ndarray) -> MeshTopology:
    n = len(ref_vertices)
    vertex_faces = build_vertex_faces(faces, n)
    cf_lists: List[List[int]] = [[] for _ in range(n)]
    cl_lists: List[List[int]] = [[] for _ in range(n)]
    for fi, tri in enumerate(faces):
        for li, vid in enumerate(tri):
            vid = int(vid)
            cf_lists[vid].append(fi)
            cl_lists[vid].append(li)
    corner_faces = [np.asarray(xs, dtype=np.int64) for xs in cf_lists]
    corner_local = [np.asarray(xs, dtype=np.int64) for xs in cl_lists]
    ref_corner_angles = _face_corner_angles(ref_vertices, faces)
    ref_ring = _vertex_one_ring_area_fast(ref_vertices, faces)
    sorted_ref = [np.sort(ref_corner_angles[fis, locs]) for fis, locs in zip(corner_faces, corner_local)]
    return MeshTopology(
        faces=faces,
        vertex_faces=vertex_faces,
        corner_faces=corner_faces,
        corner_local=corner_local,
        ref_ring_area=ref_ring,
        ref_corner_angles=ref_corner_angles,
        sorted_ref_corner_angles=sorted_ref,
    )


def build_one_ring(faces: np.ndarray, num_vertices: int) -> List[List[int]]:
    """每个顶点的一邻域顶点索引（不含自身）。"""
    nbrs: List[set[int]] = [set() for _ in range(num_vertices)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return [sorted(s) for s in nbrs]


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _face_corner_angles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """每个三角形三个内角，返回 (F, 3)。"""
    p = vertices[faces]
    p0, p1, p2 = p[:, 0], p[:, 1], p[:, 2]

    def _angles(curr: np.ndarray, prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        e1 = prev - curr
        e2 = nxt - curr
        n1 = np.linalg.norm(e1, axis=1)
        n2 = np.linalg.norm(e2, axis=1)
        denom = np.maximum(n1 * n2, 1e-12)
        cos_a = np.sum(e1 * e2, axis=1) / denom
        return np.arccos(np.clip(cos_a, -1.0, 1.0))

    return np.column_stack(
        [
            _angles(p0, p2, p1),
            _angles(p1, p0, p2),
            _angles(p2, p1, p0),
        ]
    )


def _vertex_one_ring_area_fast(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    areas = _face_areas(vertices, faces)
    ring = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(ring, faces[:, 0], areas)
    np.add.at(ring, faces[:, 1], areas)
    np.add.at(ring, faces[:, 2], areas)
    return ring


def _corner_angles_at_vertex(
    vertices: np.ndarray, faces: np.ndarray, vid: int
) -> List[float]:
    """顶点 vid 在所有相邻三角形中的内角（弧度）。"""
    angles: List[float] = []
    for tri in faces:
        if vid not in tri:
            continue
        idx = list(tri)
        vi = idx.index(vid)
        v_prev = vertices[idx[(vi - 1) % 3]]
        v_curr = vertices[idx[vi]]
        v_next = vertices[idx[(vi + 1) % 3]]
        e1 = v_prev - v_curr
        e2 = v_next - v_curr
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1.0, 1.0)
        angles.append(float(np.arccos(cos_a)))
    return angles


def vertex_one_ring_area(
    vertices: np.ndarray, faces: np.ndarray, vertex_faces: List[List[int]]
) -> np.ndarray:
    """每个顶点 1-邻域内所有三角形面积之和。"""
    areas = _face_areas(vertices, faces)
    ring = np.zeros(len(vertices), dtype=np.float64)
    for vi, fids in enumerate(vertex_faces):
        if fids:
            ring[vi] = areas[fids].sum()
    return ring


def build_vertex_faces(faces: np.ndarray, num_vertices: int) -> List[List[int]]:
    vf: List[List[int]] = [[] for _ in range(num_vertices)]
    for fi, tri in enumerate(faces):
        for vid in tri:
            vf[int(vid)].append(fi)
    return vf


def compute_angular_distortion(
    ref_vertices: np.ndarray,
    def_vertices: np.ndarray,
    faces: np.ndarray,
    topo: Optional[MeshTopology] = None,
    vertex_ids: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """相对参考网格，逐顶点角度畸变（相邻角差值的均值）。"""
    if topo is None:
        topo = build_mesh_topology(ref_vertices, faces)
    def_angles = _face_corner_angles(def_vertices, faces)
    n = len(ref_vertices)
    out = np.zeros(n, dtype=np.float64)
    for vid in range(n):
        fis = topo.corner_faces[vid]
        if fis.size == 0:
            continue
        locs = topo.corner_local[vid]
        r = topo.sorted_ref_corner_angles[vid]
        d = np.sort(def_angles[fis, locs])
        m = min(len(r), len(d))
        out[vid] = float(np.mean(np.abs(d[:m] - r[:m])))
    return out


def compute_area_distortion(
    ref_vertices: np.ndarray,
    def_vertices: np.ndarray,
    faces: np.ndarray,
    topo: Optional[MeshTopology] = None,
    vertex_faces: Optional[List[List[int]]] = None,
) -> np.ndarray:
    """1-邻域面积畸变: max(A/A0, A0/A) - 1。"""
    a0 = topo.ref_ring_area if topo is not None else _vertex_one_ring_area_fast(ref_vertices, faces)
    a1 = _vertex_one_ring_area_fast(def_vertices, faces)
    eps = 1e-12
    ratio = np.maximum(a1 / np.maximum(a0, eps), a0 / np.maximum(a1, eps))
    return ratio - 1.0


def _distortion_for_frame(
    ref_v: np.ndarray,
    frame: np.ndarray,
    topo: MeshTopology,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        compute_angular_distortion(ref_v, frame, topo.faces, topo=topo),
        compute_area_distortion(ref_v, frame, topo.faces, topo=topo),
    )


_worker_ref_v: Optional[np.ndarray] = None
_worker_topo: Optional[MeshTopology] = None


def _init_distortion_worker(ref_v: np.ndarray, topo: MeshTopology) -> None:
    global _worker_ref_v, _worker_topo
    _worker_ref_v = ref_v
    _worker_topo = topo


def _distortion_worker(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert _worker_ref_v is not None and _worker_topo is not None
    return _distortion_for_frame(_worker_ref_v, frame, _worker_topo)


def _bbox_diagonal(vertices: np.ndarray) -> float:
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def _subsample_indices(count: int, max_samples: Optional[int]) -> np.ndarray:
    if max_samples is None or count <= max_samples:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_samples, dtype=np.int64)


def _collect_animation_frame_indices(
    frames: np.ndarray,
    motion_thresh: float,
) -> np.ndarray:
    """
    利用动作连续性：相对上一关键帧位移低于阈值则跳过畸变计算。
    始终保留首帧与末帧。
    """
    n = len(frames)
    if n == 0:
        return np.array([], dtype=np.int64)
    if motion_thresh <= 0:
        return np.arange(n, dtype=np.int64)

    keep = [0]
    prev = frames[0]
    for i in range(1, n - 1):
        motion = float(np.linalg.norm(frames[i] - prev, axis=1).max())
        if motion >= motion_thresh:
            keep.append(i)
            prev = frames[i]
    keep.append(n - 1)
    return np.unique(np.asarray(keep, dtype=np.int64))


def _medoid_scores(stacks: np.ndarray) -> np.ndarray:
    """stacks: (N, V, 3)，返回各帧 medoid 分数（越小越接近其它帧）。"""
    n = stacks.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    diff = stacks[:, None, :, :] - stacks[None, :, :, :]
    dist = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    np.fill_diagonal(dist, 0.0)
    return dist.sum(axis=1) / (n - 1)


def select_medoid_for_clip(
    anim: AnimationClip,
    medoid_max_samples: Optional[int] = 128,
) -> Tuple[np.ndarray, int, float, float]:
    """在单个动作内部选取 medoid 帧（可向量化 + 子采样加速）。"""
    stacks = anim.frame_vertices
    n = len(stacks)
    if n == 0:
        raise ValueError(f"动作 {anim.name} 没有帧")
    if n == 1:
        return stacks[0].copy(), 0, float(anim.frame_times[0]), 0.0

    idx = _subsample_indices(n, medoid_max_samples)
    sub = stacks[idx]
    scores = _medoid_scores(sub)
    best_sub = int(np.argmin(scores))
    best = int(idx[best_sub])
    return (
        stacks[best].copy(),
        best,
        float(anim.frame_times[best]),
        float(scores[best_sub]),
    )


def _subsample_frame_times(
    frame_times: np.ndarray, max_frames: Optional[int]
) -> np.ndarray:
    if max_frames is None or len(frame_times) <= max_frames:
        return frame_times
    idx = np.linspace(0, len(frame_times) - 1, max_frames, dtype=np.int64)
    return frame_times[idx]


# ---------------------------------------------------------------------------
# GLTF / GLB 加载（含蒙皮动画）
# ---------------------------------------------------------------------------


def _gltf_buffer_data(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    if accessor.bufferView is None:
        raise ValueError(
            f"accessor {accessor_index} 无 bufferView（网格可能为 Draco 压缩，需通过 primitive 解码）"
        )
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]
    if buffer.uri is None:
        blob = gltf.binary_blob()
    else:
        blob = gltf.get_data_from_buffer_uri(buffer.uri)
    if blob is None:
        raise ValueError(f"无法读取 accessor {accessor_index} 的二进制数据")

    start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    dtype_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    comp_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
    dtype = dtype_map[accessor.componentType]
    type_name = accessor.type if isinstance(accessor.type, str) else str(accessor.type)
    comps = comp_map.get(type_name, 1)
    count = accessor.count
    itemsize = np.dtype(dtype).itemsize
    stride = buffer_view.byteStride or (itemsize * comps)
    if stride == itemsize * comps:
        arr = np.frombuffer(blob, dtype=dtype, count=count * comps, offset=start)
        if comps == 1:
            return arr
        if type_name == "MAT4":
            return arr.reshape(count, 4, 4)
        return arr.reshape(count, comps)
    rows = []
    for i in range(count):
        off = start + i * stride
        chunk = blob[off : off + itemsize * comps]
        rows.append(np.frombuffer(chunk, dtype=dtype, count=comps))
    stacked = np.vstack(rows)
    if type_name == "MAT4":
        return stacked.reshape(count, 4, 4)
    return stacked


def _gltf_read_mat4(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    """读取 glTF MAT4 accessor（列主序）并转为 numpy 行主序 4x4。"""
    raw = _gltf_buffer_data(gltf, accessor_index).reshape(-1, 4, 4)
    return np.transpose(raw, (0, 2, 1))


def _gltf_is_draco_primitive(prim) -> bool:
    return bool(prim.extensions and prim.extensions.get("KHR_draco_mesh_compression"))


def _gltf_get_buffer_bytes(gltf: GLTF2, buffer_view_index: int) -> bytes:
    buffer_view = gltf.bufferViews[buffer_view_index]
    buffer = gltf.buffers[buffer_view.buffer]
    if buffer.uri is None:
        blob = gltf.binary_blob()
    else:
        blob = gltf.get_data_from_buffer_uri(buffer.uri)
    if blob is None:
        raise ValueError(f"无法读取 bufferView {buffer_view_index}")
    start = buffer_view.byteOffset or 0
    end = start + buffer_view.byteLength
    return blob[start:end]


def _gltf_decode_draco_primitive(gltf: GLTF2, prim) -> Dict[str, np.ndarray]:
    try:
        import DracoPy
    except ImportError as exc:
        raise ImportError(
            "该 glTF 使用 KHR_draco_mesh_compression，请安装: pip install DracoPy"
        ) from exc

    ext = prim.extensions["KHR_draco_mesh_compression"]
    chunk = _gltf_get_buffer_bytes(gltf, ext["bufferView"])
    dmesh = DracoPy.decode(chunk)
    by_uid = {int(a["unique_id"]): np.asarray(a["data"]) for a in dmesh.attributes}
    attr_map = ext["attributes"]

    pos = by_uid[int(attr_map["POSITION"])].astype(np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3)
    faces = np.asarray(dmesh.faces, dtype=np.int64)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)

    out: Dict[str, np.ndarray] = {"POSITION": pos, "faces": faces}
    if "JOINTS_0" in attr_map:
        joints = by_uid[int(attr_map["JOINTS_0"])].astype(np.int64).reshape(-1, 4)
        out["JOINTS_0"] = joints
    if "WEIGHTS_0" in attr_map:
        weights = by_uid[int(attr_map["WEIGHTS_0"])].astype(np.float64).reshape(-1, 4)
        wsum = weights.sum(axis=1, keepdims=True)
        wsum[wsum < 1e-8] = 1.0
        out["WEIGHTS_0"] = weights / wsum
    return out


def _gltf_decode_uncompressed_primitive(gltf: GLTF2, prim) -> Dict[str, np.ndarray]:
    pos = _gltf_buffer_data(gltf, prim.attributes.POSITION).astype(np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3)
    if prim.indices is not None:
        idx = _gltf_buffer_data(gltf, prim.indices).astype(np.int64).reshape(-1)
        faces = idx.reshape(-1, 3)
    else:
        faces = np.arange(len(pos), dtype=np.int64).reshape(-1, 3)

    out: Dict[str, np.ndarray] = {"POSITION": pos, "faces": faces}
    if prim.attributes.JOINTS_0 is not None and prim.attributes.WEIGHTS_0 is not None:
        weights = _gltf_buffer_data(gltf, prim.attributes.WEIGHTS_0).astype(np.float64)
        joints = _gltf_buffer_data(gltf, prim.attributes.JOINTS_0).astype(np.int64)
        if weights.ndim == 1:
            weights = weights.reshape(-1, 4)
        if joints.ndim == 1:
            joints = joints.reshape(-1, 4)
        wsum = weights.sum(axis=1, keepdims=True)
        wsum[wsum < 1e-8] = 1.0
        out["WEIGHTS_0"] = weights / wsum
        out["JOINTS_0"] = joints
    return out


def _gltf_decode_primitive(gltf: GLTF2, prim) -> Dict[str, np.ndarray]:
    if _gltf_is_draco_primitive(prim):
        return _gltf_decode_draco_primitive(gltf, prim)
    return _gltf_decode_uncompressed_primitive(gltf, prim)


def _gltf_load_binary(path: Path) -> GLTF2:
    gltf = GLTF2.load(str(path))
    gltf._path = str(path.parent)  # type: ignore[attr-defined]
    return gltf


def _mat4_from_trs(
    t: Optional[np.ndarray],
    r: Optional[np.ndarray],
    s: Optional[np.ndarray],
) -> np.ndarray:
    T = np.eye(4)
    if t is not None:
        T[:3, 3] = t
    R = np.eye(4)
    if r is not None:
        x, y, z, w = r
        R[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
    S = np.eye(4)
    if s is not None:
        S[0, 0], S[1, 1], S[2, 2] = s
    return T @ R @ S


def _interp_keyframes(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    if len(times) == 0:
        return values[0]
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    idx = np.searchsorted(times, t) - 1
    idx = max(0, min(idx, len(times) - 2))
    t0, t1 = times[idx], times[idx + 1]
    w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    v0, v1 = values[idx], values[idx + 1]
    if v0.shape[-1] == 4:  # quaternion slerp 简化为 lerp（小角度足够）
        q = (1 - w) * v0 + w * v1
        q /= max(np.linalg.norm(q), 1e-12)
        return q
    return (1 - w) * v0 + w * v1


@dataclass
class _GLTFScene:
    gltf: GLTF2
    mesh_index: int
    nodes: List[int]
    joint_nodes: List[int]
    inverse_bind: np.ndarray  # (J, 4, 4)
    weights: np.ndarray  # (V, 4)
    joints: np.ndarray  # (V, 4) joint indices
    rest_vertices: np.ndarray
    faces: np.ndarray
    node_parents: List[int]
    node_roots: List[int]
    part_ranges: List[Tuple[int, int, int]] = field(default_factory=list)  # start, end, mesh_node_idx
    root_node_idx: int = -1
    local_rest_vertices: np.ndarray = field(default_factory=lambda: np.empty(0))
    rest_v_h: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def _collect_gltf_skin_mesh(gltf: GLTF2) -> _GLTFScene:
    """合并所有带 skin 的 mesh 部件（支持 Draco 压缩）。"""
    skin_idx: Optional[int] = None
    skinned_nodes: List[Tuple[int, int]] = []
    for ni, node in enumerate(gltf.nodes):
        if node.mesh is None:
            continue
        mesh = gltf.meshes[node.mesh]
        has_skin_attr = any(
            p.attributes.JOINTS_0 is not None
            or (p.extensions and "KHR_draco_mesh_compression" in (p.extensions or {}))
            for p in mesh.primitives
        )
        if node.skin is not None and has_skin_attr:
            if skin_idx is None:
                skin_idx = node.skin
            skinned_nodes.append((ni, node.mesh))

    if not skinned_nodes:
        for ni, node in enumerate(gltf.nodes):
            if node.mesh is not None:
                skinned_nodes = [(ni, node.mesh)]
                break
    if not skinned_nodes:
        raise ValueError("GLTF 中未找到 mesh")

    vert_chunks: List[np.ndarray] = []
    face_chunks: List[np.ndarray] = []
    weight_chunks: List[np.ndarray] = []
    joint_chunks: List[np.ndarray] = []
    part_ranges: List[Tuple[int, int, int]] = []
    mesh_node_indices: List[int] = []
    voff = 0

    for node_idx, mesh_idx in skinned_nodes:
        mesh = gltf.meshes[mesh_idx]
        for prim in mesh.primitives:
            decoded = _gltf_decode_primitive(gltf, prim)
            pos = decoded["POSITION"]
            faces = decoded["faces"] + voff
            vert_chunks.append(pos)
            face_chunks.append(faces)
            if "WEIGHTS_0" in decoded:
                weight_chunks.append(decoded["WEIGHTS_0"])
                joint_chunks.append(decoded["JOINTS_0"])
            part_ranges.append((voff, voff + len(pos), node_idx))
            mesh_node_indices.append(node_idx)
            voff += len(pos)

    rest_v = np.vstack(vert_chunks) if vert_chunks else np.empty((0, 3), dtype=np.float64)
    local_rest = rest_v.copy()
    faces = np.vstack(face_chunks) if face_chunks else np.empty((0, 3), dtype=np.int64)

    joint_nodes: List[int] = []
    inv_bind = np.stack([np.eye(4)] * 1)
    if skin_idx is not None and gltf.skins[skin_idx].joints:
        skin = gltf.skins[skin_idx]
        joint_nodes = list(skin.joints)
        inv_bind = _gltf_read_mat4(gltf, skin.inverseBindMatrices)

    if weight_chunks:
        weights = np.vstack(weight_chunks)
        joints = np.vstack(joint_chunks)
    else:
        weights = np.zeros((len(rest_v), 4), dtype=np.float64)
        joints = np.zeros((len(rest_v), 4), dtype=np.int64)

    node_parents = [-1] * len(gltf.nodes)
    node_roots: List[int] = []
    for i, node in enumerate(gltf.nodes):
        for ch in node.children or []:
            node_parents[ch] = i
    for i, p in enumerate(node_parents):
        if p < 0:
            node_roots.append(i)

    bind_locals = _gltf_node_bind_locals(gltf)
    bind_globals = _gltf_globals_from_locals(gltf, node_parents, node_roots, bind_locals)
    root_node_idx = (
        gltf.skins[skin_idx].skeleton
        if skin_idx is not None and gltf.skins[skin_idx].skeleton is not None
        else (joint_nodes[0] if joint_nodes else skinned_nodes[0][0])
    )
    root_inv = np.linalg.inv(bind_globals[root_node_idx])

    # 各部件顶点处于不同 mesh 节点局部系，统一变换到 skeleton 根节点空间
    aligned_chunks: List[np.ndarray] = []
    for (pos, (_, _, mesh_node_idx)) in zip(vert_chunks, part_ranges):
        node_g = bind_globals[mesh_node_idx]
        v_h = np.hstack([pos, np.ones((len(pos), 1))])
        aligned_chunks.append(((root_inv @ node_g @ v_h.T).T)[:, :3])
    rest_v = np.vstack(aligned_chunks) if aligned_chunks else np.empty((0, 3), dtype=np.float64)

    return _GLTFScene(
        gltf=gltf,
        mesh_index=skinned_nodes[0][1],
        nodes=mesh_node_indices,
        joint_nodes=joint_nodes,
        inverse_bind=inv_bind,
        weights=weights,
        joints=joints,
        rest_vertices=rest_v.copy(),
        faces=faces,
        node_parents=node_parents,
        node_roots=node_roots,
        part_ranges=part_ranges,
        root_node_idx=root_node_idx,
        local_rest_vertices=local_rest,
        rest_v_h=np.hstack([local_rest, np.ones((len(local_rest), 1))]),
    )


def _gltf_globals_from_locals(
    gltf: GLTF2, parents: List[int], roots: List[int], locals_: List[np.ndarray]
) -> List[np.ndarray]:
    globals_ = [np.eye(4) for _ in gltf.nodes]

    def dfs(i: int) -> None:
        p = parents[i]
        globals_[i] = globals_[p] @ locals_[i] if p >= 0 else locals_[i]
        for ch in gltf.nodes[i].children or []:
            dfs(ch)

    for i in roots:
        dfs(i)
    return globals_


def _gltf_node_bind_locals(gltf: GLTF2) -> List[np.ndarray]:
    locals_: List[np.ndarray] = []
    for node in gltf.nodes:
        t = np.array(node.translation or [0, 0, 0], dtype=np.float64)
        r = np.array(node.rotation or [0, 0, 0, 1], dtype=np.float64)
        s = np.array(node.scale or [1, 1, 1], dtype=np.float64)
        locals_.append(_mat4_from_trs(t, r, s))
    return locals_


def _gltf_skin_part(
    v_h: np.ndarray,
    joints: np.ndarray,
    weights: np.ndarray,
    joint_globals: List[np.ndarray],
    mesh_node_idx: int,
    inverse_bind: np.ndarray,
    joint_nodes: List[int],
) -> np.ndarray:
    mesh_inv = np.linalg.inv(joint_globals[mesh_node_idx])
    skin_mats = np.stack(
        [
            mesh_inv @ joint_globals[node_id] @ inverse_bind[j]
            for j, node_id in enumerate(joint_nodes)
        ]
    )
    out = np.zeros((len(v_h), 3), dtype=np.float64)
    for j in range(len(skin_mats)):
        transformed = (skin_mats[j] @ v_h.T).T[:, :3]
        for k in range(4):
            sel = (joints[:, k] == j) & (weights[:, k] > 1e-8)
            if np.any(sel):
                out[sel] += weights[sel, k, None] * transformed[sel]
    return out


def _gltf_apply_skin(scene: _GLTFScene, joint_globals: List[np.ndarray]) -> np.ndarray:
    if not scene.joint_nodes:
        return scene.rest_vertices.copy()
    root_inv = (
        np.linalg.inv(joint_globals[scene.root_node_idx])
        if scene.root_node_idx >= 0
        else np.eye(4)
    )
    local_v_h = (
        np.hstack([scene.local_rest_vertices, np.ones((len(scene.local_rest_vertices), 1))])
        if len(scene.local_rest_vertices)
        else scene.rest_v_h
    )
    if scene.part_ranges:
        out = np.zeros((len(local_v_h), 3), dtype=np.float64)
        for start, end, mesh_node_idx in scene.part_ranges:
            skinned_local = _gltf_skin_part(
                local_v_h[start:end],
                scene.joints[start:end],
                scene.weights[start:end],
                joint_globals,
                mesh_node_idx,
                scene.inverse_bind,
                scene.joint_nodes,
            )
            v_h = np.hstack([skinned_local, np.ones((end - start, 1))])
            out[start:end] = ((root_inv @ joint_globals[mesh_node_idx] @ v_h.T).T)[:, :3]
        return out
    if scene.nodes:
        skinned_local = _gltf_skin_part(
            local_v_h,
            scene.joints,
            scene.weights,
            joint_globals,
            scene.nodes[0],
            scene.inverse_bind,
            scene.joint_nodes,
        )
        v_h = np.hstack([skinned_local, np.ones((len(skinned_local), 1))])
        return ((root_inv @ joint_globals[scene.nodes[0]] @ v_h.T).T)[:, :3]
    skin_mats = np.stack(
        [
            joint_globals[node_id] @ scene.inverse_bind[j]
            for j, node_id in enumerate(scene.joint_nodes)
        ]
    )
    out = np.zeros((len(scene.rest_v_h), 3), dtype=np.float64)
    for j in range(len(skin_mats)):
        transformed = (skin_mats[j] @ scene.rest_v_h.T).T[:, :3]
        for k in range(4):
            sel = (scene.joints[:, k] == j) & (scene.weights[:, k] > 1e-8)
            if np.any(sel):
                out[sel] += scene.weights[sel, k, None] * transformed[sel]
    return out


def _gltf_evaluate_nodes_at_time(
    gltf: GLTF2,
    channels: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    t: float,
    parents: List[int],
    roots: List[int],
) -> List[np.ndarray]:
    locals_ = _gltf_node_bind_locals(gltf)
    for i, node in enumerate(gltf.nodes):
        if i not in channels:
            continue
        ch = channels[i]
        t_def = np.array(node.translation or [0, 0, 0], dtype=np.float64)
        r_def = np.array(node.rotation or [0, 0, 0, 1], dtype=np.float64)
        s_def = np.array(node.scale or [1, 1, 1], dtype=np.float64)
        if "translation" in ch:
            t_def = _interp_keyframes(ch["translation"][0], ch["translation"][1], t)
        if "rotation" in ch:
            r_def = _interp_keyframes(ch["rotation"][0], ch["rotation"][1], t)
        if "scale" in ch:
            s_def = _interp_keyframes(ch["scale"][0], ch["scale"][1], t)
        locals_[i] = _mat4_from_trs(t_def, r_def, s_def)
    return _gltf_globals_from_locals(gltf, parents, roots, locals_)


def _gltf_node_trs_defaults(gltf: GLTF2) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    translations: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    scales: List[np.ndarray] = []
    for node in gltf.nodes:
        translations.append(np.array(node.translation or [0, 0, 0], dtype=np.float64))
        rotations.append(np.array(node.rotation or [0, 0, 0, 1], dtype=np.float64))
        scales.append(np.array(node.scale or [1, 1, 1], dtype=np.float64))
    return translations, rotations, scales


def _rig_globals_from_locals(rig: SkeletonRig, locals_: List[np.ndarray]) -> List[np.ndarray]:
    n = len(locals_)
    globals_ = [np.eye(4) for _ in range(n)]

    def dfs(i: int) -> None:
        p = rig.node_parents[i]
        globals_[i] = globals_[p] @ locals_[i] if p >= 0 else locals_[i]
        for ci, parent in enumerate(rig.node_parents):
            if parent == i:
                dfs(ci)

    for i in rig.node_roots:
        dfs(i)
    return globals_


def evaluate_skeleton_joint_positions(
    rig: SkeletonRig,
    anim_name: str = "",
    time: float = 0.0,
) -> np.ndarray:
    """在 skeleton 根空间求各 skin joint 位置（与蒙皮 mesh 顶点同坐标系）。"""
    if not rig.joint_nodes:
        raise ValueError("骨架无关节")

    ch_map = rig.animation_channels.get(anim_name, {})
    locals_: List[np.ndarray] = []
    for i in range(len(rig.bind_locals)):
        if i not in ch_map:
            locals_.append(rig.bind_locals[i].copy())
            continue
        ch = ch_map[i]
        t_def = rig.node_translations[i].copy()
        r_def = rig.node_rotations[i].copy()
        s_def = rig.node_scales[i].copy()
        if "translation" in ch:
            t_def = _interp_keyframes(ch["translation"][0], ch["translation"][1], time)
        if "rotation" in ch:
            r_def = _interp_keyframes(ch["rotation"][0], ch["rotation"][1], time)
        if "scale" in ch:
            s_def = _interp_keyframes(ch["scale"][0], ch["scale"][1], time)
        locals_.append(_mat4_from_trs(t_def, r_def, s_def))

    joint_globals = _rig_globals_from_locals(rig, locals_)
    root_inv = np.linalg.inv(joint_globals[rig.root_node_idx])
    positions = np.empty((len(rig.joint_nodes), 3), dtype=np.float64)
    for j, node_id in enumerate(rig.joint_nodes):
        positions[j] = (root_inv @ joint_globals[node_id])[:3, 3]
    return positions


def _build_skeleton_rig_from_gltf_scene(
    gltf: GLTF2,
    scene: _GLTFScene,
    animation_channels: Dict[str, Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
) -> Optional[SkeletonRig]:
    if not scene.joint_nodes:
        return None
    bind_locals = _gltf_node_bind_locals(gltf)
    trans, rots, scales = _gltf_node_trs_defaults(gltf)
    return SkeletonRig(
        joint_nodes=list(scene.joint_nodes),
        node_parents=list(scene.node_parents),
        node_roots=list(scene.node_roots),
        root_node_idx=scene.root_node_idx,
        weights=scene.weights.copy(),
        joints=scene.joints.copy(),
        bind_locals=bind_locals,
        node_translations=trans,
        node_rotations=rots,
        node_scales=scales,
        animation_channels=animation_channels,
    )


def load_gltf_animated(
    path: Path,
    samples_per_second: float = 30.0,
    max_frames: Optional[int] = None,
) -> MeshData:
    gltf = _gltf_load_binary(path)
    scene = _collect_gltf_skin_mesh(gltf)
    animations: List[AnimationClip] = []
    anim_channels: Dict[str, Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = {}

    if not gltf.animations:
        skeleton = _build_skeleton_rig_from_gltf_scene(gltf, scene, anim_channels)
        return MeshData(scene.rest_vertices, scene.faces, [], skeleton=skeleton)

    for anim in gltf.animations:
        ch_map: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        t_max = 0.0
        for ch in anim.channels:
            if ch.target.node is None:
                continue
            node = ch.target.node
            path_attr = ch.target.path
            sampler = anim.samplers[ch.sampler]
            times = _gltf_buffer_data(gltf, sampler.input).flatten().astype(np.float64)
            vals = _gltf_buffer_data(gltf, sampler.output).astype(np.float64)
            ch_map.setdefault(node, {})[path_attr] = (times, vals)
            if len(times):
                t_max = max(t_max, float(times[-1]))
        if t_max <= 0:
            continue
        n_frames = max(2, int(t_max * samples_per_second) + 1)
        frame_times = _subsample_frame_times(np.linspace(0.0, t_max, n_frames), max_frames)
        frames = [
            _gltf_apply_skin(
                scene,
                _gltf_evaluate_nodes_at_time(
                    gltf, ch_map, float(t), scene.node_parents, scene.node_roots
                ),
            )
            for t in frame_times
        ]
        name = anim.name or f"anim_{len(animations)}"
        anim_channels[name] = ch_map
        animations.append(
            AnimationClip(
                name=name,
                duration=t_max,
                frame_times=frame_times,
                frame_vertices=np.stack(frames, axis=0),
                weight=t_max,
            )
        )

    skeleton = _build_skeleton_rig_from_gltf_scene(gltf, scene, anim_channels)
    return MeshData(scene.rest_vertices, scene.faces, animations, skeleton=skeleton)


# ---------------------------------------------------------------------------
# FBX 加载（ufbx — 无 GUI / 无 Blender）
# ---------------------------------------------------------------------------


def _ufbx_vec3_list_to_numpy(vec3_list, count: int) -> np.ndarray:
    out = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        v = vec3_list[i]
        out[i, 0] = v.x
        out[i, 1] = v.y
        out[i, 2] = v.z
    return out


def _ufbx_triangulate_faces(mesh) -> np.ndarray:
    tris: List[List[int]] = []
    for fi in range(mesh.num_faces):
        face = mesh.faces[fi]
        ib, ni = face.index_begin, face.num_indices
        idxs = [int(mesh.vertex_indices[ib + j]) for j in range(ni)]
        for j in range(1, ni - 1):
            tris.append([idxs[0], idxs[j], idxs[j + 1]])
    if not tris:
        raise ValueError("FBX mesh 没有可用的面数据")
    return np.asarray(tris, dtype=np.int64)


def _ufbx_pick_mesh(scene) -> Any:
    if not scene.meshes:
        raise ValueError("FBX 中未找到 mesh")
    return max(scene.meshes, key=lambda m: m.num_vertices)


def _ufbx_match_mesh(eval_scene, ref_mesh) -> Any:
    ref_name = ref_mesh.name
    ref_id = ref_mesh.element_id
    for mesh in eval_scene.meshes:
        if ref_name and mesh.name == ref_name:
            return mesh
        if mesh.element_id == ref_id:
            return mesh
    return _ufbx_pick_mesh(eval_scene)


def _ufbx_skinned_vertices(eval_mesh) -> np.ndarray:
    sp = eval_mesh.skinned_position
    if sp.exists and len(sp.values) > 0:
        return _ufbx_vec3_list_to_numpy(sp.values, eval_mesh.num_vertices)
    return _ufbx_vec3_list_to_numpy(eval_mesh.vertices, eval_mesh.num_vertices)


def load_fbx_animated(
    path: Path,
    samples_per_second: float = 30.0,
    max_frames: Optional[int] = None,
) -> MeshData:
    """使用 ufbx 加载 FBX（含蒙皮动画），无需 Blender/GUI。"""
    try:
        import ufbx
    except ImportError as exc:
        raise ImportError("FBX 加载需要 ufbx，请运行: pip install ufbx") from exc

    scene = ufbx.load_file(str(path))
    mesh = _ufbx_pick_mesh(scene)
    rest_vertices = _ufbx_vec3_list_to_numpy(mesh.vertices, mesh.num_vertices)
    faces = _ufbx_triangulate_faces(mesh)

    stacks = list(scene.anim_stacks)
    if not stacks and scene.anim is not None:
        stacks = [scene]

    animations: List[AnimationClip] = []
    for si, stack in enumerate(stacks):
        anim = getattr(stack, "anim", None) or scene.anim
        if anim is None:
            continue
        t0 = float(getattr(stack, "time_begin", 0.0))
        t1 = float(getattr(stack, "time_end", t0))
        if t1 <= t0:
            t1 = t0 + 1.0 / max(samples_per_second, 1.0)
        duration = t1 - t0
        n_frames = max(2, int(duration * samples_per_second) + 1)
        frame_times = _subsample_frame_times(np.linspace(t0, t1, n_frames), max_frames)
        frames = []
        for t in frame_times:
            ev = scene.evaluate(anim, float(t), evaluate_skinning=True)
            em = _ufbx_match_mesh(ev, mesh)
            frames.append(_ufbx_skinned_vertices(em))
        name = getattr(stack, "name", None) or f"anim_{si}"
        animations.append(
            AnimationClip(
                name=name,
                duration=duration,
                frame_times=frame_times,
                frame_vertices=np.stack(frames, axis=0),
                weight=duration,
            )
        )

    return MeshData(rest_vertices, faces, animations)


# ---------------------------------------------------------------------------
# 通用加载
# ---------------------------------------------------------------------------


def load_mesh(
    path: str | Path,
    samples_per_second: float = 30.0,
    max_frames: Optional[int] = None,
) -> MeshData:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".gltf", ".glb"):
        return load_gltf_animated(path, samples_per_second, max_frames)

    if ext == ".fbx":
        return load_fbx_animated(path, samples_per_second, max_frames)

    # 静态格式
    tm = trimesh.load(str(path), process=True, force="mesh")
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError(f"无法解析 mesh: {path}")
    return MeshData(
        np.array(tm.vertices, dtype=np.float64),
        np.array(tm.faces, dtype=np.int64),
        [],
    )


# ---------------------------------------------------------------------------
# 畸变分析
# ---------------------------------------------------------------------------


def weighted_average_vertices(
    animations: Sequence[AnimationClip],
    num_vertices: int,
    uniform_weight: bool = False,
) -> np.ndarray:
    """所有动作、所有帧的加权平均顶点位置。"""
    if not animations:
        raise ValueError("没有动画数据，无法计算加权平均")
    chunks: List[np.ndarray] = []
    weights: List[float] = []
    for anim in animations:
        anim_w = 1.0 if uniform_weight else max(anim.weight, 1e-6)
        frame_w = anim_w / len(anim.frame_vertices)
        chunks.append(anim.frame_vertices)
        weights.extend([frame_w] * len(anim.frame_vertices))
    all_frames = np.concatenate(chunks, axis=0)
    w = np.asarray(weights, dtype=np.float64)[:, None, None]
    return np.sum(all_frames * w, axis=0) / max(np.sum(weights), 1e-12)


def select_medoid_render_frame(
    mesh: MeshData,
    medoid_max_samples: Optional[int] = 128,
) -> Tuple[np.ndarray, str, int, float, float]:
    """
    选取与其它所有帧平均形状距离最小的时刻（几何 medoid）作为渲染姿态。
    返回: vertices, anim_name, frame_index, time, mean_distance
    """
    if not mesh.animations:
        return mesh.vertices.copy(), "", 0, 0.0, 0.0

    entries: List[Tuple[str, int, float, np.ndarray]] = []
    for anim in mesh.animations:
        for fi, frame in enumerate(anim.frame_vertices):
            entries.append((anim.name, fi, float(anim.frame_times[fi]), frame))

    n = len(entries)
    if n == 1:
        name, fi, t, verts = entries[0]
        return verts.copy(), name, fi, t, 0.0

    stacks = np.stack([e[3] for e in entries], axis=0)
    idx = _subsample_indices(n, medoid_max_samples)
    scores = _medoid_scores(stacks[idx])
    best_sub = int(np.argmin(scores))
    best = int(idx[best_sub])
    name, fi, t, verts = entries[best]
    return verts.copy(), name, fi, t, float(scores[best_sub])


def analyze_distortion(
    mesh: MeshData,
    reference: str = "weighted_average",
    uniform_anim_weight: bool = False,
    workers: int = 1,
    motion_skip_ratio: float = 1e-4,
    medoid_max_samples: Optional[int] = 128,
    stats: Optional[Dict[str, Any]] = None,
) -> DistortionResult:
    """
    reference: 'rest' | 'weighted_average'
    workers: 并行进程数（>1 时用 ProcessPoolExecutor 跨帧并行）
    motion_skip_ratio: 相邻关键帧最大位移 / 包围盒对角线，低于此则跳过该帧（0=关闭）
    medoid_max_samples: medoid 子采样上限
    """
    faces = mesh.faces

    if reference == "weighted_average":
        ref_v = weighted_average_vertices(mesh.animations, len(mesh.vertices), uniform_anim_weight)
    else:
        ref_v = mesh.vertices.copy()

    wavg = (
        weighted_average_vertices(mesh.animations, len(mesh.vertices), uniform_anim_weight)
        if mesh.animations
        else ref_v.copy()
    )

    topo = build_mesh_topology(ref_v, faces)
    n = len(mesh.vertices)
    ang_max = np.zeros(n, dtype=np.float64)
    area_max = np.zeros(n, dtype=np.float64)
    per_ang: Dict[str, np.ndarray] = {}
    per_area: Dict[str, np.ndarray] = {}

    if not mesh.animations:
        return DistortionResult(
            reference_vertices=ref_v,
            weighted_average_vertices=wavg,
            angular_distortion=ang_max,
            area_distortion=area_max,
            combined_distortion=ang_max,
            render_vertices=mesh.vertices.copy(),
        )

    motion_thresh = motion_skip_ratio * _bbox_diagonal(ref_v) if motion_skip_ratio > 0 else 0.0
    tasks: List[Tuple[str, np.ndarray]] = []
    frames_total = 0
    for anim in mesh.animations:
        frames = anim.frame_vertices
        frames_total += len(frames)
        idxs = _collect_animation_frame_indices(frames, motion_thresh)
        for i in idxs:
            tasks.append((anim.name, frames[int(i)]))

    per_ang = {anim.name: np.zeros(n, dtype=np.float64) for anim in mesh.animations}
    per_area = {anim.name: np.zeros(n, dtype=np.float64) for anim in mesh.animations}

    if workers > 1 and len(tasks) > 1:
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_distortion_worker,
                initargs=(ref_v, topo),
            ) as pool:
                chunk = max(1, len(tasks) // (workers * 4))
                results = pool.map(
                    _distortion_worker,
                    [frame for _, frame in tasks],
                    chunksize=chunk,
                )
            for (anim_name, _), (ang, area) in zip(tasks, results):
                per_ang[anim_name] = np.maximum(per_ang[anim_name], ang)
                per_area[anim_name] = np.maximum(per_area[anim_name], area)
        except (BrokenProcessPool, RuntimeError, OSError):
            for anim_name, frame in tasks:
                ang, area = _distortion_for_frame(ref_v, frame, topo)
                per_ang[anim_name] = np.maximum(per_ang[anim_name], ang)
                per_area[anim_name] = np.maximum(per_area[anim_name], area)
            if stats is not None:
                stats["parallel_fallback"] = True
    else:
        for anim_name, frame in tasks:
            ang, area = _distortion_for_frame(ref_v, frame, topo)
            per_ang[anim_name] = np.maximum(per_ang[anim_name], ang)
            per_area[anim_name] = np.maximum(per_area[anim_name], area)

    for anim in mesh.animations:
        ang_max = np.maximum(ang_max, per_ang[anim.name])
        area_max = np.maximum(area_max, per_area[anim.name])

    if stats is not None:
        stats["frames_total"] = frames_total
        stats["frames_evaluated"] = len(tasks)
        stats["motion_skip_ratio"] = motion_skip_ratio
        stats["motion_threshold"] = motion_thresh
        stats["workers"] = workers

    # 归一化后取 max 作为综合热力值
    ang_n = ang_max / max(ang_max.max(), 1e-12)
    area_n = area_max / max(area_max.max(), 1e-12)
    combined = np.maximum(ang_n, area_n)

    r_verts, r_anim, r_fi, r_time, r_score = select_medoid_render_frame(
        mesh, medoid_max_samples=medoid_max_samples
    )
    render_ang, render_area = _distortion_for_frame(ref_v, r_verts, topo)
    r_ang_n = render_ang / max(render_ang.max(), 1e-12)
    r_area_n = render_area / max(render_area.max(), 1e-12)
    render_combined = np.maximum(r_ang_n, r_area_n)

    return DistortionResult(
        reference_vertices=ref_v,
        weighted_average_vertices=wavg,
        angular_distortion=ang_max,
        area_distortion=area_max,
        combined_distortion=combined,
        per_animation_angular=per_ang,
        per_animation_area=per_area,
        render_vertices=r_verts,
        render_anim_name=r_anim,
        render_frame_index=r_fi,
        render_time=r_time,
        render_medoid_score=r_score,
        render_frame_angular=render_ang,
        render_frame_area=render_area,
        render_frame_combined=render_combined,
    )


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem.strip()
    safe = re.sub(r'[<>:"/\\|?*]', "_", stem)
    return safe or "mesh"


def session_output_dir(mesh_path: str | Path, base: str | Path = "output") -> Path:
    """单次分析/渲染的输出目录: output/<模型名>/"""
    return Path(base) / safe_stem(mesh_path)


def relocate_root_artifacts(project_root: str | Path | None = None) -> List[Path]:
    """将项目根目录散落的旧产物移入 output/_legacy/。"""
    root = Path(project_root or Path(__file__).resolve().parent)
    legacy = root / "output" / "_legacy"
    names = ("out.json", "imgui.ini", ".polyscope.ini")
    moved: List[Path] = []
    for name in names:
        src = root / name
        if not src.is_file():
            continue
        legacy.mkdir(parents=True, exist_ok=True)
        dst = legacy / name
        if dst.exists():
            dst = legacy / f"{src.stem}_{int(time.time())}{src.suffix}"
        src.rename(dst)
        moved.append(dst)
    return moved


def export_analysis_json(
    result: DistortionResult,
    json_path: str | Path,
    seg_mat: Optional[Dict[str, Any]] = None,
    spectral_cluster: Optional[Dict[str, Any]] = None,
) -> Path:
    payload = {
        "angular_max": float(result.angular_distortion.max()),
        "area_max": float(result.area_distortion.max()),
        "angular_mean": float(result.angular_distortion.mean()),
        "area_mean": float(result.area_distortion.mean()),
        "per_vertex_combined": result.combined_distortion.tolist(),
        "render_frame": {
            "anim": result.render_anim_name,
            "frame_index": result.render_frame_index,
            "time": result.render_time,
            "mean_distance_to_others": result.render_medoid_score,
        },
    }
    if seg_mat is not None:
        payload["seg_mat"] = seg_mat
    if spectral_cluster is not None:
        payload["spectral_cluster"] = spectral_cluster
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mesh 动画畸变分析（自动导出 JSON 并渲染热力图）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mesh", type=str, help="Mesh 文件路径 (.glb/.gltf/.fbx/.obj/...)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认 output/<模型名>/，含 analysis.json、heatmaps/、medoid_pose/、segmentation/、spectral_clustering/）",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="output",
        help="输出根目录，与 --output-dir 二选一（默认 output）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/vlm_config.yaml",
        help="VLM 配置文件（可选，供后续 VLM 集成使用）",
    )
    parser.add_argument(
        "--reference",
        choices=("weighted_average", "rest"),
        default="weighted_average",
        help="畸变参考形状",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="动画采样帧率")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="每个动作最多采样帧数（均匀降采样，可显著加速）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="畸变分析并行进程数（跨所有动作帧；默认约为 CPU 核数一半）",
    )
    parser.add_argument(
        "--motion-skip",
        type=float,
        default=1e-4,
        metavar="RATIO",
        help="连续帧运动跳过阈值（相对包围盒对角线，0=逐帧全算；默认 1e-4）",
    )
    parser.add_argument(
        "--medoid-samples",
        type=int,
        default=128,
        help="medoid 帧子采样上限（默认 128）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印各阶段耗时",
    )
    parser.add_argument(
        "--uniform-weight",
        action="store_true",
        help="所有动作等权重（默认按动作时长加权）",
    )
    parser.add_argument(
        "--no-export-json",
        action="store_true",
        help="不导出 analysis.json",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="自定义 JSON 路径（默认 <output-dir>/analysis.json）",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="跳过所有渲染",
    )
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        metavar="ELEV,AZIM",
        help="medoid_pose 统一视角，如 22.6,45（默认 22.6,45）",
    )
    parser.add_argument(
        "--spectral-clusters",
        type=int,
        default=None,
        metavar="K",
        help="谱聚类簇数 K（手动指定；省略则按特征值间隙自动估计）",
    )
    parser.add_argument(
        "--spectral-scalar-weight",
        type=float,
        default=0.35,
        metavar="W",
        help="谱聚类中畸变标量特征权重（默认 0.35）",
    )
    adv = parser.add_argument_group("高级选项（一般无需指定）")
    adv.add_argument(
        "--no-seg-mat",
        action="store_true",
        help="跳过 SEG-MAT 分割",
    )
    adv.add_argument(
        "--no-spectral-cluster",
        action="store_true",
        help="跳过谱聚类分割",
    )
    adv.add_argument(
        "--no-animation-heatmaps",
        action="store_true",
        help="跳过各动作 12 视角热力图（默认会渲染到 heatmaps/）",
    )
    adv.add_argument("--seg-mat-exe", type=str, default=None, help=argparse.SUPPRESS)
    adv.add_argument("--qmat-exe", type=str, default=None, help=argparse.SUPPRESS)
    adv.add_argument("--seg-mat-initial-ma", type=str, default=None, help=argparse.SUPPRESS)
    adv.add_argument("--seg-mat-bmat", type=str, default=None, help=argparse.SUPPRESS)
    adv.add_argument("--seg-mat-smat", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-size", type=int, default=640, help="热力图单视角分辨率")
    parser.add_argument("--cols", type=int, default=4, help="热力图拼接列数")
    parser.add_argument("--cmap", type=str, default="coolwarm", help="热力图色图")
    parser.add_argument(
        "--metrics",
        type=str,
        default="angular,area,weighted,combined",
        help="渲染指标，逗号分隔（angular,area,weighted,combined）",
    )
    args = parser.parse_args()
    if args.spectral_clusters is not None and args.spectral_clusters < 1:
        parser.error("--spectral-clusters 必须 >= 1；省略该参数则自动估计簇数")
    workers = max(1, args.workers)

    out_dir = Path(args.output_dir) if args.output_dir else session_output_dir(args.mesh, args.output_base)
    json_path = Path(args.export_json) if args.export_json else out_dir / "analysis.json"
    medoid_pose_dir = out_dir / "medoid_pose"
    segmentation_dir = out_dir / "segmentation"
    spectral_cluster_dir = out_dir / "spectral_clustering"
    heatmap_dir = out_dir / "heatmaps"

    if args.view:
        parts = [p.strip() for p in args.view.split(",")]
        if len(parts) != 2:
            parser.error("--view 格式应为 ELEV,AZIM，例如 22.6,45")
        view_elev, view_azim = float(parts[0]), float(parts[1])
    else:
        from heatmap_renderer import DEFAULT_OVERVIEW_VIEW

        view_elev, view_azim = DEFAULT_OVERVIEW_VIEW

    def _log(msg: str) -> None:
        if args.verbose:
            print(msg)

    t0 = time.perf_counter()
    relocated = relocate_root_artifacts()
    for p in relocated:
        print(f"[cleanup] 已移入 {p}")

    if Path(args.config).exists():
        cfg = load_vlm_config(args.config)
        print(f"[config] 已加载 VLM 配置，active_provider={cfg.get('active_provider')}")

    print(f"[load] {args.mesh}")
    mesh = load_mesh(args.mesh, samples_per_second=args.fps, max_frames=args.max_frames)
    _log(f"  加载耗时: {time.perf_counter() - t0:.2f}s")
    print(
        f"  顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}, "
        f"动画: {len(mesh.animations)}"
    )
    for a in mesh.animations:
        print(f"    - {a.name}: {a.duration:.3f}s, {len(a.frame_times)} frames")

    if not mesh.animations:
        print("[warn] 未检测到动画")

    t1 = time.perf_counter()
    run_stats: Dict[str, Any] = {}
    result = analyze_distortion(
        mesh,
        reference=args.reference,
        uniform_anim_weight=args.uniform_weight,
        workers=workers,
        motion_skip_ratio=args.motion_skip,
        medoid_max_samples=args.medoid_samples,
        stats=run_stats,
    )
    _log(f"  畸变分析耗时: {time.perf_counter() - t1:.2f}s (workers={workers})")
    if run_stats:
        ev = run_stats.get("frames_evaluated", 0)
        tot = run_stats.get("frames_total", 0)
        if tot:
            _log(f"  评估帧: {ev}/{tot} ({100.0 * ev / tot:.1f}%)")
        if args.motion_skip > 0:
            _log(f"  运动跳过阈值: {run_stats.get('motion_threshold', 0):.6g}")

    if run_stats and run_stats.get("frames_total", 0):
        ev = int(run_stats["frames_evaluated"])
        tot = int(run_stats["frames_total"])
        if ev < tot:
            print(f"[speed] 运动跳过: 评估 {ev}/{tot} 帧 ({100.0 * ev / tot:.1f}%)")

    print(
        f"[distortion] angular max={result.angular_distortion.max():.6f}, "
        f"area max={result.area_distortion.max():.6f}"
    )
    if mesh.animations and result.render_vertices is not None:
        print(
            f"[medoid] 全局 medoid 帧: {result.render_anim_name} "
            f"#{result.render_frame_index} t={result.render_time:.3f}s "
            f"(mean_dist={result.render_medoid_score:.6f})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir.resolve()}")

    seg_mat_summary: Optional[Dict[str, Any]] = None
    seg_face_labels: Optional[List[int]] = None
    seg_num_segments = 0
    if not args.no_seg_mat and result.render_vertices is not None:
        from seg_mat_runner import run_seg_mat_on_render_pose

        print(f"[seg-mat] -> {medoid_pose_dir.resolve()}")
        seg_result = run_seg_mat_on_render_pose(
            result.render_vertices,
            mesh.faces,
            medoid_pose_dir,
            source_mesh=args.mesh,
            seg_mat_exe=args.seg_mat_exe,
            qmat_exe=args.qmat_exe,
            initial_ma=args.seg_mat_initial_ma,
            base_ma=args.seg_mat_bmat,
            smat_ma=args.seg_mat_smat,
            skeleton=mesh.skeleton,
            skeleton_anim_name=result.render_anim_name or "",
            skeleton_time=float(result.render_time or 0.0),
        )
        if seg_result.ok:
            print(f"[seg-mat] {seg_result.message}")
            seg_face_labels = seg_result.face_labels
            seg_num_segments = seg_result.num_segments
            seg_mat_summary = {
                "ok": True,
                "num_segments": seg_result.num_segments,
                "initial_ma": str(seg_result.initial_ma) if seg_result.initial_ma else None,
                "initial_ma_source": seg_result.extra.get("initial_ma_source"),
                "mesh_off": str(seg_result.mesh_off) if seg_result.mesh_off else None,
                "seg_off": str(seg_result.seg_off) if seg_result.seg_off else None,
                "labels_json": seg_result.extra.get("labels_json"),
            }
        else:
            print(f"[seg-mat] {seg_result.message}")
            seg_mat_summary = {"ok": False, "message": seg_result.message}

    spectral_cluster_summary: Optional[Dict[str, Any]] = None
    if not args.no_spectral_cluster and result.render_vertices is not None:
        from spectral_cluster import run_spectral_clustering_on_render_pose

        n_clusters = args.spectral_clusters
        print(f"[spectral] -> {spectral_cluster_dir.resolve()}")
        if n_clusters is not None:
            print(f"[spectral] 手动指定簇数: {n_clusters}")
        else:
            print("[spectral] 簇数: 自动估计")
        t_spectral = time.perf_counter()
        spectral_result = run_spectral_clustering_on_render_pose(
            result.render_vertices,
            mesh.faces,
            spectral_cluster_dir,
            n_clusters=n_clusters,
            vertex_scalars=result.render_frame_combined,
            scalar_weight=args.spectral_scalar_weight,
            render=not args.no_render,
            image_size=args.image_size,
            cols=args.cols,
            render_anim_name=result.render_anim_name or "",
            render_frame_index=result.render_frame_index,
            render_time=float(result.render_time or 0.0),
            render_medoid_score=result.render_medoid_score,
        )
        _log(f"  谱聚类耗时: {time.perf_counter() - t_spectral:.2f}s")
        if spectral_result.ok:
            print(f"[spectral] {spectral_result.message}")
            if spectral_result.panel_path:
                print(f"[render] {spectral_result.panel_path}")
            spectral_cluster_summary = {
                "ok": True,
                "num_clusters": spectral_result.num_clusters,
                "requested_clusters": n_clusters,
                "labels_json": spectral_result.extra.get("labels_json"),
                "panel_png": spectral_result.extra.get("panel_png"),
                "overview_png": spectral_result.extra.get("overview_png"),
                "auto_clusters": n_clusters is None,
            }
        else:
            print(f"[spectral] {spectral_result.message}")
            spectral_cluster_summary = {"ok": False, "message": spectral_result.message}

    if (
        not args.no_render
        and result.render_vertices is not None
        and seg_face_labels is not None
        and len(seg_face_labels) == len(mesh.faces)
    ):
        from heatmap_renderer import render_medoid_segmentation_12view

        t_seg_render = time.perf_counter()
        seg_panel = render_medoid_segmentation_12view(
            result.render_vertices,
            mesh.faces,
            seg_face_labels,
            segmentation_dir,
            render_anim_name=result.render_anim_name,
            render_frame_index=result.render_frame_index,
            render_time=result.render_time,
            render_medoid_score=result.render_medoid_score,
            num_segments=seg_num_segments,
            image_size=args.image_size,
            cols=args.cols,
        )
        _log(f"  segmentation 12 视角耗时: {time.perf_counter() - t_seg_render:.2f}s")
        print(f"[render] {seg_panel}")
        if seg_mat_summary and seg_mat_summary.get("ok"):
            seg_mat_summary["segmentation_12view"] = str(seg_panel)

    if not args.no_export_json:
        export_analysis_json(
            result,
            json_path,
            seg_mat=seg_mat_summary,
            spectral_cluster=spectral_cluster_summary,
        )
        print(f"[export] {json_path.resolve()}")

    if not args.no_render and result.render_vertices is not None:
        from heatmap_renderer import render_medoid_pose_overview

        ang = result.render_frame_angular
        area = result.render_frame_area
        if ang.size == 0:
            ang = np.zeros(len(mesh.vertices), dtype=np.float64)
        if area.size == 0:
            area = np.zeros(len(mesh.vertices), dtype=np.float64)

        t2 = time.perf_counter()
        overview = render_medoid_pose_overview(
            result.render_vertices,
            mesh.faces,
            ang,
            area,
            medoid_pose_dir,
            face_labels=seg_face_labels,
            elevation_deg=view_elev,
            azimuth_deg=view_azim,
            image_size=args.image_size,
            cmap=args.cmap,
            render_anim_name=result.render_anim_name,
            render_frame_index=result.render_frame_index,
            render_time=result.render_time,
            render_medoid_score=result.render_medoid_score,
            num_segments=seg_num_segments,
        )
        _log(f"  medoid_pose 渲染耗时: {time.perf_counter() - t2:.2f}s")
        print(f"[render] {overview.composite}  (视角 elev={view_elev}, azim={view_azim})")

    if not args.no_render and not args.no_animation_heatmaps:
        from heatmap_renderer import render_all_animations_from_mesh

        t3 = time.perf_counter()
        metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
        paths = render_all_animations_from_mesh(
            mesh,
            heatmap_dir,
            reference=args.reference,
            uniform_anim_weight=args.uniform_weight,
            image_size=args.image_size,
            cols=args.cols,
            cmap=args.cmap,
            medoid_max_samples=args.medoid_samples,
            metrics=metric_list,
        )
        _log(f"  各动作热力图耗时: {time.perf_counter() - t3:.2f}s ({len(paths)} 张)")
        if paths:
            folders = sorted({p.parent.name for p in paths})
            print(f"[render] heatmaps/ 下已生成: {', '.join(folders)}/")

    _log(f"  总耗时: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
