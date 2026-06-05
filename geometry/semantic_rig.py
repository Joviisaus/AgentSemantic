"""
把"粗糙的带语义分块"适配成一具可变形骨架（Semantic Rig）。

输入是 full_coverage 产出的 ``face_semantic_labels.json`` 那种逐面语义标签：
每个面属于某个语义部位（头/躯干/手臂/腿/底座/武器…）。本模块把每个语义
部位当作一根**刚性骨头**，在相邻部位的公共接缝处建关节，自动估计蒙皮权重，
并提供一个自洽的线性混合蒙皮（LBS）变形函数。

这一层是"语义块 → 可变形骨架"的桥：
    - ``build_part_adjacency``   : 由面邻接算出部位之间的接缝（共享边/顶点）
    - ``compute_seam_joints``    : 每对相邻部位的接缝质心 = 关节位置
    - ``build_bone_hierarchy``   : 由邻接图选根并生成骨骼父子树（生成森林）
    - ``assign_skinning_weights``: 刚性权重 + 接缝过渡带平滑混合
    - ``SemanticSkeleton.pose``  : 给每根骨一个绕自身关节的旋转，做 LBS 变形

骨架不依赖 glTF；变形结果可以直接喂给 ``mesh_distortion_analyzer`` 的
``AnimationClip`` 做后续畸变分析（闭环优化分块的第二步）。

设计取舍（原型阶段）：
    - 一个语义部位 = 一根骨，即使它在几何上不连通也合并成一根。
    - 关节自由度先用任意 3D 旋转；轴/角度约束留给后续 LLM 拓扑步骤。
    - 权重用稠密 (V, B)（B = 部位数，通常很小），便于归一化与调试。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # trimesh 已是项目依赖，用它的快速面邻接
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise ImportError("semantic_rig 需要 trimesh: pip install trimesh") from exc


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class SemanticJoint:
    """两根相邻骨头之间的关节。"""

    parent_bone: int          # 父骨在 bones 列表里的索引
    child_bone: int           # 子骨索引
    position: np.ndarray      # (3,) 关节世界坐标（接缝质心）
    seam_vertex_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "parent_bone": int(self.parent_bone),
            "child_bone": int(self.child_bone),
            "position": [float(x) for x in self.position],
            "num_seam_vertices": len(self.seam_vertex_ids),
        }


@dataclass
class SemanticBone:
    """一根骨头 = 一个语义部位。"""

    index: int
    name: str
    part_index: int            # 对应 semantic_parts 里的下标
    parent: int = -1           # 父骨索引；-1 表示是某棵树的根
    children: List[int] = field(default_factory=list)
    joint_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    face_ids: List[int] = field(default_factory=list)
    # 关节旋转轴（世界系单位向量）。为零向量时表示"未指定"，
    # 由 SemanticSkeleton.bone_axis 按几何默认推断（垂直于骨向）。
    rotation_axis: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": int(self.index),
            "name": self.name,
            "part_index": int(self.part_index),
            "parent": int(self.parent),
            "children": [int(c) for c in self.children],
            "joint_position": [float(x) for x in self.joint_position],
            "rotation_axis": [float(x) for x in self.rotation_axis],
            "num_faces": len(self.face_ids),
        }


@dataclass
class Keyframe:
    """某一时刻每根骨绕自身轴的旋转角（度）。未列出的骨视为 0°。"""

    t: float
    angles: Dict[int, float] = field(default_factory=dict)  # bone_index -> deg


@dataclass
class RigAnimation:
    """一段由关键帧角度描述的动画。"""

    name: str
    fps: float
    duration: float
    keyframes: List[Keyframe] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "fps": self.fps,
            "duration": self.duration,
            "keyframes": [
                {"t": float(k.t), "angles": {int(b): float(a) for b, a in k.angles.items()}}
                for k in self.keyframes
            ],
        }


@dataclass
class SemanticSkeleton:
    """语义骨架 + 蒙皮权重 + LBS 变形。

    顶点变形遵循标准 LBS：每根骨有一个"静止全局变换" = 平移到关节位置；
    给定每根骨绕自身关节的旋转后，沿父子树求姿态全局变换，蒙皮矩阵
    ``skin_b = M_pose_b @ inv(M_rest_b)``，顶点 = Σ_b w[v,b] · (skin_b · v)。
    静止姿态（全部旋转为单位）下 skin_b = I，顶点保持不变。
    """

    vertices: np.ndarray              # (V, 3) 静止/绑定姿态
    faces: np.ndarray                 # (F, 3)
    semantic_parts: List[str]
    bones: List[SemanticBone]
    joints: List[SemanticJoint]
    roots: List[int]                  # 森林的根骨索引
    weights: np.ndarray               # (V, B) 行和为 1
    vertex_bone: np.ndarray           # (V,) 主导骨索引（用于上色/调试）

    # -- 查询辅助 -----------------------------------------------------------

    def bone_by_name(self, name: str) -> Optional[SemanticBone]:
        for b in self.bones:
            if b.name == name:
                return b
        return None

    def bone_index_of_part(self, part_index: int) -> Optional[int]:
        for b in self.bones:
            if b.part_index == part_index:
                return b.index
        return None

    def bones_of_part(self, part_index: int) -> List[int]:
        """该语义部位对应的所有骨（同部位多连通块时不止一根）。"""
        return [b.index for b in self.bones if b.part_index == part_index]

    @property
    def num_bones(self) -> int:
        return len(self.bones)

    def _offsets(self) -> np.ndarray:
        """每根骨相对父骨关节的平移；根骨为其自身关节绝对位置。"""
        off = np.zeros((self.num_bones, 3), dtype=np.float64)
        for b in self.bones:
            if b.parent < 0:
                off[b.index] = b.joint_position
            else:
                off[b.index] = b.joint_position - self.bones[b.parent].joint_position
        return off

    # -- 变形 ---------------------------------------------------------------

    def pose(self, rotations: Dict[int, np.ndarray]) -> np.ndarray:
        """按给定每根骨的旋转做 LBS 变形，返回 (V, 3) 形变后顶点。

        参数
        ----
        rotations: {bone_index: R} ，R 为 (3,3) 旋转矩阵。缺省的骨视为单位旋转。
        """
        n = self.num_bones
        offsets = self._offsets()

        def rot4(b: int) -> np.ndarray:
            m = np.eye(4)
            r = rotations.get(b)
            if r is not None:
                m[:3, :3] = np.asarray(r, dtype=np.float64)
            return m

        def trans4(t: np.ndarray) -> np.ndarray:
            m = np.eye(4)
            m[:3, 3] = t
            return m

        # 姿态全局变换：M_b = M_parent @ T(offset_b) @ R(b)
        local_pose = [trans4(offsets[i]) @ rot4(i) for i in range(n)]
        global_pose: List[Optional[np.ndarray]] = [None] * n

        def dfs(i: int) -> None:
            p = self.bones[i].parent
            global_pose[i] = (
                local_pose[i] if p < 0 else global_pose[p] @ local_pose[i]
            )
            for c in self.bones[i].children:
                dfs(c)

        for r in self.roots:
            dfs(r)

        # 静止全局变换 = 平移到关节位置；蒙皮矩阵 = M_pose @ T(-joint)
        vh = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
        out = np.zeros((len(self.vertices), 3), dtype=np.float64)
        for i in range(n):
            gp = global_pose[i]
            assert gp is not None  # 森林覆盖所有骨
            skin = gp @ trans4(-self.bones[i].joint_position)
            w = self.weights[:, i]
            sel = w > 1e-9
            if not np.any(sel):
                continue
            transformed = (skin @ vh[sel].T).T[:, :3]
            out[sel] += w[sel, None] * transformed
        return out

    def pose_axis_angle(
        self,
        bone_rotations: Sequence[Tuple[int, Sequence[float], float]],
    ) -> np.ndarray:
        """便捷接口：用 [(bone_index, axis(3,), angle_deg), ...] 描述姿态。"""
        rmap = {b: axis_angle_matrix(ax, ang) for b, ax, ang in bone_rotations}
        return self.pose(rmap)

    # -- 关节轴 / 角度动画 --------------------------------------------------

    def bone_axis(self, bone_index: int) -> np.ndarray:
        """返回骨的旋转轴：优先用显式 rotation_axis，否则按几何默认推断
        （垂直于 关节→骨质心 方向、尽量绕侧向，使其像"弯曲"而非"自转"）。"""
        b = self.bones[bone_index]
        ax = np.asarray(b.rotation_axis, dtype=np.float64)
        if float(np.linalg.norm(ax)) > 1e-9:
            return ax / np.linalg.norm(ax)
        face_ids = np.asarray(b.face_ids, dtype=np.int64)
        if face_ids.size:
            vids = np.unique(self.faces[face_ids].reshape(-1))
            bone_dir = self.vertices[vids].mean(axis=0) - b.joint_position
        else:
            bone_dir = np.array([0.0, 1.0, 0.0])
        if np.linalg.norm(bone_dir) < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        axis = np.cross(bone_dir, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(bone_dir, np.array([1.0, 0.0, 0.0]))
        return axis / max(float(np.linalg.norm(axis)), 1e-12)

    def pose_from_angles(self, angles: Dict[int, float]) -> np.ndarray:
        """按 {bone_index: 角度(度)} 绕各骨自身轴旋转，做 LBS 变形。"""
        rmap: Dict[int, np.ndarray] = {}
        for bi, ang in angles.items():
            if abs(float(ang)) < 1e-9:
                continue
            rmap[int(bi)] = axis_angle_matrix(self.bone_axis(int(bi)), float(ang))
        return self.pose(rmap)

    def sample_animation(
        self,
        anim: "RigAnimation",
    ) -> List[Tuple[float, np.ndarray]]:
        """把关键帧角度按 fps 线性插值，烘焙出 [(time, vertices), ...]。"""
        if not anim.keyframes:
            return [(0.0, self.pose({}))]
        kfs = sorted(anim.keyframes, key=lambda k: k.t)
        duration = anim.duration or kfs[-1].t
        fps = anim.fps if anim.fps > 0 else 24.0
        n_frames = max(2, int(round(duration * fps)) + 1)
        times = np.linspace(0.0, duration, n_frames)
        all_bones = sorted({b for k in kfs for b in k.angles.keys()})

        def angle_at(bone: int, t: float) -> float:
            ts = [k.t for k in kfs]
            vs = [float(k.angles.get(bone, 0.0)) for k in kfs]
            if t <= ts[0]:
                return vs[0]
            if t >= ts[-1]:
                return vs[-1]
            j = int(np.searchsorted(ts, t)) - 1
            j = max(0, min(j, len(ts) - 2))
            t0, t1 = ts[j], ts[j + 1]
            w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            return (1 - w) * vs[j] + w * vs[j + 1]

        frames: List[Tuple[float, np.ndarray]] = []
        for t in times:
            angles = {b: angle_at(b, float(t)) for b in all_bones}
            frames.append((float(t), self.pose_from_angles(angles)))
        return frames

    def to_dict(self) -> Dict[str, object]:
        return {
            "semantic_parts": list(self.semantic_parts),
            "num_vertices": int(len(self.vertices)),
            "num_faces": int(len(self.faces)),
            "roots": [int(r) for r in self.roots],
            "bones": [b.to_dict() for b in self.bones],
            "joints": [j.to_dict() for j in self.joints],
        }

    def hierarchy_lines(self) -> List[str]:
        """人类可读的层级文本（缩进树）。"""
        lines: List[str] = []

        def walk(i: int, depth: int) -> None:
            b = self.bones[i]
            n_w = int(np.count_nonzero(self.vertex_bone == i))
            lines.append(f"{'  ' * depth}- {b.name} (bone#{i}, {n_w} verts)")
            for c in b.children:
                walk(c, depth + 1)

        for r in self.roots:
            walk(r, 0)
        return lines


# ---------------------------------------------------------------------------
# 旋转辅助
# ---------------------------------------------------------------------------


def axis_angle_matrix(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    """Rodrigues：绕单位轴 axis 旋转 angle_deg 度的 (3,3) 矩阵。"""
    a = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        return np.eye(3)
    a = a / norm
    th = np.deg2rad(angle_deg)
    x, y, z = a
    c, s = np.cos(th), np.sin(th)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


# ---------------------------------------------------------------------------
# 1) 部位邻接 + 接缝
# ---------------------------------------------------------------------------


def _face_part_majority_vertex_labels(
    faces: np.ndarray,
    face_part_index: np.ndarray,
    num_parts: int,
) -> np.ndarray:
    """把逐面标签投到逐顶点：每个顶点取其相邻面中出现最多的有效部位。

    无有效标签的顶点记为 -1，留给最近质心兜底。
    """
    num_v = int(faces.max()) + 1 if len(faces) else 0
    counts = np.zeros((num_v, num_parts), dtype=np.int64)
    valid = face_part_index >= 0
    for li in range(3):
        vids = faces[valid, li]
        parts = face_part_index[valid]
        np.add.at(counts, (vids, parts), 1)
    vbone = np.full(num_v, -1, dtype=np.int64)
    has = counts.sum(axis=1) > 0
    vbone[has] = counts[has].argmax(axis=1)
    return vbone


def build_part_adjacency(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_part_index: np.ndarray,
) -> Dict[Tuple[int, int], List[int]]:
    """返回 {(part_a, part_b): [接缝顶点 id, ...]}，part_a < part_b。

    用 trimesh 的面邻接（共享边的面对）找出"两侧面属于不同部位"的边，
    收集这些边的顶点作为接缝。
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    fa = np.asarray(mesh.face_adjacency)            # (E, 2) 面对
    if len(fa) == 0:
        return {}
    fae = np.asarray(mesh.face_adjacency_edges)     # (E, 2) 共享边顶点
    pa = face_part_index[fa]                         # (E, 2) 两侧部位
    seam = (pa[:, 0] != pa[:, 1]) & (pa[:, 0] >= 0) & (pa[:, 1] >= 0)
    out: Dict[Tuple[int, int], set] = defaultdict(set)
    for (p0, p1), (v0, v1) in zip(pa[seam], fae[seam]):
        key = (int(min(p0, p1)), int(max(p0, p1)))
        out[key].add(int(v0))
        out[key].add(int(v1))
    return {k: sorted(v) for k, v in out.items()}


def compute_seam_joints(
    vertices: np.ndarray,
    seams: Dict[Tuple[int, int], List[int]],
) -> Dict[Tuple[int, int], np.ndarray]:
    """每条接缝的顶点质心作为关节候选位置。"""
    joints: Dict[Tuple[int, int], np.ndarray] = {}
    for key, vids in seams.items():
        if not vids:
            continue
        joints[key] = vertices[np.asarray(vids, dtype=np.int64)].mean(axis=0)
    return joints


# ---------------------------------------------------------------------------
# 2) 骨骼层级（最大生成森林）
# ---------------------------------------------------------------------------


def build_bone_hierarchy(
    present_parts: Sequence[int],
    seams: Dict[Tuple[int, int], List[int]],
    part_face_counts: Dict[int, int],
    root_part: Optional[int] = None,
) -> Tuple[Dict[int, int], List[int]]:
    """由部位邻接图生成父子关系（BFS 生成森林）。

    返回 (parent_part: {part: parent_part(-1=root)}, root_parts)。
    根选择：显式 root_part 优先；否则取"邻接度数最高"的部位，平票按面数。
    接缝权重用接缝顶点数（越多越像主连接），保证更稳的生成树。
    """
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # part -> [(nbr, weight)]
    for (a, b), vids in seams.items():
        w = len(vids)
        adj[a].append((b, w))
        adj[b].append((a, w))

    present = list(present_parts)
    present_set = set(present)
    parent: Dict[int, int] = {p: -1 for p in present}
    roots: List[int] = []
    visited: set = set()

    def degree(p: int) -> int:
        return len(adj.get(p, []))

    # 按"度数→面数"排序，依次作为各连通分量的根
    order = sorted(
        present,
        key=lambda p: (degree(p), part_face_counts.get(p, 0)),
        reverse=True,
    )
    if root_part is not None and root_part in present_set:
        order = [root_part] + [p for p in order if p != root_part]

    for start in order:
        if start in visited:
            continue
        roots.append(start)
        visited.add(start)
        parent[start] = -1
        # BFS：优先沿接缝顶点多的边扩展
        q: deque = deque([start])
        while q:
            cur = q.popleft()
            nbrs = sorted(adj.get(cur, []), key=lambda nw: nw[1], reverse=True)
            for nbr, _w in nbrs:
                if nbr in visited or nbr not in present_set:
                    continue
                visited.add(nbr)
                parent[nbr] = cur
                q.append(nbr)
    return parent, roots


def _hierarchy_from_override(
    present_parts: Sequence[int],
    parent_override: Dict[int, int],
) -> Tuple[Dict[int, int], List[int]]:
    """根据显式 {part: parent_part} 生成 (parent_part, roots)，并校验成树。

    - 只保留 present_parts 内的边；父部位不在场则降级为根。
    - 检测环：沿父链回溯若回到自身则切断该边（降级为根）并保证可用。
    """
    present = list(present_parts)
    present_set = set(present)
    parent: Dict[int, int] = {}
    for p in present:
        pp = int(parent_override.get(p, -1))
        if pp == p or pp not in present_set:
            pp = -1
        parent[p] = pp

    # 断环：任意节点沿父链若超过 N 步或回到自身 → 设为根
    for p in present:
        seen = set()
        cur = p
        steps = 0
        while parent.get(cur, -1) >= 0:
            cur = parent[cur]
            steps += 1
            if cur in seen or cur == p or steps > len(present) + 1:
                parent[p] = -1
                break
            seen.add(cur)

    roots = [p for p in present if parent[p] < 0]
    if not roots:  # 极端情况下全成环被断光，兜底取第一个
        roots = [present[0]]
        parent[present[0]] = -1
    return parent, roots


# ---------------------------------------------------------------------------
# 连通分量：把一个语义部位拆成若干几何连通块（如狗的 4 条腿 -> 4 根骨）
# ---------------------------------------------------------------------------


def _label_face_components(
    fpi: np.ndarray,
    fa: np.ndarray,
    n_faces: int,
    min_component_faces: int,
) -> Tuple[np.ndarray, List[int]]:
    """把同部位的面按"共享边连通"切成分量。

    返回 (comp_of_face (F,), comp_part)：comp_of_face[i] = 该面所属分量 id
    （未标注/被丢弃的面为 -1）；comp_part[c] = 分量 c 的部位下标。
    小于 min_component_faces 的分量会并入"共享边最多"的邻接分量（去噪）。
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    a, b = fa[:, 0], fa[:, 1]
    same = (fpi[a] == fpi[b]) & (fpi[a] >= 0)
    ea, eb = a[same], b[same]
    rows = np.concatenate([ea, eb])
    cols = np.concatenate([eb, ea])
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(n_faces, n_faces)).tocsr()
    _, lab = connected_components(graph, directed=False)
    lab = lab.astype(np.int64)

    if min_component_faces and min_component_faces > 1:
        for _ in range(8):
            sizes = np.bincount(lab)
            tiny = set(int(x) for x in np.where(sizes < min_component_faces)[0])
            if not tiny:
                break
            diff = lab[a] != lab[b]
            if not diff.any():
                break
            ca, cb = lab[a[diff]], lab[b[diff]]
            nb: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
            for x, y in zip(ca.tolist(), cb.tolist()):
                nb[x][y] += 1
                nb[y][x] += 1
            remap: Dict[int, int] = {}
            for c in tiny:
                cand = nb.get(c)
                if not cand:
                    continue
                # 优先并入"非小且共享边最多"的邻居
                remap[c] = max(
                    cand.items(),
                    key=lambda kv: (0 if kv[0] in tiny else 1, kv[1]),
                )[0]
            if not remap:
                break

            def _root(c: int) -> int:
                seen: set = set()
                while c in remap and c not in seen:
                    seen.add(c)
                    c = remap[c]
                return c

            new = lab.copy()
            for c in list(remap.keys()):
                new[lab == c] = _root(c)
            if np.array_equal(new, lab):
                break
            lab = new

    comp_of_face = np.full(n_faces, -1, dtype=np.int64)
    comp_part: List[int] = []
    next_id = 0
    for old in np.unique(lab).tolist():
        mask = lab == old
        valid = fpi[mask]
        valid = valid[valid >= 0]
        if valid.size == 0:
            continue  # 整块都没标签 → 留作 -1，权重阶段最近质心兜底
        comp_of_face[mask] = next_id
        comp_part.append(int(np.bincount(valid).argmax()))
        next_id += 1
    return comp_of_face, comp_part


def connected_components_per_part(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    min_component_faces: int = 20,
) -> Tuple[np.ndarray, List[int]]:
    """把每个语义部位按共享边切成连通块（与 build_semantic_skeleton 同口径）。

    返回 (comp_of_face (F,), comp_part)：comp_of_face[i] = 面 i 的连通块 id（未标注
    /被丢弃为 -1）；comp_part[c] = 块 c 的部位下标。供"同部位不连通块"的下游使用
    （如 VLM 细分命名），保证命名与最终建骨的分块一致。
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    fa = np.asarray(mesh.face_adjacency)
    if len(fa) == 0:
        # 无邻接（如未合并顶点）→ 一部位一块
        present = sorted(int(p) for p in np.unique(fpi) if int(p) >= 0)
        remap = {p: i for i, p in enumerate(present)}
        comp_of_face = np.array([remap.get(int(p), -1) for p in fpi], dtype=np.int64)
        return comp_of_face, list(present)
    return _label_face_components(fpi, fa, len(faces), min_component_faces)


def _component_seams(
    vertices: np.ndarray,
    comp_of_face: np.ndarray,
    fa: np.ndarray,
    fae: np.ndarray,
) -> Dict[Tuple[int, int], List[int]]:
    """分量级接缝：两侧面属于不同分量（且都有效）的边的顶点集合。"""
    ca, cb = comp_of_face[fa[:, 0]], comp_of_face[fa[:, 1]]
    seam = (ca != cb) & (ca >= 0) & (cb >= 0)
    out: Dict[Tuple[int, int], set] = defaultdict(set)
    for (c0, c1), (v0, v1) in zip(np.stack([ca[seam], cb[seam]], 1), fae[seam]):
        key = (int(min(c0, c1)), int(max(c0, c1)))
        out[key].add(int(v0))
        out[key].add(int(v1))
    return {k: sorted(v) for k, v in out.items()}


def _comp_parent_from_part_override(
    present_comps: Sequence[int],
    comp_part: Sequence[int],
    comp_seams: Dict[Tuple[int, int], List[int]],
    comp_centroids: Dict[int, np.ndarray],
    parent_override: Dict[int, int],
    root_part_idx: Optional[int],
) -> Dict[int, int]:
    """把"部位级"父子（LLM 给的 part->parent_part）落到"分量级"父子。

    每个子分量的父 = 与它相邻、且属于目标父部位、共享接缝最大的那个分量；
    若无相邻同部位分量，则取该父部位中质心最近的分量。
    """
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for (ca, cb), vids in comp_seams.items():
        adj[ca].append((cb, len(vids)))
        adj[cb].append((ca, len(vids)))

    parent: Dict[int, int] = {}
    for c in present_comps:
        p = comp_part[c]
        pp = int(parent_override.get(p, -1))
        if (root_part_idx is not None and p == root_part_idx) or pp < 0:
            parent[c] = -1
            continue
        cands = [(nb, sz) for nb, sz in adj.get(c, []) if comp_part[nb] == pp]
        if cands:
            parent[c] = max(cands, key=lambda x: x[1])[0]
        else:
            pp_comps = [cc for cc in present_comps if comp_part[cc] == pp]
            if pp_comps:
                d = [float(np.linalg.norm(comp_centroids[cc] - comp_centroids[c]))
                     for cc in pp_comps]
                parent[c] = pp_comps[int(np.argmin(d))]
            else:
                parent[c] = -1
    return parent


# ---------------------------------------------------------------------------
# 3) 蒙皮权重（刚性 + 接缝过渡带）
# ---------------------------------------------------------------------------


def _nearest_seam_distance(
    vertices: np.ndarray,
    query_idx: np.ndarray,
    seam_vertex_ids: Sequence[int],
) -> np.ndarray:
    """query 顶点到接缝顶点集合的最近距离。"""
    seam_pts = vertices[np.asarray(seam_vertex_ids, dtype=np.int64)]
    if len(seam_pts) == 0:
        return np.full(len(query_idx), np.inf)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(seam_pts)
        d, _ = tree.query(vertices[query_idx], k=1)
        return d
    except ImportError:  # pragma: no cover - scipy 是项目依赖
        q = vertices[query_idx]
        d2 = ((q[:, None, :] - seam_pts[None, :, :]) ** 2).sum(axis=2)
        return np.sqrt(d2.min(axis=1))


def blend_skinning_at_seams(
    vertices: np.ndarray,
    weights: np.ndarray,
    joints: List[SemanticJoint],
    blend_radius: float,
) -> np.ndarray:
    """在已有权重上沿接缝带做双侧骨过渡（与刚性蒙皮相同规则）。"""
    w = np.asarray(weights, dtype=np.float64).copy()
    if blend_radius <= 0 or w.size == 0:
        return w
    dominant = w.argmax(axis=1)
    has_w = w.max(axis=1) > 1e-12
    for j in joints:
        b1, b2 = j.parent_bone, j.child_bone
        if not j.seam_vertex_ids:
            continue
        sel = ((dominant == b1) | (dominant == b2)) & has_w
        idx = np.where(sel)[0]
        if idx.size == 0:
            continue
        d = _nearest_seam_distance(vertices, idx, j.seam_vertex_ids)
        near = d < blend_radius
        idx, d = idx[near], d[near]
        if idx.size == 0:
            continue
        transfer = 0.5 * (1.0 - d / blend_radius)
        own = dominant[idx]
        dst = np.where(own == b1, b2, b1)
        moved = w[idx, own] * transfer
        w[idx, own] -= moved
        w[idx, dst] += moved
    row = w.sum(axis=1)
    nz = row > 1e-12
    w[nz] /= row[nz, None]
    return w


def assign_skinning_weights(
    vertices: np.ndarray,
    vertex_part: np.ndarray,
    part_to_bone: Dict[int, int],
    num_bones: int,
    joints: List[SemanticJoint],
    blend_radius: float,
) -> np.ndarray:
    """生成 (V, B) 权重。

    基础：顶点按其主导部位硬绑定到对应骨（权重 1）。
    过渡：对每个关节 (b1,b2)，把"到接缝线距离 < blend_radius"且属于 b1 或 b2
    的顶点向对侧骨转移权重，转移量 = 0.5·(1 - d/R)（d=到最近接缝顶点距离），
    使接缝处趋近 50/50、远处保持刚性，从而弯曲时不撕裂、不硬折。
    （注意：用"到接缝线"的距离而非"到关节质心"，因为宽环接缝的质心通常
    深埋体内，离实际切缝很远。）多关节叠加后整体重新归一化。
    """
    num_v = len(vertices)
    w = np.zeros((num_v, num_bones), dtype=np.float64)

    for vi in range(num_v):
        p = int(vertex_part[vi])
        if p >= 0 and p in part_to_bone:
            w[vi, part_to_bone[p]] = 1.0

    return blend_skinning_at_seams(vertices, w, joints, blend_radius)


# ---------------------------------------------------------------------------
# 顶层构造
# ---------------------------------------------------------------------------


def build_semantic_skeleton(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    semantic_parts: Sequence[str],
    *,
    root_part: Optional[object] = None,
    blend_radius_ratio: float = 0.06,
    parent_override: Optional[Dict[int, int]] = None,
    axis_override: Optional[Dict[int, Sequence[float]]] = None,
    split_components: bool = True,
    min_component_faces: int = 20,
) -> SemanticSkeleton:
    """从逐面语义标签构造一具可变形语义骨架。

    每个"骨"对应一个**几何连通块**：同一语义部位若有多个不连通区域（如狗的 4
    条腿都叫"腿"），会拆成多根骨，各自在自己的接缝处建关节，互不牵连。

    参数
    ----
    vertices, faces        : 网格
    face_semantic_index    : (F,) 每个面的语义部位下标（<0 视为未标注）
    semantic_parts         : 部位名列表
    root_part              : 根部位，可传名字或下标；None=自动（度数最高）
    blend_radius_ratio     : 接缝过渡带半径占 bbox 对角线的比例（默认 6%）
    parent_override        : {part_index: parent_part_index(-1=root)}，部位级层级
                             （如 LLM 给出的拓扑）；会按邻接落到各连通块上。
    axis_override          : {part_index: [x,y,z]}，该部位所有连通块的关节轴。
    split_components       : True=把不连通的同部位拆成多根骨（默认）；
                             False=一个部位一根骨（旧行为）。
    min_component_faces    : 小于此面数的连通块并入相邻块去噪（默认 20）。
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    fpi = np.asarray(face_semantic_index, dtype=np.int64)
    if len(fpi) != len(faces):
        raise ValueError(
            f"face_semantic_index 长度 {len(fpi)} 与 faces {len(faces)} 不一致"
        )
    num_parts = len(semantic_parts)
    if not np.any((fpi >= 0) & (fpi < num_parts)):
        raise ValueError("没有任何有效的语义面标签，无法建骨架")

    # 面邻接（一次性算出，供连通块标注 + 接缝复用）
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    fa = np.asarray(mesh.face_adjacency)
    fae = (np.asarray(mesh.face_adjacency_edges) if len(fa)
           else np.empty((0, 2), dtype=np.int64))

    # 1) 连通块（= 骨）。split_components=False 时退化为"一部位一块"。
    if split_components and len(fa):
        comp_of_face, comp_part = _label_face_components(
            fpi, fa, len(faces), min_component_faces
        )
    else:
        present_parts = sorted(int(p) for p in np.unique(fpi)
                               if 0 <= int(p) < num_parts)
        remap = {p: i for i, p in enumerate(present_parts)}
        comp_of_face = np.array(
            [remap.get(int(p), -1) for p in fpi], dtype=np.int64
        )
        comp_part = list(present_parts)

    n_comp = len(comp_part)
    if n_comp == 0:
        raise ValueError("连通块标注后无有效骨")
    present = list(range(n_comp))

    # 各块面/质心/接缝
    comp_face_counts: Dict[int, int] = {}
    centroids: Dict[int, np.ndarray] = {}
    comp_face_ids: Dict[int, np.ndarray] = {}
    for c in present:
        fids = np.where(comp_of_face == c)[0]
        comp_face_ids[c] = fids
        comp_face_counts[c] = int(len(fids))
        vids = np.unique(faces[fids].reshape(-1)) if len(fids) else np.array([], int)
        centroids[c] = vertices[vids].mean(axis=0) if len(vids) else np.zeros(3)

    seams = _component_seams(vertices, comp_of_face, fa, fae)
    seam_joint_pos = compute_seam_joints(vertices, seams)

    # 2) 层级（部位级根/override → 落到连通块）
    root_idx: Optional[int] = None
    if isinstance(root_part, str):
        if root_part in semantic_parts:
            root_idx = semantic_parts.index(root_part)
    elif isinstance(root_part, (int, np.integer)):
        root_idx = int(root_part)

    if parent_override is not None:
        comp_parent_in = _comp_parent_from_part_override(
            present, comp_part, seams, centroids, parent_override, root_idx
        )
        parent_map, root_comps = _hierarchy_from_override(present, comp_parent_in)
    else:
        root_comp: Optional[int] = None
        if root_idx is not None:
            cands = [c for c in present if comp_part[c] == root_idx]
            if cands:
                root_comp = max(cands, key=lambda c: comp_face_counts[c])
        parent_map, root_comps = build_bone_hierarchy(
            present, seams, comp_face_counts, root_part=root_comp
        )

    # 3) 建骨（同部位多块时名字加后缀）
    part_comp_count: Dict[int, int] = defaultdict(int)
    for p in comp_part:
        part_comp_count[p] += 1
    part_seen: Dict[int, int] = defaultdict(int)
    bones: List[SemanticBone] = []
    for c in present:
        p = comp_part[c]
        base = semantic_parts[p]
        if part_comp_count[p] > 1:
            name = f"{base}_{part_seen[p]}"
            part_seen[p] += 1
        else:
            name = base
        bones.append(SemanticBone(
            index=c, name=name, part_index=p,
            face_ids=[int(x) for x in comp_face_ids[c]],
        ))

    def _joint_pos_for(c: int, pc: int) -> np.ndarray:
        key = (min(c, pc), max(c, pc))
        jp = seam_joint_pos.get(key)
        return jp if jp is not None else 0.5 * (centroids[c] + centroids[pc])

    for c in present:
        pc = parent_map[c]
        if pc < 0:
            bones[c].parent = -1
            bones[c].joint_position = centroids[c]
        else:
            bones[c].parent = pc
            bones[pc].children.append(c)
            bones[c].joint_position = _joint_pos_for(c, pc)
    roots = list(root_comps)

    # 显式旋转轴：部位级 → 该部位所有块
    if axis_override:
        for c in present:
            ax = axis_override.get(comp_part[c])
            if ax is not None:
                bones[c].rotation_axis = np.asarray(ax, dtype=np.float64)

    joints: List[SemanticJoint] = []
    for c in present:
        pc = parent_map[c]
        if pc < 0:
            continue
        key = (min(c, pc), max(c, pc))
        joints.append(SemanticJoint(
            parent_bone=pc, child_bone=c,
            position=_joint_pos_for(c, pc),
            seam_vertex_ids=seams.get(key, []),
        ))

    # 4) 权重（连通块级）
    vertex_comp = _face_part_majority_vertex_labels(faces, comp_of_face, n_comp)
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    blend_radius = max(bbox_diag * blend_radius_ratio, 1e-9)
    weights = assign_skinning_weights(
        vertices, vertex_comp, {c: c for c in present}, n_comp, joints, blend_radius
    )

    row = weights.sum(axis=1)
    orphan = np.where(row <= 1e-12)[0]
    if orphan.size:
        cen = np.stack([centroids[c] for c in present])
        for vi in orphan:
            d = np.linalg.norm(cen - vertices[vi][None, :], axis=1)
            weights[vi, int(d.argmin())] = 1.0

    vertex_bone = weights.argmax(axis=1).astype(np.int64)

    return SemanticSkeleton(
        vertices=vertices,
        faces=faces,
        semantic_parts=list(semantic_parts),
        bones=bones,
        joints=joints,
        roots=roots,
        weights=weights,
        vertex_bone=vertex_bone,
    )


# ---------------------------------------------------------------------------
# 测试姿态：每根"叶子骨"绕关节摆一下，用来激活所有关节
# ---------------------------------------------------------------------------


def make_test_pose(
    skeleton: SemanticSkeleton,
    angle_deg: float = 25.0,
) -> Dict[int, np.ndarray]:
    """给每根非根骨一个绕"垂直于骨向"的小旋转，制造一个明显的检验姿态。

    旋转轴取 (父关节→子骨质心) 与全局 up 的叉积，保证是"弯曲"而非"自转"。
    """
    rmap: Dict[int, np.ndarray] = {}
    up = np.array([0.0, 1.0, 0.0])
    for b in skeleton.bones:
        if b.parent < 0:
            continue
        face_ids = np.asarray(b.face_ids, dtype=np.int64)
        if face_ids.size == 0:
            continue
        vids = np.unique(skeleton.faces[face_ids].reshape(-1))
        child_centroid = skeleton.vertices[vids].mean(axis=0)
        bone_dir = child_centroid - b.joint_position
        if np.linalg.norm(bone_dir) < 1e-9:
            continue
        axis = np.cross(bone_dir, up)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(bone_dir, np.array([1.0, 0.0, 0.0]))
        # 交替正负角，避免所有肢体往同一边叠
        sign = 1.0 if (b.index % 2 == 0) else -1.0
        rmap[b.index] = axis_angle_matrix(axis, angle_deg * sign)
    return rmap


def make_test_animation(
    skeleton: SemanticSkeleton,
    angle_deg: float = 25.0,
    *,
    name: str = "test",
    fps: float = 24.0,
) -> RigAnimation:
    """构造一段「每根非根骨 0→±angle→0」的检验动画（角度版，用于 GLB 导出）。"""
    angles: Dict[int, float] = {}
    for b in skeleton.bones:
        if b.parent < 0 or not b.face_ids:
            continue
        angles[b.index] = angle_deg * (1.0 if (b.index % 2 == 0) else -1.0)
    return RigAnimation(
        name=name,
        fps=fps,
        duration=2.0,
        keyframes=[
            Keyframe(t=0.0, angles={}),
            Keyframe(t=1.0, angles=dict(angles)),
            Keyframe(t=2.0, angles={}),
        ],
    )


# ---------------------------------------------------------------------------
# Rig 规格（LLM 给定的骨架层级 + 关节轴 + 动画关键帧）的解析
# ---------------------------------------------------------------------------


def match_part_name(name: object, semantic_parts: Sequence[str]) -> Optional[int]:
    """把一个（可能来自 LLM 的）部位名对回到 semantic_parts 的下标。

    支持：整数下标、精确名、去空格大小写不敏感、子串。匹配失败返回 None。
    """
    if isinstance(name, (int, np.integer)):
        i = int(name)
        return i if 0 <= i < len(semantic_parts) else None
    s = str(name or "").strip()
    if not s:
        return None
    for i, p in enumerate(semantic_parts):
        if s == p:
            return i
    sn = s.lower().replace(" ", "").replace("　", "")
    for i, p in enumerate(semantic_parts):
        if sn == p.lower().replace(" ", "").replace("　", ""):
            return i
    for i, p in enumerate(semantic_parts):
        if len(p) >= 2 and (p in s or s in p):
            return i
    return None


@dataclass
class ResolvedRigSpec:
    """rig-spec 解析结果（已把部位名映射成 part_index）。"""

    object_kind: str = ""
    root_part: Optional[int] = None
    parent_override: Dict[int, int] = field(default_factory=dict)
    axis_override: Dict[int, List[float]] = field(default_factory=dict)
    # 动画的角度键此时仍是 part_index；建好骨架后用 build_animations 转 bone_index
    animations: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def resolve_rig_spec(
    spec: Dict[str, object],
    semantic_parts: Sequence[str],
) -> ResolvedRigSpec:
    """解析 rig-spec（LLM 输出或手写 JSON）。

    期望 schema：
        {
          "object_kind": "...",
          "root": "躯干",
          "bones": [{"name":"手臂","parent":"躯干","axis":[0,0,1]}, ...],
          "animations": [
            {"name":"wave","fps":24,"duration":2.0,
             "keyframes":[{"t":0.0,"angles":{"手臂":0}},
                          {"t":1.0,"angles":{"手臂":80,"头部":-15}}]}
          ]
        }
    角度单位为度，绕该骨的 axis 旋转；未列出的骨/关键帧角度按 0 处理。
    """
    out = ResolvedRigSpec()
    out.object_kind = str(spec.get("object_kind") or "").strip()

    root_idx = match_part_name(spec.get("root"), semantic_parts)
    out.root_part = root_idx

    bones = spec.get("bones") or []
    if isinstance(bones, list):
        for entry in bones:
            if not isinstance(entry, dict):
                continue
            pi = match_part_name(entry.get("name"), semantic_parts)
            if pi is None:
                out.warnings.append(f"bones 中无法识别的部位: {entry.get('name')!r}")
                continue
            parent_raw = entry.get("parent", None)
            if parent_raw in (None, "", "root", "ROOT", -1):
                out.parent_override[pi] = -1
            else:
                ppi = match_part_name(parent_raw, semantic_parts)
                out.parent_override[pi] = ppi if ppi is not None else -1
                if ppi is None:
                    out.warnings.append(
                        f"{entry.get('name')!r} 的父 {parent_raw!r} 未识别 → 设为根"
                    )
            ax = entry.get("axis")
            if isinstance(ax, (list, tuple)) and len(ax) == 3:
                try:
                    out.axis_override[pi] = [float(ax[0]), float(ax[1]), float(ax[2])]
                except (TypeError, ValueError):
                    pass

    # root 兜底：若 bones 没把 root 标成 -1，强制之
    if root_idx is not None:
        out.parent_override[root_idx] = -1

    anims = spec.get("animations") or []
    if isinstance(anims, list):
        for a in anims:
            if not isinstance(a, dict):
                continue
            kfs_in = a.get("keyframes") or []
            kfs: List[Dict[str, object]] = []
            for k in kfs_in:
                if not isinstance(k, dict):
                    continue
                angles_in = k.get("angles") or {}
                angles: Dict[int, float] = {}
                if isinstance(angles_in, dict):
                    for nm, deg in angles_in.items():
                        pi = match_part_name(nm, semantic_parts)
                        if pi is None:
                            continue
                        try:
                            angles[pi] = float(deg)
                        except (TypeError, ValueError):
                            continue
                try:
                    t = float(k.get("t", 0.0))
                except (TypeError, ValueError):
                    t = 0.0
                kfs.append({"t": t, "angles": angles})
            if not kfs:
                continue
            try:
                fps = float(a.get("fps", 24.0)) or 24.0
            except (TypeError, ValueError):
                fps = 24.0
            dur = a.get("duration")
            try:
                duration = float(dur) if dur is not None else max(k["t"] for k in kfs)
            except (TypeError, ValueError):
                duration = max(k["t"] for k in kfs)
            out.animations.append(
                {
                    "name": str(a.get("name") or f"anim_{len(out.animations)}"),
                    "fps": fps,
                    "duration": duration,
                    "keyframes": kfs,
                }
            )
    return out


def build_animations(
    resolved: ResolvedRigSpec,
    skeleton: SemanticSkeleton,
) -> List[RigAnimation]:
    """把解析出的（part_index 键）动画转成以 bone_index 为键的 RigAnimation。"""
    out: List[RigAnimation] = []
    for a in resolved.animations:
        kfs: List[Keyframe] = []
        for k in a["keyframes"]:  # type: ignore[index]
            angles_bone: Dict[int, float] = {}
            for pi, deg in k["angles"].items():  # type: ignore[index]
                # 同部位多连通块 → 同一角度施加到该部位的每根骨
                for bi in skeleton.bones_of_part(int(pi)):
                    angles_bone[bi] = float(deg)
            kfs.append(Keyframe(t=float(k["t"]), angles=angles_bone))  # type: ignore[index]
        out.append(
            RigAnimation(
                name=str(a["name"]),
                fps=float(a["fps"]),  # type: ignore[arg-type]
                duration=float(a["duration"]),  # type: ignore[arg-type]
                keyframes=kfs,
            )
        )
    return out
