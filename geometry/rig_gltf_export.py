"""
把 SemanticSkeleton（骨架 + 蒙皮权重 + 关键帧动画）导出成**带蒙皮的 GLB**，
可在任意 glTF 查看器里直接点播放，无需逐帧 .off。

glTF 约定与本项目 LBS 的一致性：
    - 每根骨对应一个 joint 节点，节点 local 平移 = 该骨关节相对父关节的偏移，
      绑定姿态旋转为单位四元数；
    - inverseBindMatrix_j = 平移 -joint_position_j（绑定全局变换为纯平移的逆）；
    - 运行时 skinMat_j = globalPose_j · IBM_j，静止时 = I（顶点不动），与
      SemanticSkeleton.pose 完全一致；
    - 动画通道写 joint 节点的 rotation 四元数关键帧：绕该骨 axis 旋转 angle。
      因绑定旋转为单位阵，骨的世界轴即其父-局部轴，可直接作为 local rotation。

只用 pygltflib + numpy，按 GLB 单 buffer 二进制块手工打包。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from pygltflib import (
        GLTF2,
        Accessor,
        Animation,
        AnimationChannel,
        AnimationChannelTarget,
        AnimationSampler,
        Asset,
        Attributes,
        Buffer,
        BufferView,
        Mesh,
        Node,
        Primitive,
        Scene,
        Skin,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("GLB 导出需要 pygltflib: pip install pygltflib") from exc

from .semantic_rig import RigAnimation, SemanticSkeleton

try:
    from .skin_optimize import SkinOptimizeConfig, optimize_semantic_skin
except ImportError:  # pragma: no cover
    SkinOptimizeConfig = None  # type: ignore[misc, assignment]
    optimize_semantic_skin = None  # type: ignore[misc, assignment]

# glTF 常量
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125
_FLOAT = 5126


def _axis_angle_quat(axis: np.ndarray, angle_deg: float) -> Tuple[float, float, float, float]:
    """轴角 -> 四元数 (x, y, z, w)。"""
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-12 or abs(angle_deg) < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    a = a / n
    h = np.deg2rad(angle_deg) * 0.5
    s = np.sin(h)
    return (float(a[0] * s), float(a[1] * s), float(a[2] * s), float(np.cos(h)))


def _top4_skin_weights(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """稠密 (V, B) 权重 -> glTF 的 (JOINTS_0 uint16 (V,4), WEIGHTS_0 float32 (V,4))。"""
    num_v, num_b = weights.shape
    k = min(4, num_b)
    order = np.argsort(-weights, axis=1)[:, :k]          # (V, k) 最大的 k 个骨
    joints = np.zeros((num_v, 4), dtype=np.uint16)
    wt = np.zeros((num_v, 4), dtype=np.float32)
    rows = np.arange(num_v)[:, None]
    joints[:, :k] = order.astype(np.uint16)
    wt[:, :k] = weights[rows, order].astype(np.float32)
    s = wt.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    wt = wt / s
    return joints, wt


class _BinPacker:
    """把多个数组顺序写入一个 buffer，4 字节对齐，记录每块 (offset, length)。"""

    def __init__(self) -> None:
        self.buf = bytearray()

    def add(self, arr: np.ndarray) -> Tuple[int, int]:
        data = np.ascontiguousarray(arr).tobytes()
        offset = len(self.buf)
        self.buf.extend(data)
        # 对齐到 4 字节
        pad = (-len(self.buf)) % 4
        if pad:
            self.buf.extend(b"\x00" * pad)
        return offset, len(data)


def export_skinned_glb(
    skeleton: SemanticSkeleton,
    animations: Sequence[RigAnimation],
    out_path: str | Path,
    *,
    optimize_skin: bool = False,
    skin_opt_config: "SkinOptimizeConfig | None" = None,
    mo_animations: Optional[Sequence[RigAnimation]] = None,
    test_angle_deg: Optional[float] = None,
) -> Path:
    """导出带蒙皮 + 动画的 GLB。

    optimize_skin=True 时先跑 ``geometry.skin_optimize``；R_mo 使用 ``mo_animations``
    或 ``animations`` 中的给定关键帧姿态。
    """
    if optimize_skin:
        if optimize_semantic_skin is None:
            raise ImportError(
                "optimize_skin 需要 geometry.skin_optimize（scipy + trimesh）"
            )
        mo_anims = mo_animations if mo_animations is not None else animations
        optimize_semantic_skin(
            skeleton,
            skin_opt_config,
            animations=mo_anims,
            test_angle_deg=test_angle_deg,
            report_dir=Path(out_path).parent,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(skeleton.vertices, dtype=np.float32)
    faces = np.asarray(skeleton.faces, dtype=np.uint32)
    n_bones = skeleton.num_bones

    joints_arr, weights_arr = _top4_skin_weights(skeleton.weights)

    # inverseBindMatrices：列优先的 T(-joint_position)
    ibm = np.zeros((n_bones, 16), dtype=np.float32)
    offsets = skeleton._offsets()  # noqa: SLF001 - 复用本项目内部辅助
    for b in skeleton.bones:
        jp = np.asarray(b.joint_position, dtype=np.float64)
        m = np.eye(4)
        m[:3, 3] = -jp
        ibm[b.index] = m.T.reshape(-1).astype(np.float32)  # 列优先

    packer = _BinPacker()
    accessors: List[Accessor] = []
    buffer_views: List[BufferView] = []

    def _bv(offset: int, length: int, target: int | None) -> int:
        buffer_views.append(
            BufferView(buffer=0, byteOffset=offset, byteLength=length, target=target)
        )
        return len(buffer_views) - 1

    # indices
    off, ln = packer.add(faces.reshape(-1))
    idx_bv = _bv(off, ln, _ELEMENT_ARRAY_BUFFER)
    accessors.append(Accessor(bufferView=idx_bv, componentType=_UNSIGNED_INT,
                              count=int(faces.size), type="SCALAR"))
    idx_acc = len(accessors) - 1

    # POSITION
    off, ln = packer.add(verts)
    pos_bv = _bv(off, ln, _ARRAY_BUFFER)
    accessors.append(Accessor(
        bufferView=pos_bv, componentType=_FLOAT, count=int(len(verts)), type="VEC3",
        min=[float(x) for x in verts.min(axis=0)],
        max=[float(x) for x in verts.max(axis=0)],
    ))
    pos_acc = len(accessors) - 1

    # JOINTS_0
    off, ln = packer.add(joints_arr)
    j_bv = _bv(off, ln, _ARRAY_BUFFER)
    accessors.append(Accessor(bufferView=j_bv, componentType=_UNSIGNED_SHORT,
                              count=int(len(joints_arr)), type="VEC4"))
    joints_acc = len(accessors) - 1

    # WEIGHTS_0
    off, ln = packer.add(weights_arr)
    w_bv = _bv(off, ln, _ARRAY_BUFFER)
    accessors.append(Accessor(bufferView=w_bv, componentType=_FLOAT,
                              count=int(len(weights_arr)), type="VEC4"))
    weights_acc = len(accessors) - 1

    # inverseBindMatrices
    off, ln = packer.add(ibm)
    ibm_bv = _bv(off, ln, None)
    accessors.append(Accessor(bufferView=ibm_bv, componentType=_FLOAT,
                              count=n_bones, type="MAT4"))
    ibm_acc = len(accessors) - 1

    # ---- 节点：joint 节点 (0..n_bones-1) + mesh 节点 (n_bones) ----
    nodes: List[Node] = []
    for b in skeleton.bones:
        nodes.append(Node(
            translation=[float(x) for x in offsets[b.index]],
            rotation=[0.0, 0.0, 0.0, 1.0],
            children=[int(c) for c in b.children] or None,
            name=f"joint_{b.index}_{b.name}",
        ))
    mesh_node_idx = len(nodes)
    nodes.append(Node(mesh=0, skin=0, name="skinned_mesh"))

    # ---- 动画 ----
    gltf_animations: List[Animation] = []
    for anim in animations:
        kfs = sorted(anim.keyframes, key=lambda k: k.t)
        if not kfs:
            continue
        times = np.array([k.t for k in kfs], dtype=np.float32)
        animated_bones = sorted({b for k in kfs for b in k.angles.keys()})
        if not animated_bones:
            continue
        samplers: List[AnimationSampler] = []
        channels: List[AnimationChannel] = []
        # 共享一个 time accessor
        off, ln = packer.add(times)
        time_bv = _bv(off, ln, None)
        accessors.append(Accessor(
            bufferView=time_bv, componentType=_FLOAT, count=len(times), type="SCALAR",
            min=[float(times.min())], max=[float(times.max())],
        ))
        time_acc = len(accessors) - 1

        for bone in animated_bones:
            axis = skeleton.bone_axis(bone)
            quats = np.array(
                [_axis_angle_quat(axis, float(k.angles.get(bone, 0.0))) for k in kfs],
                dtype=np.float32,
            )
            off, ln = packer.add(quats)
            q_bv = _bv(off, ln, None)
            accessors.append(Accessor(bufferView=q_bv, componentType=_FLOAT,
                                      count=len(quats), type="VEC4"))
            q_acc = len(accessors) - 1
            samplers.append(AnimationSampler(input=time_acc, output=q_acc,
                                             interpolation="LINEAR"))
            channels.append(AnimationChannel(
                sampler=len(samplers) - 1,
                target=AnimationChannelTarget(node=int(bone), path="rotation"),
            ))
        gltf_animations.append(Animation(name=anim.name, samplers=samplers,
                                         channels=channels))

    # ---- 组装 ----
    primitive = Primitive(
        attributes=Attributes(POSITION=pos_acc, JOINTS_0=joints_acc, WEIGHTS_0=weights_acc),
        indices=idx_acc,
    )
    skin = Skin(joints=[b.index for b in skeleton.bones],
                inverseBindMatrices=ibm_acc,
                skeleton=skeleton.roots[0] if skeleton.roots else 0)

    blob = bytes(packer.buf)
    gltf = GLTF2(
        asset=Asset(version="2.0", generator="AgentSematic semantic_rig"),
        scene=0,
        scenes=[Scene(nodes=list(skeleton.roots) + [mesh_node_idx])],
        nodes=nodes,
        meshes=[Mesh(primitives=[primitive])],
        skins=[skin],
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[Buffer(byteLength=len(blob))],
        animations=gltf_animations,
    )
    gltf.set_binary_blob(blob)
    gltf.save_binary(str(out_path))
    return out_path
