"""
点云输入：两条来源统一成 (N, 3) float64 点数组。

1. 从 mesh 表面均匀采样（沿用现有 .off/.glb/.obj/.ply/.stl... 输入，trimesh 采样）；
2. 直接读裸点云文件 .npy / .ply / .pcd / .xyz / .txt。

输出仅含坐标（可选返回法向，留作后续渲染/特征用）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise ImportError("点云 IO 需要 trimesh: pip install trimesh") from exc


_MESH_EXTS = {".off", ".glb", ".gltf", ".fbx", ".obj", ".stl", ".dae", ".3mf"}
_RAW_PC_EXTS = {".npy", ".pcd", ".xyz", ".txt"}
# .ply 既可能是 mesh 也可能是裸点云，单独探测。


def _as_xyz(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"点云数组形状非法: {a.shape}，需要 (N, >=3)")
    return np.ascontiguousarray(a[:, :3])


def sample_points_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    *,
    seed: Optional[int] = 0,
    include_vertices: bool = False,
) -> np.ndarray:
    """从三角网格表面按面积均匀采样 num_points 个点。"""
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if seed is not None:
        np.random.seed(int(seed))
    pts, _face_idx = trimesh.sample.sample_surface(mesh, int(num_points))
    pts = np.asarray(pts, dtype=np.float64)
    if include_vertices:
        pts = np.vstack([pts, np.asarray(vertices, dtype=np.float64)])
    return _as_xyz(pts)


def _load_raw_pointcloud(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return _as_xyz(np.load(str(path)))
    if ext in (".xyz", ".txt"):
        return _as_xyz(np.loadtxt(str(path)))
    # .pcd / .ply 走 trimesh 的 PointCloud 解析
    obj = trimesh.load(str(path), file_type=ext.lstrip("."))
    if isinstance(obj, trimesh.PointCloud):
        return _as_xyz(np.asarray(obj.vertices, dtype=np.float64))
    if isinstance(obj, trimesh.Trimesh):
        return _as_xyz(np.asarray(obj.vertices, dtype=np.float64))
    raise ValueError(f"无法把 {path} 解析为点云，得到 {type(obj)}")


def load_pointcloud(
    path: str | Path,
    *,
    num_sample_points: int = 20000,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, str]:
    """
    统一点云加载入口。返回 (points[N,3], source)，source ∈ {"mesh_sampled", "raw"}。

    - mesh 类扩展名（.off/.glb/.obj/.stl...）→ 表面均匀采样 num_sample_points 个点；
    - 裸点云扩展名（.npy/.pcd/.xyz/.txt）→ 直接读坐标；
    - .ply → 先尝试当裸点云读；若实际是带面的 mesh 则改为表面采样。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"点云/mesh 文件不存在: {path}")
    ext = path.suffix.lower()

    if ext in _RAW_PC_EXTS:
        return _load_raw_pointcloud(path), "raw"

    if ext in _MESH_EXTS:
        obj = trimesh.load(str(path), file_type=ext.lstrip("."))
        verts, faces = _mesh_arrays(obj, path)
        return (
            sample_points_from_mesh(verts, faces, num_sample_points, seed=seed),
            "mesh_sampled",
        )

    if ext == ".ply":
        obj = trimesh.load(str(path), file_type="ply")
        if isinstance(obj, trimesh.Trimesh) and len(obj.faces) > 0:
            return (
                sample_points_from_mesh(
                    np.asarray(obj.vertices), np.asarray(obj.faces),
                    num_sample_points, seed=seed,
                ),
                "mesh_sampled",
            )
        if isinstance(obj, trimesh.PointCloud):
            return _as_xyz(np.asarray(obj.vertices, dtype=np.float64)), "raw"
        return _as_xyz(np.asarray(obj.vertices, dtype=np.float64)), "raw"

    # 兜底：交给 trimesh 自动探测
    obj = trimesh.load(str(path))
    if isinstance(obj, trimesh.Trimesh) and len(obj.faces) > 0:
        return (
            sample_points_from_mesh(
                np.asarray(obj.vertices), np.asarray(obj.faces),
                num_sample_points, seed=seed,
            ),
            "mesh_sampled",
        )
    return _as_xyz(np.asarray(obj.vertices, dtype=np.float64)), "raw"


def _mesh_arrays(obj: object, path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"{path} 中找不到 trimesh.Trimesh")
        obj = trimesh.util.concatenate(geoms)
    if not isinstance(obj, trimesh.Trimesh):
        raise ValueError(f"{path} 不是三角网格，得到 {type(obj)}")
    return (
        np.asarray(obj.vertices, dtype=np.float64),
        np.asarray(obj.faces, dtype=np.int64),
    )
