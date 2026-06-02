"""
Mesh 热力图离线渲染：每个动作单独输出 12 视角拼接图。

依赖 pyvista（离屏渲染）+ Pillow（拼图），无需 GUI。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from geometry.mesh_distortion_analyzer import (
    MeshData,
    MeshTopology,
    build_mesh_topology,
    compute_angular_distortion,
    compute_area_distortion,
    load_mesh,
    select_medoid_for_clip,
    session_output_dir,
    weighted_average_vertices,
)

try:
    import pyvista as pv
except ImportError as exc:
    pv = None  # type: ignore
    _PV_IMPORT_ERROR = exc
else:
    _PV_IMPORT_ERROR = None

try:
    from PIL import Image, ImageDraw
except ImportError as exc:
    Image = None  # type: ignore
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None


DEFAULT_CMAP = "coolwarm"
DEFAULT_CAMERA_FOV_DEG = 30.0
DEFAULT_CAMERA_MARGIN = 1.55
DEFAULT_OVERVIEW_VIEW: Tuple[float, float] = (22.6, 45.0)  # elev, azim — 与 VIEW_GRID 同系
VIEW_COLS = 4
METRIC_WEIGHT_ANGULAR = 0.5
METRIC_WEIGHT_AREA = 0.5

# SEG-MAT 默认调色板（与 third_party/SEG-MAT/src/main.cpp 一致）
SEG_MAT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (166, 206, 227),
    (31, 120, 180),
    (178, 223, 138),
    (51, 160, 44),
    (251, 154, 153),
    (227, 26, 28),
    (253, 191, 111),
    (255, 127, 0),
    (166, 216, 84),
    (202, 178, 214),
    (106, 61, 154),
    (255, 255, 153),
    (177, 89, 40),
    (102, 194, 165),
    (252, 141, 98),
    (141, 160, 203),
    (231, 138, 195),
    (166, 216, 84),
    (255, 217, 47),
)

# 指标名 -> (子文件夹, 标题前缀)
RENDER_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("angular", "angular", "角度畸变"),
    ("area", "area", "面积畸变"),
    ("weighted", "weighted", "加权畸变"),
    ("combined", "combined", "综合畸变(max)"),
)


def _icosahedron_view_grid(cols: int = VIEW_COLS) -> Tuple[Tuple[float, float], ...]:
    """
    正二十面体 12 顶点 → 球面上最均匀的 12 视角。
    按仰角从高到低、方位角递增排列，便于 4×3 拼图。
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = (
        (0, 1, phi),
        (0, -1, phi),
        (0, 1, -phi),
        (0, -1, -phi),
        (1, phi, 0),
        (-1, phi, 0),
        (1, -phi, 0),
        (-1, -phi, 0),
        (phi, 0, 1),
        (-phi, 0, 1),
        (phi, 0, -1),
        (-phi, 0, -1),
    )
    views: List[Tuple[float, float]] = []
    for x, y, z in raw:
        v = np.array([x, y, z], dtype=np.float64)
        v /= np.linalg.norm(v)
        elev = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
        azim = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
        views.append((round(elev, 1), round(azim, 1)))
    views.sort(key=lambda t: (-t[0], t[1]))
    return tuple(views)


# 12 视角：正二十面体均匀采样，4 列 × 3 行
VIEW_GRID: Tuple[Tuple[float, float], ...] = _icosahedron_view_grid(VIEW_COLS)


@dataclass
class MedoidPoseRenderResult:
    output_dir: Path
    composite: Path
    angular: Path
    area: Path
    segmentation: Optional[Path]
    meta_json: Path
    segmentation_12view: Optional[Path] = None


@dataclass
class AnimationRenderPack:
    name: str
    vertices: np.ndarray
    angular: np.ndarray
    area: np.ndarray
    weighted: np.ndarray
    combined: np.ndarray
    frame_index: int
    time: float
    medoid_score: float

    def scalars_for(self, metric: str) -> np.ndarray:
        table = {
            "angular": self.angular,
            "area": self.area,
            "weighted": self.weighted,
            "combined": self.combined,
        }
        if metric not in table:
            raise ValueError(f"未知指标: {metric}")
        return table[metric]


def _safe_filename(name: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    return s or "animation"


def compute_frame_heatmap(
    ref_vertices: np.ndarray,
    frame_vertices: np.ndarray,
    faces: np.ndarray,
    topo: Optional[MeshTopology] = None,
    weight_angular: float = METRIC_WEIGHT_ANGULAR,
    weight_area: float = METRIC_WEIGHT_AREA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算单帧热力值：angular, area, weighted, combined（均为 [0,1] 归一化标量）。"""
    if topo is None:
        topo = build_mesh_topology(ref_vertices, faces)
    ang = compute_angular_distortion(ref_vertices, frame_vertices, topo.faces, topo=topo)
    area = compute_area_distortion(ref_vertices, frame_vertices, topo.faces, topo=topo)
    ang_n = ang / max(float(ang.max()), 1e-12)
    area_n = area / max(float(area.max()), 1e-12)
    w_sum = max(weight_angular + weight_area, 1e-12)
    weighted = (weight_angular * ang_n + weight_area * area_n) / w_sum
    combined = np.maximum(ang_n, area_n)
    return ang_n, area_n, weighted, combined


def _to_pyvista_faces(faces: np.ndarray) -> np.ndarray:
    return np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)])


def _mesh_center_and_radius(vertices: np.ndarray) -> Tuple[np.ndarray, float]:
    center = np.asarray(vertices.mean(axis=0), dtype=np.float64)
    radii = np.linalg.norm(vertices - center, axis=1)
    radius = float(radii.max()) if radii.size else 1e-3
    return center, max(radius, 1e-3)


def _camera_distance(
    sphere_radius: float,
    elevation_deg: float,
    view_angle_deg: float = DEFAULT_CAMERA_FOV_DEG,
    margin: float = DEFAULT_CAMERA_MARGIN,
) -> float:
    """根据包围球与视角 FOV 计算相机距离，保证模型完整入画。"""
    el = np.deg2rad(abs(elevation_deg))
    # 斜视时投影范围变大，略增有效半径
    effective_r = sphere_radius * (1.0 + 0.35 * np.sin(el))
    half_fov = np.deg2rad(view_angle_deg) / 2.0
    return effective_r / max(np.tan(half_fov), 1e-6) * margin


def _camera_position(
    center: np.ndarray, radius: float, elevation_deg: float, azimuth_deg: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    offset = radius * np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float64,
    )
    cam = center + offset
    return (tuple(cam), tuple(center), (0.0, 0.0, 1.0))


def prepare_animation_packs(
    mesh: MeshData,
    reference: str = "weighted_average",
    uniform_anim_weight: bool = False,
    medoid_max_samples: Optional[int] = 128,
) -> Tuple[List[AnimationRenderPack], np.ndarray]:
    if not mesh.animations:
        verts = mesh.vertices.copy()
        topo = build_mesh_topology(verts, mesh.faces)
        ang_n, area_n, weighted, combined = compute_frame_heatmap(verts, verts, mesh.faces, topo)
        return [
            AnimationRenderPack(
                "rest", verts, ang_n, area_n, weighted, combined, 0, 0.0, 0.0
            )
        ], mesh.faces

    if reference == "weighted_average":
        ref_v = weighted_average_vertices(mesh.animations, len(mesh.vertices), uniform_anim_weight)
    else:
        ref_v = mesh.vertices.copy()
    topo = build_mesh_topology(ref_v, mesh.faces)

    packs: List[AnimationRenderPack] = []
    for anim in mesh.animations:
        verts, fi, t, score = select_medoid_for_clip(anim, medoid_max_samples)
        ang_n, area_n, weighted, combined = compute_frame_heatmap(ref_v, verts, mesh.faces, topo)
        packs.append(
            AnimationRenderPack(
                name=anim.name,
                vertices=verts,
                angular=ang_n,
                area=area_n,
                weighted=weighted,
                combined=combined,
                frame_index=fi,
                time=t,
                medoid_score=score,
            )
        )
    return packs, mesh.faces


def _apply_camera(
    plotter: "pv.Plotter",
    vertices: np.ndarray,
    elevation_deg: float,
    azimuth_deg: float,
) -> None:
    center, sphere_r = _mesh_center_and_radius(vertices)
    dist = _camera_distance(sphere_r, elevation_deg)
    plotter.camera_position = _camera_position(center, dist, elevation_deg, azimuth_deg)
    plotter.camera.zoom(0.9)


def face_labels_to_rgb(face_labels: Sequence[int]) -> np.ndarray:
    """面片标签 -> (F, 3) uint8 RGB。"""
    out = np.zeros((len(face_labels), 3), dtype=np.uint8)
    for i, label in enumerate(face_labels):
        if label < 0:
            out[i] = (200, 200, 200)
        else:
            out[i] = SEG_MAT_COLORS[int(label) % len(SEG_MAT_COLORS)]
    return out


def render_single_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    elevation_deg: float,
    azimuth_deg: float,
    image_size: Tuple[int, int] = (640, 640),
    cmap: str = DEFAULT_CMAP,
) -> np.ndarray:
    if pv is None:
        raise ImportError("需要 pyvista: pip install pyvista") from _PV_IMPORT_ERROR

    pv_faces = _to_pyvista_faces(faces)
    pdata = pv.PolyData(vertices, pv_faces)
    pdata["distortion"] = scalars.astype(np.float64)

    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    plotter.set_background("white")
    plotter.add_mesh(
        pdata,
        scalars="distortion",
        cmap=cmap,
        clim=(0.0, 1.0),
        show_scalar_bar=False,
        smooth_shading=True,
    )
    _apply_camera(plotter, vertices, elevation_deg, azimuth_deg)
    plotter.enable_anti_aliasing("ssaa")
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def render_segmentation_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    elevation_deg: float,
    azimuth_deg: float,
    image_size: Tuple[int, int] = (640, 640),
) -> np.ndarray:
    if pv is None:
        raise ImportError("需要 pyvista: pip install pyvista") from _PV_IMPORT_ERROR

    pv_faces = _to_pyvista_faces(faces)
    pdata = pv.PolyData(vertices, pv_faces)
    pdata.cell_data["RGB"] = face_labels_to_rgb(face_labels)

    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    plotter.set_background("white")
    plotter.add_mesh(
        pdata,
        scalars="RGB",
        rgb=True,
        show_scalar_bar=False,
        smooth_shading=True,
    )
    _apply_camera(plotter, vertices, elevation_deg, azimuth_deg)
    plotter.enable_anti_aliasing("ssaa")
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def render_segmentation_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    output_path: Path,
    *,
    title: str = "SEG-MAT 分割",
    image_size: Tuple[int, int] = (640, 640),
    cols: int = VIEW_COLS,
) -> Path:
    """12 视角（正二十面体采样）拼接 SEG-MAT 分块着色图。"""
    labels = [f"elev={el:.0f} az={az:.0f}" for el, az in VIEW_GRID]
    views: List[np.ndarray] = []
    for elev, azim in VIEW_GRID:
        views.append(
            render_segmentation_view(
                vertices,
                faces,
                face_labels,
                elev,
                azim,
                image_size=image_size,
            )
        )
    stitched = stitch_views(views, cols=cols, labels=labels)

    if Image is not None:
        grid = Image.fromarray(stitched)
        title_h = 40
        out = Image.new("RGB", (grid.width, grid.height + title_h), (255, 255, 255))
        draw = ImageDraw.Draw(out)
        draw.rectangle((0, 0, out.width, title_h), fill=(245, 245, 245))
        draw.text((8, 10), title, fill=(20, 20, 20))
        out.paste(grid, (0, title_h))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(output_path)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path.with_suffix(".npy"), stitched)

    return output_path


def render_medoid_segmentation_12view(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    output_dir: str | Path,
    *,
    render_anim_name: str = "",
    render_frame_index: int = -1,
    render_time: float = 0.0,
    render_medoid_score: float = 0.0,
    num_segments: int = 0,
    image_size: int = 640,
    cols: int = VIEW_COLS,
    filename: str = "medoid_segmentation.png",
) -> Path:
    """在 medoid 帧上渲染 SEG-MAT 分块 12 视角拼图，写入 output/<模型>/segmentation/。"""
    out_dir = Path(output_dir)
    title = (
        f"[SEG-MAT 分割] {render_anim_name or 'medoid'}  "
        f"frame={render_frame_index}  t={render_time:.3f}s  "
        f"medoid={render_medoid_score:.4f}  parts={num_segments}"
    )
    return render_segmentation_panel(
        vertices,
        faces,
        face_labels,
        out_dir / filename,
        title=title,
        image_size=(image_size, image_size),
        cols=cols,
    )


def _compose_labeled_row(
    panels: Sequence[Tuple[str, np.ndarray]],
    border: int = 8,
    title_height: int = 36,
) -> np.ndarray:
    if Image is None:
        raise ImportError("需要 Pillow: pip install pillow") from _PIL_IMPORT_ERROR

    h, w = panels[0][1].shape[:2]
    canvas_w = border + len(panels) * (w + border)
    canvas_h = border + title_height + h + border
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = border
    y = border
    for title, img in panels:
        draw.rectangle((x, y, x + w, y + title_height), fill=(245, 245, 245))
        draw.text((x + 8, y + 10), title, fill=(20, 20, 20))
        canvas.paste(Image.fromarray(img), (x, y + title_height))
        x += w + border

    return np.asarray(canvas)


def render_medoid_pose_overview(
    vertices: np.ndarray,
    faces: np.ndarray,
    angular: np.ndarray,
    area: np.ndarray,
    output_dir: str | Path,
    *,
    face_labels: Optional[Sequence[int]] = None,
    elevation_deg: float = DEFAULT_OVERVIEW_VIEW[0],
    azimuth_deg: float = DEFAULT_OVERVIEW_VIEW[1],
    image_size: int = 640,
    cmap: str = DEFAULT_CMAP,
    render_anim_name: str = "",
    render_frame_index: int = -1,
    render_time: float = 0.0,
    render_medoid_score: float = 0.0,
    num_segments: int = 0,
) -> MedoidPoseRenderResult:
    """
    在 medoid 输出姿态上，用同一相机视角渲染角度/面积畸变热力图与 SEG-MAT 分割，并拼成一张图。
    所有产物写入 output_dir（建议 output/<模型>/medoid_pose/）。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (image_size, image_size)

    ang_n = angular / max(float(np.max(angular)), 1e-12)
    area_n = area / max(float(np.max(area)), 1e-12)

    ang_img = render_single_view(
        vertices, faces, ang_n, elevation_deg, azimuth_deg, image_size=size, cmap=cmap
    )
    area_img = render_single_view(
        vertices, faces, area_n, elevation_deg, azimuth_deg, image_size=size, cmap=cmap
    )

    ang_path = out_dir / "angular_distortion.png"
    area_path = out_dir / "area_distortion.png"
    Image.fromarray(ang_img).save(ang_path)
    Image.fromarray(area_img).save(area_path)

    panels: List[Tuple[str, np.ndarray]] = [
        ("角度畸变", ang_img),
        ("面积畸变", area_img),
    ]
    seg_path: Optional[Path] = None
    if face_labels is not None and len(face_labels) == len(faces):
        seg_img = render_segmentation_view(
            vertices, faces, face_labels, elevation_deg, azimuth_deg, image_size=size
        )
        seg_path = out_dir / "segmentation.png"
        Image.fromarray(seg_img).save(seg_path)
        panels.append(("SEG-MAT 分割", seg_img))

    composite_arr = _compose_labeled_row(panels)
    composite_path = out_dir / "composite.png"
    Image.fromarray(composite_arr).save(composite_path)

    meta = {
        "camera": {"elevation_deg": elevation_deg, "azimuth_deg": azimuth_deg},
        "render_frame": {
            "anim": render_anim_name,
            "frame_index": render_frame_index,
            "time": render_time,
            "medoid_score": render_medoid_score,
        },
        "num_segments": num_segments,
        "files": {
            "composite": composite_path.name,
            "angular": ang_path.name,
            "area": area_path.name,
            "segmentation": seg_path.name if seg_path else None,
        },
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return MedoidPoseRenderResult(
        output_dir=out_dir,
        composite=composite_path,
        angular=ang_path,
        area=area_path,
        segmentation=seg_path,
        meta_json=meta_path,
    )


def stitch_views(
    images: Sequence[np.ndarray],
    cols: int = 4,
    labels: Optional[Sequence[str]] = None,
    border: int = 4,
    label_height: int = 28,
) -> np.ndarray:
    if Image is None:
        raise ImportError("需要 Pillow: pip install pillow") from _PIL_IMPORT_ERROR

    n = len(images)
    rows = int(np.ceil(n / cols))
    h, w = images[0].shape[:2]
    canvas_h = rows * (h + label_height + border) + border
    canvas_w = cols * (w + border) + border
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = border + c * (w + border)
        y = border + r * (h + label_height + border)
        pil = Image.fromarray(img)
        canvas.paste(pil, (x, y + label_height))
        if labels and i < len(labels):
            draw.text((x + 4, y + 4), labels[i], fill=(40, 40, 40))

    return np.asarray(canvas)


def render_animation_panel(
    pack: AnimationRenderPack,
    faces: np.ndarray,
    output_path: Path,
    scalars: np.ndarray,
    metric_label: str,
    image_size: Tuple[int, int] = (640, 640),
    cols: int = 4,
    cmap: str = DEFAULT_CMAP,
) -> Path:
    labels = [f"elev={el:.0f} az={az:.0f}" for el, az in VIEW_GRID]
    views: List[np.ndarray] = []
    for elev, azim in VIEW_GRID:
        views.append(
            render_single_view(
                pack.vertices,
                faces,
                scalars,
                elev,
                azim,
                image_size=image_size,
                cmap=cmap,
            )
        )
    stitched = stitch_views(views, cols=cols, labels=labels)
    title = (
        f"[{metric_label}] {pack.name}  frame={pack.frame_index}  "
        f"t={pack.time:.3f}s  medoid={pack.medoid_score:.4f}"
    )

    if Image is not None:
        grid = Image.fromarray(stitched)
        title_h = 40
        out = Image.new("RGB", (grid.width, grid.height + title_h), (255, 255, 255))
        draw = ImageDraw.Draw(out)
        draw.rectangle((0, 0, out.width, title_h), fill=(245, 245, 245))
        draw.text((8, 10), title, fill=(20, 20, 20))
        out.paste(grid, (0, title_h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(output_path)
    else:
        np.save(output_path.with_suffix(".npy"), stitched)

    return output_path


def render_all_animations(
    mesh_path: str | Path,
    output_dir: str | Path,
    reference: str = "weighted_average",
    uniform_anim_weight: bool = False,
    fps: float = 30.0,
    max_frames: int | None = None,
    image_size: int = 640,
    cols: int = 4,
    cmap: str = DEFAULT_CMAP,
    medoid_max_samples: Optional[int] = 128,
    metrics: Optional[Sequence[str]] = None,
) -> List[Path]:
    mesh = load_mesh(mesh_path, samples_per_second=fps, max_frames=max_frames)
    return render_all_animations_from_mesh(
        mesh,
        output_dir,
        reference=reference,
        uniform_anim_weight=uniform_anim_weight,
        image_size=image_size,
        cols=cols,
        cmap=cmap,
        medoid_max_samples=medoid_max_samples,
        metrics=metrics,
    )


def render_all_animations_from_mesh(
    mesh: MeshData,
    output_dir: str | Path,
    reference: str = "weighted_average",
    uniform_anim_weight: bool = False,
    image_size: int = 640,
    cols: int = 4,
    cmap: str = DEFAULT_CMAP,
    medoid_max_samples: Optional[int] = 128,
    metrics: Optional[Sequence[str]] = None,
) -> List[Path]:
    packs, faces = prepare_animation_packs(
        mesh, reference, uniform_anim_weight, medoid_max_samples=medoid_max_samples
    )

    out_dir = Path(output_dir)
    saved: List[Path] = []
    metric_specs = [
        (key, folder, label)
        for key, folder, label in RENDER_METRICS
        if metrics is None or key in metrics
    ]

    for key, folder, label in metric_specs:
        metric_dir = out_dir / folder
        metric_dir.mkdir(parents=True, exist_ok=True)
        for pack in packs:
            fname = _safe_filename(pack.name) + ".png"
            path = metric_dir / fname
            print(f"[render] {label} / {pack.name} -> {path}")
            render_animation_panel(
                pack,
                faces,
                path,
                scalars=pack.scalars_for(key),
                metric_label=label,
                image_size=(image_size, image_size),
                cols=cols,
                cmap=cmap,
            )
            saved.append(path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线渲染每个动作的热力图（12 视角自动拼接）"
    )
    parser.add_argument("mesh", type=str, help="Mesh 文件路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="热力图根目录（其下按指标分子文件夹，默认 output/<模型名>/heatmaps/）",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="output",
        help="输出根目录（默认 output）",
    )
    parser.add_argument(
        "--reference",
        choices=("weighted_average", "rest"),
        default="weighted_average",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--uniform-weight", action="store_true")
    parser.add_argument("--image-size", type=int, default=640, help="单视角分辨率")
    parser.add_argument("--cols", type=int, default=4, help="拼接列数（默认 4×3=12 视角）")
    parser.add_argument("--cmap", type=str, default=DEFAULT_CMAP)
    parser.add_argument(
        "--metrics",
        type=str,
        default="angular,area,weighted,combined",
        help="要渲染的指标，逗号分隔（angular,area,weighted,combined）",
    )
    parser.add_argument("--medoid-samples", type=int, default=128)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else session_output_dir(args.mesh, args.output_base) / "heatmaps"
    )

    t0 = time.perf_counter()
    paths = render_all_animations(
        args.mesh,
        out_dir,
        reference=args.reference,
        uniform_anim_weight=args.uniform_weight,
        fps=args.fps,
        max_frames=args.max_frames,
        image_size=args.image_size,
        cols=args.cols,
        cmap=args.cmap,
        medoid_max_samples=args.medoid_samples,
        metrics=metric_list,
    )
    if args.verbose:
        print(f"[done] 共 {len(paths)} 张，耗时 {time.perf_counter() - t0:.1f}s")
        by_dir: dict = {}
        for p in paths:
            by_dir.setdefault(p.parent.name, []).append(p)
        for name, group in by_dir.items():
            print(f"  {name}/ ({len(group)} 张)")


if __name__ == "__main__":
    main()
