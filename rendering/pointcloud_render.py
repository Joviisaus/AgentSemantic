"""
点云渲染（pyvista 点云模式）：把 mesh 版 SEG-MAT/分块渲染搬到「裸点云」上。

复用 heatmap_renderer 的相机（VIEW_GRID 正二十面体 12 视角、_apply_camera）、调色板
（SEG_MAT_COLORS）与拼图（stitch_views），只是把 ``add_mesh(三角面)`` 换成
``add_points(render_points_as_spheres=True)``，按 per-point RGB 着色。

两种模式：
- 分块着色：每个点按其 cluster label 上色（给人看 / live_report）。
- 高亮：当前 block 的点鲜红、其它点淡灰（给 VLM 三分类聚焦用），对齐 mesh 版
  render_component 的 HIGHLIGHT/DIM 语义。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .heatmap_renderer import (
    SEG_MAT_COLORS,
    VIEW_COLS,
    VIEW_GRID,
    _apply_camera,
    pv,
    _PV_IMPORT_ERROR,
    stitch_views,
)

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    raise ImportError("渲染需要 Pillow: pip install pillow") from exc


HIGHLIGHT_RGB: Tuple[int, int, int] = (230, 60, 40)
DIM_RGB: Tuple[int, int, int] = (205, 205, 212)
_BG = "white"

# 与 mesh 版 pipeline._LIVE_COLOR_MAP 对齐：full=深红 / pending(partial 待分裂)=浅黄
# / none=白 / abandoned=极浅灰。用于 live_report 与迭代时间线的逐点状态着色。
STATUS_COLOR_MAP: dict = {
    "full": (140, 25, 35),
    "pending": (255, 215, 130),
    "none": (255, 255, 255),
    "abandoned": (235, 235, 235),
}


def point_labels_to_rgb(point_labels: Sequence[int]) -> np.ndarray:
    """点标签 -> (N, 3) uint8 RGB（label<0 视为背景灰）。"""
    labels = np.asarray(point_labels, dtype=np.int64)
    out = np.zeros((len(labels), 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        if label < 0:
            out[i] = (200, 200, 200)
        else:
            out[i] = SEG_MAT_COLORS[int(label) % len(SEG_MAT_COLORS)]
    return out


def build_highlight_point_colors(
    n_points: int,
    highlight_ids: Iterable[int],
    *,
    highlight_rgb: Tuple[int, int, int] = HIGHLIGHT_RGB,
    dim_rgb: Tuple[int, int, int] = DIM_RGB,
) -> np.ndarray:
    """高亮 point_ids 鲜红、其余淡灰，返回 (N, 3) uint8。"""
    colors = np.tile(np.asarray(dim_rgb, dtype=np.uint8), (n_points, 1))
    ids = np.asarray(list(highlight_ids), dtype=np.int64)
    if ids.size:
        mask = (ids >= 0) & (ids < n_points)
        colors[ids[mask]] = np.asarray(highlight_rgb, dtype=np.uint8)
    return colors


def _auto_point_size(n_points: int, image_size: Tuple[int, int]) -> float:
    """点数越多点画得越小，保证既不糊成一片也不过于稀疏。"""
    base = min(image_size) / 64.0
    scale = (20000.0 / max(n_points, 1)) ** 0.25
    return float(np.clip(base * scale, 2.0, 14.0))


def render_pointcloud_rgb_view(
    points: np.ndarray,
    point_rgb: np.ndarray,
    elevation_deg: float,
    azimuth_deg: float,
    *,
    image_size: Tuple[int, int] = (640, 640),
    point_size: Optional[float] = None,
) -> np.ndarray:
    """单视角渲染：按 per-point RGB 着色的点云。返回 HxWx3 ndarray。"""
    if pv is None:
        raise ImportError("需要 pyvista: pip install pyvista") from _PV_IMPORT_ERROR

    pts = np.asarray(points, dtype=np.float64)[:, :3]
    pdata = pv.PolyData(pts)
    pdata.point_data["RGB"] = np.asarray(point_rgb, dtype=np.uint8)

    ps = point_size if point_size is not None else _auto_point_size(len(pts), image_size)
    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    plotter.set_background(_BG)
    plotter.add_points(
        pdata,
        scalars="RGB",
        rgb=True,
        render_points_as_spheres=True,
        point_size=ps,
        show_scalar_bar=False,
    )
    _apply_camera(plotter, pts, elevation_deg, azimuth_deg)
    plotter.enable_anti_aliasing("ssaa")
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def point_status_to_rgb(
    point_status: Sequence[str],
    *,
    color_map: Optional[dict] = None,
) -> np.ndarray:
    """逐点状态字符串 -> (N, 3) uint8 RGB。未知状态按 none 处理。"""
    cmap = color_map or STATUS_COLOR_MAP
    none_rgb = cmap.get("none", (255, 255, 255))
    out = np.zeros((len(point_status), 3), dtype=np.uint8)
    for i, st in enumerate(point_status):
        out[i] = cmap.get(st, none_rgb)
    return out


def render_pointcloud_status_view(
    points: np.ndarray,
    point_status: Sequence[str],
    output_path: Path,
    *,
    elev: float,
    azim: float,
    image_size: int = 720,
    color_map: Optional[dict] = None,
    background: str = "transparent",
    point_size: Optional[float] = None,
) -> Path:
    """单视角渲染逐点状态图（live_report / 迭代时间线用）。

    - point_status[i] ∈ {"full","pending","none","abandoned"}
    - background="transparent" 输出 RGBA PNG，背景透明（与 mesh 版一致）。
    """
    if pv is None:
        raise ImportError("需要 pyvista: pip install pyvista") from _PV_IMPORT_ERROR

    pts = np.asarray(points, dtype=np.float64)[:, :3]
    rgb = point_status_to_rgb(point_status, color_map=color_map)
    pdata = pv.PolyData(pts)
    pdata.point_data["RGB"] = rgb

    win = (int(image_size), int(image_size))
    ps = point_size if point_size is not None else _auto_point_size(len(pts), win)
    transparent = background == "transparent"
    plotter = pv.Plotter(off_screen=True, window_size=win)
    if transparent:
        plotter.set_background("white")
    else:
        plotter.set_background(background or "white")
    plotter.add_points(
        pdata,
        scalars="RGB",
        rgb=True,
        render_points_as_spheres=True,
        point_size=ps,
        show_scalar_bar=False,
    )
    _apply_camera(plotter, pts, elev, azim)
    plotter.enable_anti_aliasing("ssaa")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = plotter.screenshot(
        str(output_path),
        return_img=True,
        transparent_background=transparent,
    )
    plotter.close()
    return output_path


def _panel(
    points: np.ndarray,
    point_rgb: np.ndarray,
    output_path: Path,
    *,
    title: str,
    image_size: Tuple[int, int],
    cols: int,
    point_size: Optional[float],
) -> Path:
    labels = [f"elev={el:.0f} az={az:.0f}" for el, az in VIEW_GRID]
    views = [
        render_pointcloud_rgb_view(
            points, point_rgb, elev, azim, image_size=image_size, point_size=point_size
        )
        for elev, azim in VIEW_GRID
    ]
    stitched = stitch_views(views, cols=cols, labels=labels)

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
    return output_path


def render_pointcloud_segmentation_panel(
    points: np.ndarray,
    point_labels: Sequence[int],
    output_path: Path,
    *,
    title: str = "点云分块",
    image_size: Tuple[int, int] = (640, 640),
    cols: int = VIEW_COLS,
    point_size: Optional[float] = None,
) -> Path:
    """12 视角点云分块着色拼图（每个 cluster 一个颜色）。"""
    rgb = point_labels_to_rgb(point_labels)
    return _panel(
        points, rgb, output_path,
        title=title, image_size=image_size, cols=cols, point_size=point_size,
    )


def render_pointcloud_overview(
    points: np.ndarray,
    output_path: Path,
    *,
    elev: float,
    azim: float,
    image_size: int = 720,
    rgb: Tuple[int, int, int] = (205, 210, 220),
    background: str = "transparent",
    point_size: Optional[float] = None,
) -> Path:
    """单视角整朵点云概览图（统一浅灰、无状态着色），用作报告镇页图。"""
    if pv is None:
        raise ImportError("需要 pyvista: pip install pyvista") from _PV_IMPORT_ERROR

    pts = np.asarray(points, dtype=np.float64)[:, :3]
    colors = np.tile(np.asarray(rgb, dtype=np.uint8), (len(pts), 1))
    pdata = pv.PolyData(pts)
    pdata.point_data["RGB"] = colors

    win = (int(image_size), int(image_size))
    ps = point_size if point_size is not None else _auto_point_size(len(pts), win)
    transparent = background == "transparent"
    plotter = pv.Plotter(off_screen=True, window_size=win)
    plotter.set_background("white" if transparent else (background or "white"))
    plotter.add_points(
        pdata,
        scalars="RGB",
        rgb=True,
        render_points_as_spheres=True,
        point_size=ps,
        show_scalar_bar=False,
    )
    _apply_camera(plotter, pts, elev, azim)
    plotter.enable_anti_aliasing("ssaa")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path), return_img=True, transparent_background=transparent)
    plotter.close()
    return output_path


def _annotate_thumbnail(
    img: np.ndarray, idx_1based: int, elev: float, azim: float
) -> np.ndarray:
    """缩略图左上角画白底黑字编号 + 视角参数（与 mesh view_thumbnails 一致）。"""
    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)
    badge_w, badge_h = 78, 38
    draw.rectangle((6, 6, 6 + badge_w, 6 + badge_h), fill=(255, 255, 255),
                   outline=(20, 20, 20), width=2)
    draw.text((14, 10), f"#{idx_1based}", fill=(20, 20, 20))
    draw.text((14, 24), f"e={elev:.0f} a={azim:.0f}", fill=(60, 60, 60))
    return np.array(pil)


def render_pointcloud_thumbnails_panel(
    points: np.ndarray,
    output_path: Path,
    *,
    thumb_size: int = 320,
    cols: int = 4,
    title: str = "候选视角（VLM 挑选用）",
    rgb: Tuple[int, int, int] = (205, 210, 220),
    point_size: Optional[float] = None,
) -> Path:
    """12 视角点云缩略图拼图，每张左上角带编号 1..12（供 VLM 选最佳展示视角）。

    无任何分块/状态颜色，纯浅灰点，避免 VLM 选视角时被局部颜色干扰。
    编号顺序严格对齐 ``VIEW_GRID``，与 ``vlm.view_selector.select_best_view`` 的
    ``best_index → VIEW_GRID[idx]`` 映射保持一致。
    """
    if pv is None:
        raise ImportError("需要 pyvista: pip install pyvista") from _PV_IMPORT_ERROR

    pts = np.asarray(points, dtype=np.float64)[:, :3]
    colors = np.tile(np.asarray(rgb, dtype=np.uint8), (len(pts), 1))
    win = (int(thumb_size), int(thumb_size))
    ps = point_size if point_size is not None else _auto_point_size(len(pts), win)

    annotated: List[np.ndarray] = []
    for i, (elev, azim) in enumerate(VIEW_GRID):
        view = render_pointcloud_rgb_view(
            pts, colors, elev, azim, image_size=win, point_size=ps
        )
        annotated.append(_annotate_thumbnail(view, i + 1, elev, azim))

    stitched = stitch_views(
        annotated, cols=cols, labels=[f"#{i+1}" for i in range(len(annotated))]
    )
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
    return output_path


def render_pointcloud_highlight_panel(
    points: np.ndarray,
    highlight_ids: Iterable[int],
    output_path: Path,
    *,
    title: str = "block 高亮",
    image_size: Tuple[int, int] = (640, 640),
    cols: int = VIEW_COLS,
    point_size: Optional[float] = None,
) -> Path:
    """12 视角点云高亮拼图：当前 block 鲜红、其余淡灰（喂 VLM 三分类）。"""
    rgb = build_highlight_point_colors(len(np.asarray(points)), highlight_ids)
    return _panel(
        points, rgb, output_path,
        title=title, image_size=image_size, cols=cols, point_size=point_size,
    )
