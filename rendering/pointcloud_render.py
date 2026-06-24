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


def _auto_point_size(
    points_or_count: "np.ndarray | int",
    image_size: Tuple[int, int],
    elevation_deg: Optional[float] = None,
    azimuth_deg: Optional[float] = None,
) -> float:
    """根据点数、画布和当前视角下的屏幕密度自适应点大小。

    旧逻辑只看点数，细长/低采样点云在某些视角会出现很多白缝。这里在有视角时先把点云
    投影到屏幕平面，估算像素级最近邻距离，再取较大的点大小来尽量补缝；失败时回退到
    基于点数的保守估算。
    """
    if isinstance(points_or_count, np.ndarray):
        pts = np.asarray(points_or_count, dtype=np.float64)[:, :3]
        n_points = len(pts)
    else:
        pts = None
        n_points = int(points_or_count)

    win_min = float(max(1, min(image_size)))
    # 基于点数的下限：点越少越大，点越多越小。
    count_ps = (win_min / 64.0) * (20000.0 / max(n_points, 1)) ** 0.25
    count_ps = float(np.clip(count_ps, 2.0, 16.0))

    if (
        pts is None
        or n_points < 2
        or elevation_deg is None
        or azimuth_deg is None
        or not np.isfinite(pts).all()
    ):
        return count_ps

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return count_ps

    center = pts.mean(axis=0)
    rel = pts - center
    view_dir = _view_direction(float(elevation_deg), float(azimuth_deg))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(world_up, view_dir))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, view_dir)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-12:
        return count_ps
    right = right / right_norm
    up = np.cross(view_dir, right)
    up_norm = np.linalg.norm(up)
    if up_norm <= 1e-12:
        return count_ps
    up = up / up_norm

    xy = np.column_stack([rel @ right, rel @ up])
    span = np.ptp(xy, axis=0)
    model_span = float(max(span.max(), 1e-12))
    # _apply_camera 里有 zoom(0.9) 和 margin；这里保守映射到约 82% 画布，匹配实际入画留白。
    px = (xy / model_span) * (win_min * 0.82)

    # 大点云抽样估算，避免报告渲染时最近邻统计过慢。
    max_samples = 6000
    if len(px) > max_samples:
        step = int(np.ceil(len(px) / max_samples))
        px_sample = px[::step]
    else:
        px_sample = px
    if len(px_sample) < 2:
        return count_ps

    d, _ = cKDTree(px_sample).query(px_sample, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 1e-6)]
    if nn.size == 0:
        return count_ps

    # median 反映主体密度；p75 让稀疏区域少露缝。乘 1.35 接近“圆点直径覆盖邻距”。
    density_ps = float(max(np.median(nn) * 1.35, np.percentile(nn, 75) * 1.05))
    # 允许稀疏点云比旧上限更大，但限制最大值避免 VLM 看不清边界。
    return float(np.clip(max(count_ps, density_ps), 2.0, 24.0))


# 法向着色用：环境光下限（最暗也保留这么多 base color），漫反射占 1-ambient。
_SHADE_AMBIENT = 0.45


def compute_point_normals(points: np.ndarray) -> np.ndarray:
    """整朵点云每点的 PCA 法向 (N, 3)，供逐点 shader 用（无依赖时返回全零）。"""
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    try:
        from geometry.pointcloud_cluster import compute_point_normals_and_curvature

        normals, _curv = compute_point_normals_and_curvature(pts)
        return np.asarray(normals, dtype=np.float64)
    except Exception:  # noqa: BLE001 — 法向算不出时退化为无 shader
        return np.zeros((len(pts), 3), dtype=np.float64)


def _view_direction(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """该视角下「点 → 相机」的单位方向（与 heatmap_renderer._camera_position 同系）。
    作为 headlight 光源方向，保证每个视角都从相机方向打光，shading 随视角一致变化。"""
    el = np.deg2rad(float(elevation_deg))
    az = np.deg2rad(float(azimuth_deg))
    v = np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=np.float64,
    )
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _shade_rgb_by_normal(
    base_rgb: np.ndarray,
    normals: np.ndarray,
    view_dir: np.ndarray,
    *,
    ambient: float = _SHADE_AMBIENT,
) -> np.ndarray:
    """用每点法向对 base color 做 Lambertian 明暗：rgb = base * (ambient + (1-ambient)·|n·l|)。

    点云 PCA 法向无定向（符号任意），故用 |n·l| 避免一半点恒黑；shade ∈ [ambient, 1]，
    保留 base color 的色相，仅调明暗以体现表面朝向（替代球体的立体感）。
    """
    base = np.asarray(base_rgb, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    if n.shape[0] != base.shape[0] or n.size == 0:
        return base.astype(np.uint8)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    unit = np.divide(n, norms, out=np.zeros_like(n), where=norms > 1e-12)
    ndl = np.abs(unit @ np.asarray(view_dir, dtype=np.float64))
    shade = ambient + (1.0 - ambient) * ndl
    # 法向缺失（全零）的点不打折，保持原色。
    shade = np.where(norms[:, 0] > 1e-12, shade, 1.0)
    out = base * shade[:, None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def render_pointcloud_rgb_view(
    points: np.ndarray,
    point_rgb: np.ndarray,
    elevation_deg: float,
    azimuth_deg: float,
    *,
    image_size: Tuple[int, int] = (640, 640),
    point_size: Optional[float] = None,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """单视角渲染：按 per-point RGB 着色的扁平点云。返回 HxWx3 ndarray。

    传入 ``normals`` 时，对每点 base color 按其法向做 headlight Lambertian 明暗
    （见 ``_shade_rgb_by_normal``），并以扁平点（``render_points_as_spheres=False``）
    绘制——不再画带光照的小球，而是用法向给整点上一个明暗后的纯色。
    """
    if pv is None:
        raise ImportError("需要 pyvista: pip install pyvista") from _PV_IMPORT_ERROR

    pts = np.asarray(points, dtype=np.float64)[:, :3]
    rgb = np.asarray(point_rgb, dtype=np.uint8)
    if normals is not None:
        rgb = _shade_rgb_by_normal(rgb, normals, _view_direction(elevation_deg, azimuth_deg))
    pdata = pv.PolyData(pts)
    pdata.point_data["RGB"] = rgb

    ps = point_size if point_size is not None else _auto_point_size(
        pts, image_size, elevation_deg, azimuth_deg
    )
    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    plotter.set_background(_BG)
    plotter.add_points(
        pdata,
        scalars="RGB",
        rgb=True,
        render_points_as_spheres=False,
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
    normals: Optional[np.ndarray] = None,
) -> Path:
    """单视角渲染逐点状态图（live_report / 迭代时间线用）。

    - point_status[i] ∈ {"full","pending","none","abandoned"}
    - background="transparent" 输出 RGBA PNG，背景透明（与 mesh 版一致）。
    - 传入 ``normals`` 时按法向对状态色做明暗（扁平点，不画球体）。
    """
    if pv is None:
        raise ImportError("需要 pyvista: pip install pyvista") from _PV_IMPORT_ERROR

    pts = np.asarray(points, dtype=np.float64)[:, :3]
    rgb = point_status_to_rgb(point_status, color_map=color_map)
    if normals is not None:
        rgb = _shade_rgb_by_normal(rgb, normals, _view_direction(elev, azim))
    pdata = pv.PolyData(pts)
    pdata.point_data["RGB"] = rgb

    win = (int(image_size), int(image_size))
    ps = point_size if point_size is not None else _auto_point_size(pts, win, elev, azim)
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
        render_points_as_spheres=False,
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
    views: Optional[Sequence[Tuple[float, float]]] = None,
    normals: Optional[np.ndarray] = None,
) -> Path:
    view_list = list(views) if views else list(VIEW_GRID)
    labels = [f"elev={el:.0f} az={az:.0f}" for el, az in view_list]
    rendered = [
        render_pointcloud_rgb_view(
            points, point_rgb, elev, azim,
            image_size=image_size, point_size=point_size, normals=normals,
        )
        for elev, azim in view_list
    ]
    # 列数不超过实际视角数，避免空格子（少视角时拼图更紧凑）。
    cols = max(1, min(int(cols), len(view_list)))
    stitched = stitch_views(rendered, cols=cols, labels=labels)

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
    normals: Optional[np.ndarray] = None,
) -> Path:
    """12 视角点云分块着色拼图（每个 cluster 一个颜色，按法向明暗的扁平点）。"""
    rgb = point_labels_to_rgb(point_labels)
    return _panel(
        points, rgb, output_path,
        title=title, image_size=image_size, cols=cols, point_size=point_size,
        normals=normals,
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
    normals: Optional[np.ndarray] = None,
) -> Path:
    """单视角整朵点云概览图（统一浅灰、无状态着色），用作报告镇页图。

    传入 ``normals`` 时按法向对浅灰底色做明暗的扁平点（不画球体）。
    """
    if pv is None:
        raise ImportError("需要 pyvista: pip install pyvista") from _PV_IMPORT_ERROR

    pts = np.asarray(points, dtype=np.float64)[:, :3]
    colors = np.tile(np.asarray(rgb, dtype=np.uint8), (len(pts), 1))
    if normals is not None:
        colors = _shade_rgb_by_normal(colors, normals, _view_direction(elev, azim))
    pdata = pv.PolyData(pts)
    pdata.point_data["RGB"] = colors

    win = (int(image_size), int(image_size))
    ps = point_size if point_size is not None else _auto_point_size(pts, win, elev, azim)
    transparent = background == "transparent"
    plotter = pv.Plotter(off_screen=True, window_size=win)
    plotter.set_background("white" if transparent else (background or "white"))
    plotter.add_points(
        pdata,
        scalars="RGB",
        rgb=True,
        render_points_as_spheres=False,
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
    normals: Optional[np.ndarray] = None,
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
    annotated: List[np.ndarray] = []
    for i, (elev, azim) in enumerate(VIEW_GRID):
        view = render_pointcloud_rgb_view(
            pts, colors, elev, azim, image_size=win, point_size=point_size, normals=normals
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
    views: Optional[Sequence[Tuple[float, float]]] = None,
    normals: Optional[np.ndarray] = None,
) -> Path:
    """多视角点云高亮拼图：当前 block 鲜红、其余淡灰（喂 VLM 三分类）。

    ``views`` 不传时用全部 ``VIEW_GRID``（12 视角）；传入 (elev, azim) 子集时只渲染
    这些视角——由 VLM 在选最佳视角阶段决定的「分类视角子集」，少而精以降低幻觉。
    ``normals`` 传入时，红/灰 base color 按每点法向做明暗的扁平点（不画球体）。
    """
    rgb = build_highlight_point_colors(len(np.asarray(points)), highlight_ids)
    return _panel(
        points, rgb, output_path,
        title=title, image_size=image_size, cols=cols, point_size=point_size,
        views=views, normals=normals,
    )


# 大连通域分色：每个分量一个高饱和的固定色，未选/小块用淡灰。颜色名进 prompt 图例。
COMPONENT_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (230, 60, 40),    # red
    (40, 110, 230),   # blue
    (40, 180, 70),    # green
    (240, 160, 30),   # orange
    (160, 70, 200),   # purple
    (30, 190, 200),   # cyan
)
COMPONENT_COLOR_NAMES: Tuple[str, ...] = ("red", "blue", "green", "orange", "purple", "cyan")


def build_component_point_colors(
    n_points: int,
    components: Sequence[Iterable[int]],
    *,
    dim_rgb: Tuple[int, int, int] = DIM_RGB,
) -> np.ndarray:
    """按分量给点上色：第 i 个分量取 COMPONENT_PALETTE[i]，其余点淡灰。返回 (N, 3) uint8。"""
    colors = np.tile(np.asarray(dim_rgb, dtype=np.uint8), (n_points, 1))
    for rank, ids in enumerate(components):
        rgb = COMPONENT_PALETTE[rank % len(COMPONENT_PALETTE)]
        arr = np.asarray(list(ids), dtype=np.int64)
        if arr.size:
            valid = arr[(arr >= 0) & (arr < n_points)]
            colors[valid] = np.asarray(rgb, dtype=np.uint8)
    return colors


def component_color_legend(n_components: int) -> str:
    """生成 "C0=red, C1=blue, ..." 图例字符串，喂给 VLM 对齐颜色与分量号。"""
    parts = [
        f"C{i}={COMPONENT_COLOR_NAMES[i % len(COMPONENT_COLOR_NAMES)]}"
        for i in range(max(0, int(n_components)))
    ]
    return ", ".join(parts)


def render_pointcloud_components_panel(
    points: np.ndarray,
    components: Sequence[Iterable[int]],
    output_path: Path,
    *,
    title: str = "connected components",
    image_size: Tuple[int, int] = (640, 640),
    cols: int = VIEW_COLS,
    point_size: Optional[float] = None,
    views: Optional[Sequence[Tuple[float, float]]] = None,
    normals: Optional[np.ndarray] = None,
) -> Path:
    """多视角拼图：把选区里若干（大）连通分量按固定调色板分色，其余点淡灰。

    供「选区分裂成两个都很大的连通域」时让 VLM 区分各块、判断该保留谁。
    ``components`` 按 rank 顺序传入（rank 0 = 最大块），颜色与 ``component_color_legend`` 对齐。
    """
    rgb = build_component_point_colors(len(np.asarray(points)), components)
    return _panel(
        points, rgb, output_path,
        title=title, image_size=image_size, cols=cols, point_size=point_size,
        views=views, normals=normals,
    )
