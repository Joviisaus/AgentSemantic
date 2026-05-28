"""
单分块高亮的 12 视角渲染：复用 heatmap_renderer 的相机/拼图逻辑，
但给指定 face_ids 涂亮色、其余涂淡色，便于 VLM 聚焦到"这一块"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from heatmap_renderer import (
    VIEW_GRID,
    render_segmentation_panel,
    stitch_views,
)

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    raise ImportError("渲染需要 Pillow: pip install pillow") from exc


HIGHLIGHT_RGB: Tuple[int, int, int] = (230, 60, 40)      # 鲜艳红
HIGHLIGHT_RGB_SECONDARY: Tuple[int, int, int] = (242, 142, 43)  # 橙（备选）
DIM_RGB: Tuple[int, int, int] = (210, 210, 215)          # 淡灰
ACCEPTED_RGB: Tuple[int, int, int] = (46, 158, 94)       # 已判定"完全符合"
REJECTED_RGB: Tuple[int, int, int] = (220, 220, 225)     # 已判定"完全不符合"


def _faces_count(faces: np.ndarray) -> int:
    return int(faces.shape[0])


def build_highlight_face_colors(
    n_faces: int,
    highlight_face_ids: Iterable[int],
    *,
    highlight_rgb: Tuple[int, int, int] = HIGHLIGHT_RGB,
    dim_rgb: Tuple[int, int, int] = DIM_RGB,
) -> np.ndarray:
    """
    根据高亮 face_ids 生成 (F, 3) uint8 颜色数组，可直接喂给
    render_segmentation_view 的 face_labels_to_rgb 的等价分支。
    实际上这里我们走自定义路径：通过临时 face_labels 来复用 render_segmentation_view。
    """
    colors = np.tile(np.asarray(dim_rgb, dtype=np.uint8), (n_faces, 1))
    ids = np.asarray(list(highlight_face_ids), dtype=np.int64)
    if ids.size:
        mask = (ids >= 0) & (ids < n_faces)
        colors[ids[mask]] = np.asarray(highlight_rgb, dtype=np.uint8)
    return colors


def _render_views_with_color_table(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    *,
    image_size: Tuple[int, int],
) -> List[np.ndarray]:
    """
    借用 render_segmentation_view：它通过 face_labels_to_rgb 上色，
    但是只支持调色板索引。这里改写：用 SEG-MAT 不感知的"虚拟 label"，
    具体上色我们在 vlm/render_component 内部直接用 pyvista。
    """
    # 用更直接的方式：调 pyvista 一次渲染 12 视角。
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 pyvista: pip install pyvista") from exc

    # 复用 heatmap_renderer 中的相机辅助
    from heatmap_renderer import _apply_camera, _to_pyvista_faces

    pv_faces = _to_pyvista_faces(faces)
    pdata = pv.PolyData(vertices, pv_faces)
    pdata.cell_data["RGB"] = face_colors

    views: List[np.ndarray] = []
    for elev, azim in VIEW_GRID:
        plotter = pv.Plotter(off_screen=True, window_size=image_size)
        plotter.set_background("white")
        plotter.add_mesh(
            pdata,
            scalars="RGB",
            rgb=True,
            show_scalar_bar=False,
            smooth_shading=True,
        )
        _apply_camera(plotter, vertices, elev, azim)
        plotter.enable_anti_aliasing("ssaa")
        views.append(plotter.screenshot(return_img=True))
        plotter.close()
    return views


def _compose_panel(
    views: Sequence[np.ndarray],
    *,
    output_path: Path,
    title: str,
    cols: int = 4,
) -> Path:
    labels = [f"elev={el:.0f} az={az:.0f}" for el, az in VIEW_GRID]
    stitched = stitch_views(list(views), cols=cols, labels=labels)
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


def render_highlighted_component_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    highlight_face_ids: Sequence[int],
    output_path: str | Path,
    *,
    title: str = "VLM 分块",
    image_size: int = 512,
    cols: int = 4,
    highlight_rgb: Tuple[int, int, int] = HIGHLIGHT_RGB,
    dim_rgb: Tuple[int, int, int] = DIM_RGB,
) -> Path:
    """
    渲染 12 视角拼图：highlight_face_ids 用高亮色，其它面用淡色。
    输出 PNG 路径，供 VLM 直接读图。
    """
    n_faces = _faces_count(faces)
    face_colors = build_highlight_face_colors(
        n_faces,
        highlight_face_ids,
        highlight_rgb=highlight_rgb,
        dim_rgb=dim_rgb,
    )
    views = _render_views_with_color_table(
        vertices, faces, face_colors, image_size=(image_size, image_size)
    )
    return _compose_panel(views, output_path=Path(output_path), title=title, cols=cols)


def render_semantic_mask_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    semantic_face_ids: Sequence[int],
    output_path: str | Path,
    *,
    title: str = "VLM 语义掩码",
    image_size: int = 512,
    cols: int = 4,
    semantic_rgb: Tuple[int, int, int] = ACCEPTED_RGB,
    rest_rgb: Tuple[int, int, int] = REJECTED_RGB,
) -> Path:
    """渲染最终语义结果（绿色=属于语义，淡色=不属于）。"""
    return render_highlighted_component_panel(
        vertices,
        faces,
        semantic_face_ids,
        output_path,
        title=title,
        image_size=image_size,
        cols=cols,
        highlight_rgb=semantic_rgb,
        dim_rgb=rest_rgb,
    )


def render_initial_segmentation_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Sequence[int],
    output_path: str | Path,
    *,
    title: str = "初始分块",
    image_size: int = 512,
    cols: int = 4,
) -> Path:
    """便捷复用：直接调 heatmap_renderer 的 SEG-MAT 风格调色。"""
    return render_segmentation_panel(
        vertices,
        faces,
        list(face_labels),
        Path(output_path),
        title=title,
        image_size=(image_size, image_size),
        cols=cols,
    )


__all__ = [
    "ACCEPTED_RGB",
    "DIM_RGB",
    "HIGHLIGHT_RGB",
    "HIGHLIGHT_RGB_SECONDARY",
    "REJECTED_RGB",
    "build_highlight_face_colors",
    "render_highlighted_component_panel",
    "render_initial_segmentation_panel",
    "render_semantic_mask_panel",
]
