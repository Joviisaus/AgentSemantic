"""
给 VLM 选视角用的 12 张缩略图拼图。

每张缩略图：纯白底 + 浅灰几何 + 左上角编号 1..12（白底黑字带视角参数），
没有任何颜色信息，避免 VLM 在选视角时被局部颜色干扰。

供 `vlm.view_selector.select_best_view` 调用：
    panel_path = render_thumbnails_panel(vertices, faces, "out/thumbs.png")
    selection = select_best_view(client, panel_path, semantic="眼镜")
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .heatmap_renderer import (
    VIEW_GRID,
    _apply_camera,
    _to_pyvista_faces,
    stitch_views,
)


def _render_thumbnail_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    thumb_size: int = 320,
) -> List[np.ndarray]:
    """对 VIEW_GRID 里 12 个视角分别渲染一张小图。"""
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("视图选择需要 pyvista") from exc

    pv_faces = _to_pyvista_faces(faces)
    pdata = pv.PolyData(vertices, pv_faces)
    face_colors = np.tile(np.asarray((205, 210, 220), dtype=np.uint8),
                          (faces.shape[0], 1))
    pdata.cell_data["RGB"] = face_colors

    views: List[np.ndarray] = []
    for elev, azim in VIEW_GRID:
        plotter = pv.Plotter(off_screen=True, window_size=(thumb_size, thumb_size))
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


def _annotate_thumbnail(img: np.ndarray, idx_1based: int,
                        elev: float, azim: float) -> np.ndarray:
    """在缩略图左上角画一个白底黑字编号 + 视角参数。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 Pillow") from exc

    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)
    badge_w, badge_h = 78, 38
    draw.rectangle((6, 6, 6 + badge_w, 6 + badge_h), fill=(255, 255, 255),
                   outline=(20, 20, 20), width=2)
    draw.text((14, 10), f"#{idx_1based}", fill=(20, 20, 20))
    draw.text((14, 24), f"e={elev:.0f} a={azim:.0f}", fill=(60, 60, 60))
    return np.array(pil)


def render_thumbnails_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str | Path,
    *,
    thumb_size: int = 320,
    cols: int = 4,
    title: str = "候选视角（VLM 挑选用）",
) -> Path:
    """渲染 12 视角小图拼图，每张图左上角带编号 1..12。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 Pillow") from exc

    raw_views = _render_thumbnail_views(vertices, faces, thumb_size=thumb_size)
    annotated: List[np.ndarray] = []
    for i, view in enumerate(raw_views):
        elev, azim = VIEW_GRID[i]
        annotated.append(_annotate_thumbnail(view, i + 1, elev, azim))

    stitched = stitch_views(annotated, cols=cols,
                            labels=[f"#{i+1}" for i in range(len(annotated))])
    grid = Image.fromarray(stitched)
    title_h = 40
    out = Image.new("RGB", (grid.width, grid.height + title_h),
                    (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, title_h), fill=(245, 245, 245))
    draw.text((8, 10), title, fill=(20, 20, 20))
    out.paste(grid, (0, title_h))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    return output_path


__all__ = ["render_thumbnails_panel"]
