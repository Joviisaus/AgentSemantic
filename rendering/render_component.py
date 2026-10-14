"""
单分块高亮的 12 视角渲染：复用 heatmap_renderer 的相机/拼图逻辑，
但给指定 face_ids 涂亮色、其余涂淡色，便于 VLM 聚焦到"这一块"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .heatmap_renderer import (
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

# 全局进度面板调色（VLM 自评用）
PROGRESS_FULL_RGB: Tuple[int, int, int] = (220, 50, 47)        # 鲜红：已确认符合语义
PROGRESS_PENDING_RGB: Tuple[int, int, int] = (245, 200, 50)    # 亮黄：待分裂的 partial
PROGRESS_NONE_RGB: Tuple[int, int, int] = (220, 220, 225)      # 浅灰：已确认不符合
PROGRESS_ABANDONED_RGB: Tuple[int, int, int] = (170, 170, 175) # 中灰：放弃 / max-depth 截断


def _faces_count(faces: np.ndarray) -> int:
    return int(faces.shape[0])


def mesh_face_connected_components(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Per-face face-adjacency connected-component id (0..K-1)."""
    import trimesh
    from trimesh.graph import connected_component_labels

    n_faces = _faces_count(faces)
    mesh = trimesh.Trimesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        process=False,
    )
    adj = mesh.face_adjacency
    if adj is None or len(adj) == 0:
        return np.zeros(n_faces, dtype=np.int64)
    return connected_component_labels(adj, node_count=n_faces)


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
    view_list: Sequence[Tuple[float, float]] = VIEW_GRID,
) -> List[np.ndarray]:
    """
    多视角渲染。每个 face-adjacency 连通块单独 add_mesh，避免三角汤里
    多个非连通岛被 pyvista 合并后只显示最大岛。

    ``view_list`` 默认全部 12 视角；传入 (elev, azim) 子集时只渲染这些视角
    ——由 VLM 选最佳视角阶段决定的「分类视角子集」，少而精以降低幻觉。
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 pyvista: pip install pyvista") from exc

    from .heatmap_renderer import _apply_camera, _to_pyvista_faces

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_colors = np.asarray(face_colors, dtype=np.uint8)
    cc = mesh_face_connected_components(vertices, faces)

    views: List[np.ndarray] = []
    for elev, azim in view_list:
        plotter = pv.Plotter(off_screen=True, window_size=image_size)
        plotter.set_background("white")
        for cc_id in np.unique(cc):
            mask = cc == int(cc_id)
            f_sub = faces[mask]
            c_sub = face_colors[mask]
            used_v = np.unique(f_sub.reshape(-1))
            remap = -np.ones(len(vertices), dtype=np.int64)
            remap[used_v] = np.arange(len(used_v), dtype=np.int64)
            v_sub = vertices[used_v]
            f_remapped = remap[f_sub]
            pdata = pv.PolyData(v_sub, _to_pyvista_faces(f_remapped))
            pdata.cell_data["RGB"] = c_sub
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
    view_list: Sequence[Tuple[float, float]] = VIEW_GRID,
) -> Path:
    labels = [f"elev={el:.0f} az={az:.0f}" for el, az in view_list]
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
    views: Optional[Sequence[Tuple[float, float]]] = None,
) -> Path:
    """
    渲染多视角高亮拼图：highlight_face_ids 用高亮色，其它面用淡色。
    输出 PNG 路径，供 VLM 直接读图。

    ``views`` 不传时用全部 ``VIEW_GRID``（12 视角）；传入 (elev, azim) 子集时只渲染
    这些视角——由 VLM 在选最佳视角阶段决定的「分类视角子集」，少而精以降低幻觉。
    """
    n_faces = _faces_count(faces)
    face_colors = build_highlight_face_colors(
        n_faces,
        highlight_face_ids,
        highlight_rgb=highlight_rgb,
        dim_rgb=dim_rgb,
    )
    view_list = list(views) if views else list(VIEW_GRID)
    rendered = _render_views_with_color_table(
        vertices, faces, face_colors,
        image_size=(image_size, image_size),
        view_list=view_list,
    )
    return _compose_panel(
        rendered, output_path=Path(output_path), title=title, cols=cols,
        view_list=view_list,
    )


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


def render_single_view_status(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_status: Sequence[str],
    output_path: str | Path,
    *,
    elev: float,
    azim: float,
    image_size: int = 720,
    color_map: dict | None = None,
    background: str = "transparent",
    title: str | None = None,
) -> Path:
    """单视角渲染当前 VLM 进度图。

    - face_status[i] ∈ {"full", "pending", "none", "abandoned"}
        - "full"      → 深红高亮（用户指定）
        - "pending"   → 浅黄（partial 待分裂）
        - "none"      → 纯白（用户指定）
        - "abandoned" → 极浅灰
    - background="transparent" 时输出 RGBA PNG，背景全透明；none 面仍是白色几何。
    - title 非空时在图片底部画一个白色矩形 + 文字（无标题保持完全透明）。

    用于实时 HTML 报告里固定展示「这一张」图。
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 pyvista") from exc
    from .heatmap_renderer import _apply_camera, _to_pyvista_faces

    n_faces = _faces_count(faces)
    if len(face_status) != n_faces:
        raise ValueError(
            f"face_status 长度 {len(face_status)} ≠ faces.shape[0] {n_faces}"
        )

    cmap = {
        "full": (140, 25, 35),         # 深红高亮（默认）
        "pending": (255, 215, 130),    # 浅黄
        "none": (255, 255, 255),       # 纯白
        "abandoned": (235, 235, 235),  # 极浅灰
    }
    if color_map:
        cmap.update({k: tuple(v) for k, v in color_map.items()})

    face_colors = np.zeros((n_faces, 3), dtype=np.uint8)
    for i, st in enumerate(face_status):
        face_colors[i] = np.asarray(cmap.get(st, cmap["none"]), dtype=np.uint8)

    pv_faces = _to_pyvista_faces(faces)
    pdata = pv.PolyData(vertices, pv_faces)
    pdata.cell_data["RGB"] = face_colors

    transparent = background == "transparent"
    plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size))
    if transparent:
        # pyvista 透明截图需要 background_color + screenshot(transparent_background=True)
        plotter.set_background("white")
    else:
        plotter.set_background(str(background))
    plotter.add_mesh(
        pdata,
        scalars="RGB",
        rgb=True,
        show_scalar_bar=False,
        smooth_shading=True,
        lighting=True,
        ambient=0.45,
        diffuse=0.55,
    )
    _apply_camera(plotter, vertices, elev, azim)
    plotter.enable_anti_aliasing("ssaa")
    img = plotter.screenshot(return_img=True, transparent_background=transparent)
    plotter.close()

    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4 and transparent:
        pil = Image.fromarray(arr, mode="RGBA")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil = Image.fromarray(arr, mode="RGB")
        if transparent:
            pil = pil.convert("RGBA")
    else:
        pil = Image.fromarray(arr).convert("RGBA" if transparent else "RGB")

    if title:
        # 在底部贴一个半透明白条 + 标题，不破坏整体透明
        from PIL import ImageDraw

        bar_h = 36
        bar = Image.new(
            "RGBA" if transparent else "RGB",
            (pil.width, bar_h),
            (245, 245, 245, 235) if transparent else (245, 245, 245),
        )
        ImageDraw.Draw(bar).text((10, 10), title, fill=(20, 20, 20))
        combined = Image.new(
            "RGBA" if transparent else "RGB",
            (pil.width, pil.height + bar_h),
            (0, 0, 0, 0) if transparent else (255, 255, 255),
        )
        combined.paste(pil, (0, 0))
        combined.paste(bar, (0, pil.height))
        pil = combined

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    return output_path


def render_single_view_overview(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str | Path,
    *,
    elev: float,
    azim: float,
    image_size: int = 720,
    body_rgb: Tuple[int, int, int] = (210, 215, 225),
    background: str = "transparent",
) -> Path:
    """单视角整模型透明背景渲染：所有面统一染成 body_rgb，背景透明。
    用作 HTML 报告顶部的镇页图。
    """
    n_faces = _faces_count(faces)
    return render_single_view_status(
        vertices,
        faces,
        ["none"] * n_faces,  # 占位，实际颜色被 color_map 覆盖
        output_path,
        elev=elev,
        azim=azim,
        image_size=image_size,
        color_map={"none": body_rgb},
        background=background,
        title=None,
    )


def render_progress_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_status: Sequence[str],
    output_path: str | Path,
    *,
    title: str = "VLM 全局进度",
    image_size: int = 512,
    cols: int = 4,
    color_map: dict | None = None,
) -> Path:
    """
    渲染一张「全局进度」12 视角拼图，用于 VLM 自评：

    - face_status[i] ∈ {"full", "pending", "none", "abandoned"}
        - "full"      → 鲜红
        - "pending"   → 亮黄（仍在排队等待分裂的 partial）
        - "none"      → 浅灰
        - "abandoned" → 中灰

    传入 color_map 可覆盖默认配色（每个 status → (R,G,B)）。
    """
    n_faces = _faces_count(faces)
    if len(face_status) != n_faces:
        raise ValueError(
            f"face_status 长度 {len(face_status)} ≠ faces.shape[0] {n_faces}"
        )

    cmap = {
        "full": PROGRESS_FULL_RGB,
        "pending": PROGRESS_PENDING_RGB,
        "none": PROGRESS_NONE_RGB,
        "abandoned": PROGRESS_ABANDONED_RGB,
    }
    if color_map:
        cmap.update({k: tuple(v) for k, v in color_map.items()})

    face_colors = np.zeros((n_faces, 3), dtype=np.uint8)
    for i, st in enumerate(face_status):
        face_colors[i] = np.asarray(
            cmap.get(st, PROGRESS_NONE_RGB), dtype=np.uint8
        )

    views = _render_views_with_color_table(
        vertices, faces, face_colors, image_size=(image_size, image_size)
    )
    return _compose_panel(views, output_path=Path(output_path), title=title, cols=cols)


# 12 种高对比候选调色板（用于 grouped split 的 N≤12 个候选区）
CANDIDATE_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (228, 26, 28),    # 1 红
    (55, 126, 184),   # 2 蓝
    (77, 175, 74),    # 3 绿
    (152, 78, 163),   # 4 紫
    (255, 127, 0),    # 5 橙
    (255, 215, 30),   # 6 黄
    (166, 86, 40),    # 7 棕
    (247, 129, 191),  # 8 粉
    (0, 200, 200),    # 9 青
    (140, 86, 75),    # 10 巧克力
    (23, 190, 207),   # 11 天青
    (180, 180, 50),   # 12 橄榄
)


def _project_label_overlay(
    plotter: "Any",
    centroids_3d: np.ndarray,
    labels: Sequence[str],
    *,
    font_size: int = 28,
) -> None:
    """在 plotter 当前相机下，把 3D 重心投到屏幕空间并加文字标签。
    pyvista 的 add_point_labels 自动用 vtkLabeledDataMapper 做屏幕投影。
    """
    import pyvista as pv

    if len(centroids_3d) == 0:
        return
    pdata = pv.PolyData(centroids_3d.astype(np.float64))
    plotter.add_point_labels(
        pdata,
        list(labels),
        font_size=font_size,
        point_size=12,
        point_color="black",
        text_color="white",
        shape="rounded_rect",
        shape_color="black",
        shape_opacity=0.85,
        always_visible=True,
        bold=True,
        margin=4,
        render_points_as_spheres=True,
    )


def render_candidate_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    candidate_face_labels: Sequence[int],
    output_path: str | Path,
    *,
    title: str = "VLM 候选分组",
    image_size: int = 512,
    cols: int = 4,
    palette: Tuple[Tuple[int, int, int], ...] = CANDIDATE_PALETTE,
    dim_rgb: Tuple[int, int, int] = DIM_RGB,
) -> Path:
    """
    渲染过分割面板：

    - candidate_face_labels[i] ∈ {0, 1, ..., N-1, -1}
        - 0..N-1 → 候选编号，画上对应调色板颜色 + 数字标签 "1".."N"
        - -1     → 该面不在当前 partial 块内，染成淡色 (dim_rgb)，仅用于上下文

    每个候选区在 3D 重心位置画一个带数字的标签，便于 VLM 引用。
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 pyvista") from exc
    from .heatmap_renderer import _apply_camera, _to_pyvista_faces

    n_faces = _faces_count(faces)
    labels_arr = np.asarray(candidate_face_labels, dtype=np.int64)
    if labels_arr.shape[0] != n_faces:
        raise ValueError(
            f"candidate_face_labels 长度 {labels_arr.shape[0]} ≠ faces.shape[0] {n_faces}"
        )

    used_ids = sorted({int(v) for v in labels_arr if v >= 0})
    if not used_ids:
        raise ValueError("candidate_face_labels 里没有任何 ≥0 的候选 id")

    if len(used_ids) > len(palette):
        raise ValueError(
            f"候选数 {len(used_ids)} 超过调色板大小 {len(palette)}"
        )

    # 为每个 face 上色
    face_colors = np.tile(np.asarray(dim_rgb, dtype=np.uint8), (n_faces, 1))
    for new_idx, cid in enumerate(used_ids):
        mask = labels_arr == cid
        face_colors[mask] = np.asarray(palette[new_idx], dtype=np.uint8)

    # 计算每个候选的 3D 重心（取面中心的均值）
    face_centers = vertices[faces].mean(axis=1)  # (F, 3)
    centroids: List[np.ndarray] = []
    label_strs: List[str] = []
    for new_idx, cid in enumerate(used_ids):
        m = labels_arr == cid
        if not np.any(m):
            continue
        c = face_centers[m].mean(axis=0)
        centroids.append(c)
        label_strs.append(str(new_idx + 1))  # 1-indexed for VLM 友好
    centroids_arr = np.asarray(centroids, dtype=np.float64)

    pv_faces = _to_pyvista_faces(faces)
    pdata = pv.PolyData(vertices, pv_faces)
    pdata.cell_data["RGB"] = face_colors

    views: List[np.ndarray] = []
    for elev, azim in VIEW_GRID:
        plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size))
        plotter.set_background("white")
        plotter.add_mesh(
            pdata,
            scalars="RGB",
            rgb=True,
            show_scalar_bar=False,
            smooth_shading=False,  # 关闭平滑着色，保持每面纯色不混
            split_sharp_edges=False,
            lighting=True,
            ambient=0.45,           # 提亮环境光，避免暗面色相被压成同一调
            diffuse=0.55,
        )
        _apply_camera(plotter, vertices, elev, azim)
        _project_label_overlay(plotter, centroids_arr, label_strs)
        plotter.enable_anti_aliasing("ssaa")
        views.append(plotter.screenshot(return_img=True))
        plotter.close()

    # 拼图 + 标题（含调色板图例）
    legend_lines: List[str] = []
    for new_idx, cid in enumerate(used_ids):
        rgb = palette[new_idx]
        legend_lines.append(
            f"{new_idx + 1}=({rgb[0]:>3},{rgb[1]:>3},{rgb[2]:>3})"
        )
    full_title = (
        f"{title}  candidates={len(used_ids)}  | 编号→颜色: "
        + "  ".join(legend_lines)
    )
    return _compose_panel(
        views,
        output_path=Path(output_path),
        title=full_title,
        cols=cols,
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


# 全覆盖模式下叶分块染色用的高对比调色板（≤16 个语义部位）
FULL_COVERAGE_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (228, 26, 28),    # 红
    (55, 126, 184),   # 蓝
    (77, 175, 74),    # 绿
    (152, 78, 163),   # 紫
    (255, 127, 0),    # 橙
    (255, 215, 30),   # 黄
    (166, 86, 40),    # 棕
    (247, 129, 191),  # 粉
    (0, 200, 200),    # 青
    (140, 86, 75),    # 巧克力
    (23, 190, 207),   # 天青
    (180, 180, 50),   # 橄榄
    (102, 194, 165),  # 薄荷
    (252, 141, 98),   # 杏
    (141, 160, 203),  # 灰蓝
    (231, 138, 195),  # 玫瑰
)
FULL_COVERAGE_BG_RGB: Tuple[int, int, int] = (235, 235, 240)
FULL_COVERAGE_PENDING_RGB: Tuple[int, int, int] = (255, 215, 130)


def render_full_coverage_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    semantic_labels: Sequence[str],
    output_path: str | Path,
    *,
    title: str = "全覆盖语义分割",
    image_size: int = 512,
    cols: int = 4,
    palette: Tuple[Tuple[int, int, int], ...] = FULL_COVERAGE_PALETTE,
    background_rgb: Tuple[int, int, int] = FULL_COVERAGE_BG_RGB,
    pending_rgb: Tuple[int, int, int] = FULL_COVERAGE_PENDING_RGB,
) -> Path:
    """渲染全覆盖语义分割结果的 12 视角拼图：

    - ``face_semantic_index[i]``:
        - ``-1`` → background / 不属于任何已枚举部位 → 用 ``background_rgb`` 染色
        - ``-2`` → pending（迭代未结束就保存）→ 用 ``pending_rgb``
        - ``0..K-1`` → 对应 ``semantic_labels[k]`` → 用 ``palette[k % len(palette)]``
    - 标题区域附带「编号 → 颜色」图例，便于核对配色。
    """
    n_faces = _faces_count(faces)
    indices = np.asarray(face_semantic_index, dtype=np.int64)
    if indices.shape[0] != n_faces:
        raise ValueError(
            f"face_semantic_index 长度 {indices.shape[0]} ≠ faces.shape[0] {n_faces}"
        )
    n_sem = len(semantic_labels)
    if n_sem == 0:
        raise ValueError("semantic_labels 不能为空")

    face_colors = np.tile(np.asarray(background_rgb, dtype=np.uint8), (n_faces, 1))
    bg_mask = indices == -1
    pending_mask = indices == -2
    face_colors[bg_mask] = np.asarray(background_rgb, dtype=np.uint8)
    face_colors[pending_mask] = np.asarray(pending_rgb, dtype=np.uint8)
    for k in range(n_sem):
        m = indices == k
        if not np.any(m):
            continue
        face_colors[m] = np.asarray(palette[k % len(palette)], dtype=np.uint8)

    views = _render_views_with_color_table(
        vertices, faces, face_colors, image_size=(image_size, image_size)
    )

    legend_lines: List[str] = []
    for k, name in enumerate(semantic_labels):
        rgb = palette[k % len(palette)]
        legend_lines.append(
            f"{k + 1}={name} ({rgb[0]:>3},{rgb[1]:>3},{rgb[2]:>3})"
        )
    legend = "  ".join(legend_lines)
    full_title = f"{title}  parts={n_sem}  | {legend}"
    return _compose_panel(
        views,
        output_path=Path(output_path),
        title=full_title,
        cols=cols,
    )


def render_overview_only_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "整模型概览",
    image_size: int = 512,
    cols: int = 4,
    body_rgb: Tuple[int, int, int] = (210, 215, 225),
) -> Path:
    """渲染一张「整模型同色 12 视角拼图」，用于 VLM 全覆盖语义枚举的输入图。"""
    n_faces = _faces_count(faces)
    face_colors = np.tile(np.asarray(body_rgb, dtype=np.uint8), (n_faces, 1))
    views = _render_views_with_color_table(
        vertices, faces, face_colors, image_size=(image_size, image_size)
    )
    return _compose_panel(
        views,
        output_path=Path(output_path),
        title=title,
        cols=cols,
    )


__all__ = [
    "ACCEPTED_RGB",
    "CANDIDATE_PALETTE",
    "DIM_RGB",
    "FULL_COVERAGE_BG_RGB",
    "FULL_COVERAGE_PALETTE",
    "FULL_COVERAGE_PENDING_RGB",
    "HIGHLIGHT_RGB",
    "HIGHLIGHT_RGB_SECONDARY",
    "REJECTED_RGB",
    "PROGRESS_FULL_RGB",
    "PROGRESS_PENDING_RGB",
    "PROGRESS_NONE_RGB",
    "PROGRESS_ABANDONED_RGB",
    "build_highlight_face_colors",
    "mesh_face_connected_components",
    "render_candidate_panel",
    "render_full_coverage_panel",
    "render_highlighted_component_panel",
    "render_initial_segmentation_panel",
    "render_overview_only_panel",
    "render_progress_panel",
    "render_semantic_mask_panel",
    "render_single_view_overview",
    "render_single_view_status",
]
