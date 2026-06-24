"""Derive point-cloud benchmark query text from rendered GT highlights.

For each GT part we render the full point cloud with the target GT points in
red and all other points in grey, then ask the VLM for a concise semantic
description of the highlighted region. The returned phrase is used as the
semantic query for ``ours_pointcloud``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from rendering.pointcloud_render import compute_point_normals, render_pointcloud_highlight_panel
from vlm.client import VLMClient, VLMResponse, parse_json_response

LOGGER = logging.getLogger("benchmark.pointcloud_gt_semantic")

_SYS_PROMPT = (
    "You are a 3D point-cloud asset analyst. You are given a multi-view rendering "
    "of the same object. The vivid red points mark one target part and the light "
    "grey points show the rest of the complete object.\n"
    "THE RED POINTS ARE THE GROUND TRUTH AND THE ONLY AUTHORITY for what the "
    "target is. Read the actual red extent across all views before writing "
    "anything. INCLUDE every structure that the red points actually cover, and "
    "EXCLUDE only structures that are shown in grey. Your description must match "
    "the red region as drawn, not a textbook idea of the part.\n"
    "The object category and the benchmark part label are only WEAK NAMING HINTS. "
    "Do NOT let the label override what you see. Benchmark labels are often "
    "COARSE and merge several intuitive sub-parts into one: e.g. a 'base' may "
    "actually include the legs, the lower frame, the side armrests and the back "
    "uprights all together. If the red clearly covers the armrests, you MUST "
    "include the armrests even though the label only says 'base'. Never exclude a "
    "structure just because the label name would not normally contain it — only "
    "exclude it if it is grey.\n"
    "Conversely, do not add structures to include that are grey, even if the "
    "label name suggests them.\n"
    "CRITICAL — a functional relationship is NOT membership, and this rule cuts "
    "BOTH ways. Judge every structure ONLY by the colour of its own points, never "
    "by what it is attached to nor by a fixed idea of which parts 'go together':\n"
    "  (a) Do NOT add a grey structure just because it touches, holds, connects to "
    "or hangs from the red part.\n"
    "  (b) Do NOT drop a red structure just because it is attached to, hangs from, "
    "sits next to, or visually resembles a grey part.\n"
    "Before listing any structure, locate its actual points and read their colour: "
    "red → include, grey → exclude. No habit or label may override the colour.\n"
    "Worked example (same balance scale, two DIFFERENT targets, opposite answers): "
    "(1) when only the thin suspension strings/cords are red and the bowls/weighing "
    "pans, beam and pillar are grey → describe the strings alone and EXCLUDE the "
    "pans; (2) when the pillar, beam and the solid weighing pans/bowls are red while "
    "the thin cords are grey → you MUST INCLUDE the weighing pans and EXCLUDE the "
    "strings. The correct answer flips entirely with the colours.\n"
    "Watch out for adjacent-but-distinct structures, e.g. thin cords/strings/wires "
    "versus the solid bowls/pans/cups/plates hanging from them — they are easy to "
    "confuse. A thin red cord does NOT pull in the solid bowl below it; a red solid "
    "bowl does NOT pull in the thin grey cord above it. Name what the red geometry "
    "actually is (slender cord vs solid bowl/pan), and NEVER invent sub-categories "
    "(e.g. 'short vs long strings') just to reconcile the label with what you see.\n"
    "Do not mention colors, highlighting, GT, masks, point ids, or rendering artifacts. "
    "Do not output multiple alternatives.\n"
    "Output JSON only with these fields:\n"
    "{"
    "\"object_model\": \"e.g. airplane\", "
    "\"target_location\": \"object-relative location of the red region\", "
    "\"red_covers\": [\"every distinct structure the red points actually cover\"], "
    "\"grey_only\": [\"structures that are grey, i.e. NOT part of the target\"], "
    "\"include\": [\"= red_covers; semantic structures that belong to the target\"], "
    "\"exclude\": [\"= grey_only; only structures shown in grey\"], "
    "\"semantic\": \"one concise English search query combining model + location + everything the red covers + excluding only the grey structures\""
    "}"
)


def describe_gt_part_pointcloud(
    client: VLMClient,
    points: np.ndarray,
    labels: np.ndarray,
    part_id: int,
    out_dir: Path,
    *,
    object_category: str = "",
    target_part_name: str = "",
    model: Optional[str] = None,
    image_size: int = 512,
    point_size: Optional[float] = None,
) -> Tuple[str, Optional[Path], str]:
    """Render one GT-highlight panel and ask VLM for a search phrase.

    Returns ``(semantic, panel_path, raw_text)``. ``semantic`` is empty on
    render/VLM/parse failure.
    """
    pts = np.asarray(points, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int64)
    ids = np.flatnonzero(lab == int(part_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = out_dir / "gt_semantic_highlight.png"

    if ids.size == 0:
        LOGGER.warning("GT pointcloud part_id=%s has no points", part_id)
        return "", None, ""

    try:
        normals = compute_point_normals(pts)
        render_pointcloud_highlight_panel(
            pts,
            ids.tolist(),
            panel,
            title=f"GT部件高亮 points={int(ids.size)} / {len(pts)}",
            image_size=(image_size, image_size),
            cols=4,
            point_size=point_size,
            normals=normals,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("GT pointcloud panel render failed: %s", exc)
        return "", None, ""

    user = (
        "请先逐视角观察红色高亮点到底覆盖了哪些结构，再生成后续分割算法要查找的英文语义查询。\n"
        "红色 = 目标真值（唯一权威）；灰色 = 不属于目标。必须说明：\n"
        "1. 这是什么物体模型；\n"
        "2. 红色高亮区域在物体上的相对位置；\n"
        "3. red_covers：红色实际覆盖了哪些结构（逐一列出，哪怕标签名里通常不含它们）；\n"
        "4. grey_only：哪些结构是灰色、不属于目标；\n"
        "5. include 必须等于 red_covers，exclude 必须等于 grey_only；\n"
        "6. 最终 semantic 把『模型+位置+红色覆盖的全部结构+仅排除灰色结构』压缩成一句英文查询。\n"
        "关键：不要因为标签名（如 base）就排除红色明明覆盖的结构（如 armrests）；"
        "也不要把灰色结构写进 include。标签只是命名提示，红色才是准绳。\n"
        "特别注意：『功能关联』不等于『同属一个部件』，而且这条规则是双向的。每个结构只按"
        "它自身点的颜色判断，不要凭它连接到什么、也不要凭固定的『哪些部件通常在一起』的成见：\n"
        "  (a) 不要因为某灰色结构被红色部件连接/悬挂/紧邻，就把它写进 include；\n"
        "  (b) 也不要因为某红色结构挂在、贴着或长得像某个灰色部件，就把它从 include 里删掉。\n"
        "写每个结构前先定位它的实际点并读取颜色：红=include，灰=exclude，成见和标签都不能凌驾颜色。\n"
        "示例（同一台天平，两个不同目标，答案相反）：(1) 若只有细细的悬挂绳/吊索是红色，而秤盘、"
        "横梁、立柱是灰色 → 只描述绳索并排除秤盘；(2) 若立柱、横梁和实心的秤盘/盘子是红色，而细绳"
        "是灰色 → 必须把秤盘写进 include，并排除绳索。答案随颜色完全翻转。\n"
        "尤其当心相邻但不同的结构：细绳/线/缆 与挂在其下方的实心碗/盘/杯 很容易混淆。红色细绳不会"
        "牵连其下方的实心盘；红色实心盘也不会牵连其上方的灰色细绳。请如实命名红色几何到底是细绳还是"
        "实心盘，绝不要为了迁就标签而臆造出『短绳/长绳』之类的子类别。"
    )
    context_bits = []
    if object_category:
        context_bits.append(f"物体类别 object_model = {object_category}")
    if target_part_name:
        context_bits.append(
            f"基准部件标签（仅命名弱提示，可能是粗粒度合并标签）target_part_label = {target_part_name}"
        )
    if context_bits:
        user = "已知上下文（强约束）：\n" + "\n".join(f"- {b}" for b in context_bits) + "\n\n" + user
    try:
        resp: VLMResponse = client.chat(
            user_prompt=user,
            image_paths=[panel],
            system=_SYS_PROMPT,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("GT pointcloud semantic VLM call failed: %s", exc)
        return "", panel, ""

    text = resp.text or ""
    semantic = ""
    try:
        payload = parse_json_response(text)
        semantic = str(payload.get("semantic") or "").strip()
    except Exception:  # noqa: BLE001
        semantic = text.strip().splitlines()[0].strip() if text.strip() else ""
    semantic = semantic.strip().strip('“”"\'：: ')
    return semantic, panel, text
