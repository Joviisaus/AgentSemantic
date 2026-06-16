"""GT-highlight panel from HF 2D renders + VLM semantic naming."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.hf_compat200.load import decode_part_stored, load_image_field
from vlm.client import VLMClient, VLMResponse, parse_json_response

LOGGER = logging.getLogger("benchmark.hf_compat200.gt_semantic")

HIGHLIGHT = np.array([255, 40, 40], dtype=np.uint8)
GREY = np.array([190, 190, 190], dtype=np.uint8)

_SYS_PROMPT = (
    "你是3D资产分析助手。给你的是同一个物体的多视角渲染拼图，其中**鲜艳红色**"
    "高亮的是该物体的某一个部件，其余部分为浅灰色。\n"
    "请只根据红色高亮区域所在的几何位置和形状，用一个**简洁的名词短语**命名这个"
    "部件（2-6个字，例如“桌面”“椅子腿”“瓶盖”）。\n"
    "要求：不要解释，不要提到“红色/高亮/颜色”，不要输出多个候选。\n"
    '只输出 JSON：{"semantic": "<部件名>"}'
)


def _highlight_view(render_rgb: np.ndarray, mask_rgb: np.ndarray,
                    target_stored: int) -> np.ndarray:
    render = np.asarray(render_rgb, dtype=np.uint8).copy()
    stored = decode_part_stored(mask_rgb)
    target = stored == int(target_stored)
    out = np.empty_like(render)
    out[:] = GREY
    out[target] = HIGHLIGHT
    # keep some shape context from the render on non-target pixels
    blend = render.astype(np.float32) * 0.25 + GREY.astype(np.float32) * 0.75
    out[~target] = blend[~target].astype(np.uint8)
    return out


def build_multiview_panel(
    rows: List[dict], target_stored: int, out_path: Path,
    *, image_size: int = 256, cols: int = 4,
) -> Path:
    """Build a multi-view GT highlight panel from HF rows (same shape/style)."""
    views = []
    labels = []
    for row in rows:
        render = np.array(load_image_field(row["render.png"]))
        mask = np.array(load_image_field(row["mask.png"]))
        views.append(_highlight_view(render, mask, target_stored))
        labels.append(f"elev/v{int(row['view_id.cls'])}")

    n = len(views)
    if n == 0:
        raise ValueError("no views to render")

    rows_n = (n + cols - 1) // cols
    cell = image_size
    title_h = 28
    panel = Image.new("RGB", (cols * cell, rows_n * (cell + title_h)), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, (img_arr, label) in enumerate(zip(views, labels)):
        r, c = divmod(i, cols)
        x0, y0 = c * cell, r * (cell + title_h)
        draw.text((x0 + 4, y0 + 2), label, fill=(30, 30, 30), font=font)
        tile = Image.fromarray(img_arr).resize((cell, cell), Image.Resampling.BILINEAR)
        panel.paste(tile, (x0, y0 + title_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)
    return out_path


def describe_gt_part_2d(
    client: VLMClient,
    rows: List[dict],
    target_stored: int,
    out_dir: Path,
    *,
    object_category: str = "",
    model: Optional[str] = None,
    image_size: int = 256,
) -> Tuple[str, Optional[Path], str]:
    """Render a 2D multi-view GT panel and ask the VLM for a concise semantic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = out_dir / "gt_highlight.png"
    try:
        build_multiview_panel(rows, target_stored, panel, image_size=image_size)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("2D GT panel render failed: %s", exc)
        return "", None, ""

    user = "请给出红色高亮部件的简洁语义名称。"
    if object_category:
        user = f"该物体是一个「{object_category}」。" + user
    try:
        resp: VLMResponse = client.chat(
            user_prompt=user, image_paths=[panel], system=_SYS_PROMPT,
            temperature=0.0, json_mode=True, model=model,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("HF GT-semantic VLM call failed: %s", exc)
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
