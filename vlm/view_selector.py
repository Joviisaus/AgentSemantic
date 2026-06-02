"""
最佳视角选择（请求层）：让 VLM 看 12 张候选缩略图，挑出最能展示用户语义所在
区域的那一张视图（1-indexed 编号 + 理由）。

⚠️ 本模块**不**做任何渲染。调用前请由 `rendering.view_thumbnails.render_thumbnails_panel`
先把 12 视角缩略图拼成一张 PNG，再把路径传给 `select_best_view`。

VLM 调用失败 / JSON 不合法 / 编号越界时，自动 fallback 到 VIEW_GRID 里离
`heatmap_renderer.DEFAULT_OVERVIEW_VIEW` 最近的那个视图，保证主流程不被阻断。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rendering.heatmap_renderer import DEFAULT_OVERVIEW_VIEW, VIEW_GRID

from .client import VLMClient, VLMResponse, parse_json_response

LOGGER = logging.getLogger("vlm.view_selector")


SELECT_VIEW_SYSTEM_PROMPT = (
    "你是一个 3D 视图选择器。用户会给你 12 张同一模型不同视角的小图缩略图，"
    "每张图左上角都有一个白底黑字的编号 1..12。"
    "你需要挑出**最能完整、清晰地展示用户指定语义所在区域**的那一张视图，"
    "作为后续所有报告里唯一的展示视角。"
    "严格按 JSON 格式回答，禁止 Markdown，禁止额外文字。"
)


SELECT_VIEW_USER_TEMPLATE = """目标语义：「{semantic}」

下面 12 张缩略图分别是同一 3D 模型的 12 个不同视角（左上角编号 1..12，相机参数见下）：
{view_table}

请从这 12 张视图里挑出**最能展示「{semantic}」所在区域**的那一张，原则：
1. 该视图要能看到「{semantic}」的**主要几何**，不能被其他部位遮挡；
2. 优先选择「{semantic}」在画面里占比明显的视图（例如正面、3/4 侧面），
   避免选择背面或仅露出极小角度的视图；
3. 如果模型存在「左右对称」的语义（例如眼镜、双眼），优先选**正面或带轻微俯视的 3/4 角**；
4. 如果完全看不到「{semantic}」（例如它根本不存在于该模型），仍要选**最能体现整体结构**的视图，
   并在 reason 里说明「未发现明显语义区域」。

只能输出如下 JSON（缺字段视为错误）：
{{
  "best_index": <int, 必须在 1..12 之间>,
  "confidence": 0.0~1.0,
  "reason": "用一句中文说明：为什么这张视图最适合展示「{semantic}」？"
}}
"""


@dataclass
class BestViewSelection:
    """VLM 最佳视角选择结果。"""

    index: int = 0                    # 0-indexed 在 VIEW_GRID 里的下标（0..11）
    elev: float = 0.0
    azim: float = 0.0
    reason: str = ""
    confidence: float = 0.0
    fallback: bool = False            # 是否用了默认 overview 而不是 VLM 选择
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    thumbnails_panel: Optional[Path] = None
    overview_png: Optional[Path] = None  # 用 chosen view 渲染的整模型透明背景 PNG
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def view(self) -> Tuple[float, float]:
        return (self.elev, self.azim)


def _view_table_text() -> str:
    return "\n".join(
        f"- #{i+1}: elev={elev:.1f}°, azim={azim:.1f}°"
        for i, (elev, azim) in enumerate(VIEW_GRID)
    )


def _resolve_fallback_view() -> Tuple[int, float, float]:
    """选 VIEW_GRID 里离 DEFAULT_OVERVIEW_VIEW (22.6, 45.0) 最近的那个。"""
    target = np.asarray(DEFAULT_OVERVIEW_VIEW, dtype=np.float64)
    best_idx = 0
    best_dist = float("inf")
    for i, (elev, azim) in enumerate(VIEW_GRID):
        d = (elev - target[0]) ** 2 + (azim - target[1]) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = i
    elev, azim = VIEW_GRID[best_idx]
    return best_idx, float(elev), float(azim)


def select_best_view(
    client: VLMClient,
    panel_path: Path,
    semantic: str,
    *,
    model: Optional[str] = None,
) -> BestViewSelection:
    """让 VLM 看 12 张候选缩略图，挑出最能展示语义的视图。"""
    semantic = (semantic or "").strip() or "整体结构"
    user_prompt = SELECT_VIEW_USER_TEMPLATE.format(
        semantic=semantic, view_table=_view_table_text()
    )

    try:
        response: VLMResponse = client.chat(
            user_prompt=user_prompt,
            image_paths=[panel_path],
            system=SELECT_VIEW_SYSTEM_PROMPT,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 视角选择调用失败: %s — 退回默认 overview 视图", exc)
        idx, elev, azim = _resolve_fallback_view()
        return BestViewSelection(
            index=idx, elev=elev, azim=azim,
            reason="VLM 调用失败，退回默认 overview 视图",
            fallback=True, error=str(exc),
        )

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning("VLM 视角选择响应非 JSON: %s", text[:200])
        idx, elev, azim = _resolve_fallback_view()
        return BestViewSelection(
            index=idx, elev=elev, azim=azim,
            reason=f"JSON 解析失败：{text.strip()[:100]}",
            fallback=True,
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider, model=response.model,
            error=str(exc), raw_response=response.raw,
        )

    raw_idx = payload.get("best_index")
    try:
        idx_1based = int(raw_idx)
    except (TypeError, ValueError):
        idx_1based = -1
    if not 1 <= idx_1based <= len(VIEW_GRID):
        LOGGER.warning("VLM 给出非法 best_index=%r，退回默认 overview", raw_idx)
        idx, elev, azim = _resolve_fallback_view()
        return BestViewSelection(
            index=idx, elev=elev, azim=azim,
            reason=f"VLM best_index 非法：{raw_idx}",
            fallback=True, raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider, model=response.model,
            raw_response=response.raw,
        )

    idx = idx_1based - 1
    elev, azim = VIEW_GRID[idx]
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(payload.get("reason") or "").strip()
    return BestViewSelection(
        index=idx, elev=float(elev), azim=float(azim),
        reason=reason, confidence=confidence, fallback=False,
        raw_text=text, elapsed_seconds=response.elapsed_seconds,
        provider=response.provider, model=response.model,
        raw_response=response.raw,
    )


__all__ = [
    "BestViewSelection",
    "select_best_view",
]
