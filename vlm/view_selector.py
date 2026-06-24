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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rendering.heatmap_renderer import DEFAULT_OVERVIEW_VIEW, VIEW_GRID

from .client import VLMClient, VLMResponse, parse_json_response

LOGGER = logging.getLogger("vlm.view_selector")


SELECT_VIEW_SYSTEM_PROMPT = (
    "你是一个 3D 视图选择器。用户会给你 12 张同一模型不同视角的小图缩略图，"
    "每张图左上角都有一个白底黑字的编号 1..12。"
    "你需要做两件事：(1) 挑出**最能完整、清晰地展示用户指定语义所在区域**的那一张视图，"
    "作为后续所有报告里唯一的展示视角；(2) 决定后续**逐块语义判定**时，高亮拼图里应该"
    "保留**哪几张视角**——视角越少越精，能完整看清语义区域即可；视角太多会让模型在拼图里"
    "来回找红色、产生幻觉、虚报部位。通常 2~4 张互补视角足矣。"
    "严格按 JSON 格式回答，禁止 Markdown，禁止额外文字。"
)


SELECT_VIEW_USER_TEMPLATE = """目标语义：「{semantic}」

下面 12 张缩略图分别是同一 3D 模型的 12 个不同视角（左上角编号 1..12，相机参数见下）：
{view_table}

任务一·选展示视角：从这 12 张里挑出**最能展示「{semantic}」所在区域**的那一张，原则：
1. 该视图要能看到「{semantic}」的**主要几何**，不能被其他部位遮挡；
2. 优先选择「{semantic}」在画面里占比明显的视图（例如正面、3/4 侧面），
   避免选择背面或仅露出极小角度的视图；
3. 如果模型存在「左右对称」的语义（例如眼镜、双眼），优先选**正面或带轻微俯视的 3/4 角**；
4. 如果完全看不到「{semantic}」（例如它根本不存在于该模型），仍要选**最能体现整体结构**的视图，
   并在 reason 里说明「未发现明显语义区域」。

任务二·选分类视角集：后续每个分块会用「这些视角」拼成一张高亮图交给你判 full/partial/none。
请挑出一组**最少但足够**的互补视角（编号列表），原则：
1. **少而精**：视角太多会诱发幻觉/虚报，能从各角度看清「{semantic}」区域即可，**通常 {min_views}~{max_views} 张**；
2. 必须**包含 best_index**，并补上 1~3 个与它**角度互补**（正/侧/俯仰不同）的视角，避免遮挡盲区；
3. 若语义区域结构简单、单面即可看全，给 2 张也行；若左右对称或环绕分布，适当多给到 {max_views} 张；
4. 编号必须都在 1..12 内、不重复。

只能输出如下 JSON（缺字段视为错误）：
{{
  "best_index": <int, 必须在 1..12 之间>,
  "confidence": 0.0~1.0,
  "classify_view_indices": [<int 列表, 长度 {min_views}~{max_views}, 都在 1..12, 含 best_index>],
  "reason": "用一句中文说明：为什么这张视图最适合展示「{semantic}」、以及为什么选这几张分类视角？"
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
    # VLM 选定的「分类视角子集」（0-indexed，含 best index），后续逐块高亮拼图只渲染这些。
    classify_view_indices: List[int] = field(default_factory=list)

    @property
    def view(self) -> Tuple[float, float]:
        return (self.elev, self.azim)

    @property
    def classify_views(self) -> List[Tuple[float, float]]:
        """分类视角子集对应的 (elev, azim) 列表；为空时退回 best view 单视角。"""
        idxs = [i for i in self.classify_view_indices if 0 <= i < len(VIEW_GRID)]
        if not idxs:
            idxs = [self.index]
        return [VIEW_GRID[i] for i in idxs]


def _view_table_text() -> str:
    return "\n".join(
        f"- #{i+1}: elev={elev:.1f}°, azim={azim:.1f}°"
        for i, (elev, azim) in enumerate(VIEW_GRID)
    )


def _view_unit_vectors() -> np.ndarray:
    """VIEW_GRID 各视角的相机方向单位向量（用于挑角度互补的视角）。"""
    vecs = []
    for elev, azim in VIEW_GRID:
        e = np.deg2rad(elev)
        a = np.deg2rad(azim)
        vecs.append([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return np.asarray(vecs, dtype=np.float64)


def select_diverse_view_indices(
    best_index: int,
    count: int,
    *,
    must_include: Optional[Sequence[int]] = None,
) -> List[int]:
    """从 VIEW_GRID 里挑 ``count`` 个角度尽量互补的视角（0-indexed，必含 best_index）。

    用最远点采样（在相机方向单位球上）保证选出的视角彼此张角最大、互相补盲区。
    """
    n = len(VIEW_GRID)
    count = int(np.clip(count, 1, n))
    best_index = int(np.clip(best_index, 0, n - 1))
    vecs = _view_unit_vectors()

    chosen: List[int] = []
    for i in (must_include or []):
        ii = int(i)
        if 0 <= ii < n and ii not in chosen:
            chosen.append(ii)
    if best_index not in chosen:
        chosen.insert(0, best_index)
    chosen = chosen[:count]

    while len(chosen) < count:
        # 选与已选集合「最远」（角距最大）的视角
        d = np.full(n, np.inf)
        for c in chosen:
            ang = 1.0 - vecs @ vecs[c]  # 余弦距离，越大越互补
            d = np.minimum(d, ang)
        for c in chosen:
            d[c] = -1.0
        nxt = int(np.argmax(d))
        if d[nxt] < 0:
            break
        chosen.append(nxt)
    return chosen


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


def _parse_classify_view_indices(
    payload: Dict[str, Any],
    best_idx: int,
    *,
    min_views: int,
    max_views: int,
    default_views: int,
) -> List[int]:
    """解析并清洗 VLM 的 classify_view_indices；非法/缺失时按角度互补补齐。

    规则：转 0-indexed、去越界/去重、强制含 best_idx、数量裁剪到 [min_views, max_views]；
    不足时用 ``select_diverse_view_indices`` 补角度互补的视角。
    """
    n = len(VIEW_GRID)
    lo = int(np.clip(min_views, 1, n))
    hi = int(np.clip(max_views, lo, n))
    raw = payload.get("classify_view_indices")
    cleaned: List[int] = []
    if isinstance(raw, (list, tuple)):
        for v in raw:
            try:
                i0 = int(v) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= i0 < n and i0 not in cleaned:
                cleaned.append(i0)
    if best_idx not in cleaned:
        cleaned.insert(0, best_idx)
    if len(cleaned) > hi:
        # 超额：保留 best + 角度最互补的若干张
        cleaned = select_diverse_view_indices(best_idx, hi, must_include=cleaned[:1])
    elif len(cleaned) < lo:
        target = int(np.clip(default_views, lo, hi)) if not cleaned else lo
        cleaned = select_diverse_view_indices(
            best_idx, max(target, len(cleaned)), must_include=cleaned
        )
    return cleaned


def select_best_view(
    client: VLMClient,
    panel_path: Path,
    semantic: str,
    *,
    model: Optional[str] = None,
    min_classify_views: int = 2,
    max_classify_views: int = 6,
    default_classify_views: int = 4,
) -> BestViewSelection:
    """让 VLM 看 12 张候选缩略图，挑出最佳展示视图 + 后续分类用的视角子集。"""
    semantic = (semantic or "").strip() or "整体结构"
    n = len(VIEW_GRID)
    lo = int(np.clip(min_classify_views, 1, n))
    hi = int(np.clip(max_classify_views, lo, n))
    dflt = int(np.clip(default_classify_views, lo, hi))
    user_prompt = SELECT_VIEW_USER_TEMPLATE.format(
        semantic=semantic, view_table=_view_table_text(),
        min_views=lo, max_views=hi,
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
            classify_view_indices=select_diverse_view_indices(idx, dflt),
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
            classify_view_indices=select_diverse_view_indices(idx, dflt),
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
            classify_view_indices=select_diverse_view_indices(idx, dflt),
        )

    idx = idx_1based - 1
    elev, azim = VIEW_GRID[idx]
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(payload.get("reason") or "").strip()
    classify_idx = _parse_classify_view_indices(
        payload, idx, min_views=lo, max_views=hi, default_views=dflt,
    )
    LOGGER.info(
        "VLM 选定分类视角子集: %s (共 %d 张)",
        [i + 1 for i in classify_idx], len(classify_idx),
    )
    return BestViewSelection(
        index=idx, elev=float(elev), azim=float(azim),
        reason=reason, confidence=confidence, fallback=False,
        raw_text=text, elapsed_seconds=response.elapsed_seconds,
        provider=response.provider, model=response.model,
        raw_response=response.raw,
        classify_view_indices=classify_idx,
    )


__all__ = [
    "BestViewSelection",
    "select_best_view",
    "select_diverse_view_indices",
]
