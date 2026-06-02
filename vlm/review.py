"""
全局自评：每轮迭代后让 VLM 看一次「整模型 face 染色进度图」，自己评估当前红色质量
并决定是否终止迭代。

接口：
    review_progress(client, panel_path, semantic, *, iter_done, max_depth,
                    full_faces, pending_faces, abandoned_faces, total_faces,
                    model=None) -> ProgressReview
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import VLMClient, VLMResponse, parse_json_response
from .prompts import build_review_prompt

LOGGER = logging.getLogger("vlm.review")


@dataclass
class ProgressReview:
    """一次全局自评的结构化结果。"""

    should_terminate: bool = False
    red_assessment: str = ""        # matches / over_claimed / incomplete / partial
    yellow_assessment: str = ""     # should_keep_splitting / likely_not_semantic / mostly_done
    red_covered_parts: List[str] = field(default_factory=list)
    missing_regions: str = ""
    extra_regions: str = ""
    confidence: float = 0.0
    reason: str = ""
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return []


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        return s in ("true", "yes", "1")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def review_progress(
    client: VLMClient,
    panel_path: str | Path,
    semantic: str,
    *,
    iter_done: int,
    max_depth: int,
    full_faces: int,
    pending_faces: int,
    abandoned_faces: int,
    total_faces: int,
    model: Optional[str] = None,
) -> ProgressReview:
    """
    把全局进度面板送给 VLM，让它自评 + 决定是否终止。
    失败时返回 should_terminate=False 以便继续按 max_depth 收尾。
    """
    panel_path = Path(panel_path)
    prompt = build_review_prompt(
        semantic,
        iter_done=iter_done,
        max_depth=max_depth,
        full_faces=full_faces,
        pending_faces=pending_faces,
        abandoned_faces=abandoned_faces,
        total_faces=total_faces,
    )

    try:
        response: VLMResponse = client.chat(
            user_prompt=prompt.user,
            image_paths=[panel_path],
            system=prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 全局自评调用失败: %s", exc)
        return ProgressReview(
            should_terminate=False,
            reason="VLM 全局自评调用失败",
            error=str(exc),
        )

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning("VLM 全局自评响应非 JSON: %s", text[:200])
        return ProgressReview(
            should_terminate=False,
            reason=text.strip()[:200],
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            raw_response=response.raw,
        )

    return ProgressReview(
        should_terminate=_coerce_bool(payload.get("should_terminate")),
        red_assessment=str(payload.get("red_assessment") or "").strip(),
        yellow_assessment=str(payload.get("yellow_assessment") or "").strip(),
        red_covered_parts=_coerce_str_list(payload.get("red_covered_parts")),
        missing_regions=str(payload.get("missing_regions") or "").strip(),
        extra_regions=str(payload.get("extra_regions") or "").strip(),
        confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0))),
        reason=str(payload.get("reason") or "").strip(),
        raw_text=text,
        elapsed_seconds=response.elapsed_seconds,
        provider=response.provider,
        model=response.model,
        raw_response=response.raw,
    )


__all__ = ["ProgressReview", "review_progress"]
