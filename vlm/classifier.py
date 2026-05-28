"""
分块三分类：调用 VLM 把单个分块判定为
    full     完全符合语义
    partial  局部符合语义
    none     完全不符合语义
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .client import VLMClient, VLMResponse, parse_json_response
from .prompts import LABEL_FULL, LABEL_NONE, LABEL_PARTIAL, VALID_LABELS, build_classify_prompt

LOGGER = logging.getLogger("vlm.classifier")


@dataclass
class ComponentClassification:
    label: str  # full / partial / none
    confidence: float = 0.0
    reason: str = ""
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_full(self) -> bool:
        return self.label == LABEL_FULL

    @property
    def is_partial(self) -> bool:
        return self.label == LABEL_PARTIAL

    @property
    def is_none(self) -> bool:
        return self.label == LABEL_NONE


def _normalize_label(value: Any) -> str:
    if not isinstance(value, str):
        return LABEL_NONE
    s = value.strip().lower()
    # 容错：中文回答
    cn_map = {
        "完全符合": LABEL_FULL,
        "完全符合语义": LABEL_FULL,
        "局部符合": LABEL_PARTIAL,
        "局部符合语义": LABEL_PARTIAL,
        "部分符合": LABEL_PARTIAL,
        "完全不符合": LABEL_NONE,
        "完全不符合语义": LABEL_NONE,
        "不符合": LABEL_NONE,
    }
    if s in cn_map:
        return cn_map[s]
    if s in VALID_LABELS:
        return s
    if "full" in s and "partial" not in s:
        return LABEL_FULL
    if "partial" in s or "部分" in s or "局部" in s:
        return LABEL_PARTIAL
    if "none" in s or "no" == s or "不" in s:
        return LABEL_NONE
    return LABEL_NONE


def classify_component(
    client: VLMClient,
    image_path: str | Path,
    semantic: str,
    *,
    model: Optional[str] = None,
) -> ComponentClassification:
    """
    把高亮分块图发给 VLM，返回 ComponentClassification。
    失败会返回 label="none"、error 非空，由调用方决定是否当作"不符合"。
    """
    prompt = build_classify_prompt(semantic)
    image_path = Path(image_path)

    try:
        response: VLMResponse = client.chat(
            user_prompt=prompt.user,
            image_paths=[image_path],
            system=prompt.system,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 三分类调用失败 (%s): %s", image_path.name, exc)
        return ComponentClassification(
            label=LABEL_NONE,
            reason="VLM 调用失败",
            error=str(exc),
        )

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning("VLM 响应不是合法 JSON (%s): %s", image_path.name, text[:200])
        return ComponentClassification(
            label=_normalize_label(text),
            reason=text.strip()[:200],
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            raw_response=response.raw,
        )

    label = _normalize_label(payload.get("label"))
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    reason = str(payload.get("reason") or "").strip()
    return ComponentClassification(
        label=label,
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason,
        raw_text=text,
        elapsed_seconds=response.elapsed_seconds,
        provider=response.provider,
        model=response.model,
        raw_response=response.raw,
    )
