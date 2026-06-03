"""
Full-coverage（全覆盖）模式下的两次 VLM 调用：

1. ``enumerate_semantics``：看整模型 12 视角拼图，返回**短名、粗粒度、全覆盖**部位清单
   （初稿，供后续切分细化）。
2. ``classify_multi_label``：在高亮分块图上，从封闭候选集中勾选实际覆盖项（0~多项）。
3. ``resolve_empty_semantic_repairs``：对 0 face 枚举语义，决定 split 或 delete。

调用方（``pipeline.full_coverage``）会根据这两个函数的输出做：
    - matched 1 个 → 直接归到该候选
    - matched ≥2 个 → 分裂成 len(matched) 个种子，继续走 CC → segmat → FPS-spectral 级联
    - matched 为空时由解析层/流水线强制归入一项（不允许 background）

本模块**不**做任何渲染，渲染由 ``rendering.render_component`` 负责。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .client import VLMClient, VLMResponse, parse_json_response
from .prompts import (
    build_empty_semantic_repair_prompt,
    build_enumerate_prompt,
    build_multi_label_prompt,
)

LOGGER = logging.getLogger("vlm.full_coverage")


# ---------------------------------------------------------------------------
# Stage 1: enumerate
# ---------------------------------------------------------------------------


@dataclass
class SemanticEnumeration:
    """VLM 给出的全覆盖语义清单。"""

    parts: List[str] = field(default_factory=list)
    object_kind: str = ""
    reason: str = ""
    confidence: float = 0.0
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.parts) and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_kind": self.object_kind,
            "semantic_parts": list(self.parts),
            "reason": self.reason,
            "confidence": self.confidence,
            "elapsed_seconds": self.elapsed_seconds,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
        }


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


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for x in items:
        key = x.strip()
        if not key or key in seen:
            continue
        seen[key] = 1
        out.append(key)
    return out


def enumerate_semantics(
    client: VLMClient,
    overview_panel: str | Path,
    *,
    hint: str = "",
    model: Optional[str] = None,
    min_parts: int = 2,
    max_parts: int = 12,
) -> SemanticEnumeration:
    """让 VLM 看一张整模型的 12 视角拼图，返回全覆盖部位清单。

    overview_panel 应当是「整模型一种淡色、无高亮」的渲染图——例如调用
    ``rendering.render_component.render_initial_segmentation_panel``
    传入全 0 标签得到的结果，或重新渲染 12 视角同色面板。
    """
    overview_panel = Path(overview_panel)
    prompt = build_enumerate_prompt(hint=hint)
    try:
        response: VLMResponse = client.chat(
            user_prompt=prompt.user,
            image_paths=[overview_panel],
            system=prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 全覆盖枚举调用失败: %s", exc)
        return SemanticEnumeration(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning("VLM 全覆盖枚举响应非 JSON: %s", text[:200])
        return SemanticEnumeration(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            raw_response=response.raw,
        )

    parts = _dedupe_keep_order(_coerce_str_list(payload.get("semantic_parts")))
    if len(parts) > max_parts:
        LOGGER.warning(
            "VLM 给出 %d 项 semantic_parts，截断到 %d 项",
            len(parts),
            max_parts,
        )
        parts = parts[:max_parts]
    if len(parts) < max(2, int(min_parts)):
        return SemanticEnumeration(
            parts=parts,
            object_kind=str(payload.get("object_kind") or "").strip(),
            reason=str(payload.get("reason") or "").strip(),
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"semantic_parts 数量 ({len(parts)}) 少于 min_parts={min_parts}",
            raw_response=response.raw,
        )

    return SemanticEnumeration(
        parts=parts,
        object_kind=str(payload.get("object_kind") or "").strip(),
        reason=str(payload.get("reason") or "").strip(),
        raw_text=text,
        elapsed_seconds=response.elapsed_seconds,
        provider=response.provider,
        model=response.model,
        raw_response=response.raw,
    )


# ---------------------------------------------------------------------------
# Stage 2: per-block multi-label classify
# ---------------------------------------------------------------------------


@dataclass
class MultiLabelClassification:
    """VLM 对单分块给出的 multi-label 结果（限定在一个封闭候选集内）。"""

    matched_parts: List[str] = field(default_factory=list)
    is_background: bool = False
    dominant: str = ""
    confidences: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_matched(self) -> int:
        return len(self.matched_parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_parts": list(self.matched_parts),
            "is_background": bool(self.is_background),
            "dominant": self.dominant,
            "confidences": dict(self.confidences),
            "notes": self.notes,
            "elapsed_seconds": self.elapsed_seconds,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
        }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _coerce_confidences(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in value.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        out[str(k).strip()] = max(0.0, min(1.0, f))
    return out


def _match_candidate(value: str, candidates: Sequence[str]) -> Optional[str]:
    """把 VLM 返回的一项对回到 candidate 集合（精确匹配优先；
    否则去掉空格做大小写不敏感比较）。匹配失败返回 None。"""
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    for c in candidates:
        if s == c:
            return c
    s_norm = s.lower().replace(" ", "").replace("　", "")
    for c in candidates:
        c_norm = c.lower().replace(" ", "").replace("　", "")
        if s_norm == c_norm:
            return c
    # 子串匹配（候选 ⊆ 返回值，或反之）——保守只在长度 ≥ 2 时启用
    for c in candidates:
        if len(c) >= 2 and (c in s or s in c):
            return c
    return None


def classify_multi_label(
    client: VLMClient,
    image_path: str | Path,
    candidate_parts: Sequence[str],
    *,
    model: Optional[str] = None,
) -> MultiLabelClassification:
    """对一张高亮 12 视角图调 VLM，返回该分块覆盖到的候选语义子集。

    与单语义 ``classify_component`` 的区别：
      - 输入候选集是封闭的（来自 ``enumerate_semantics``）；
      - 输出是 **subset** 而非 full/partial/none；
      - 调用方据 len(matched_parts) 决定后续动作（接受 / 分裂）；空匹配会强制归入 dominant。
    """
    image_path = Path(image_path)
    candidates = [str(x).strip() for x in candidate_parts if str(x).strip()]
    if not candidates:
        raise ValueError("candidate_parts 不能为空")

    prompt = build_multi_label_prompt(candidates)
    try:
        response: VLMResponse = client.chat(
            user_prompt=prompt.user,
            image_paths=[image_path],
            system=prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 多标签分类调用失败 (%s): %s", image_path.name, exc)
        return MultiLabelClassification(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning("VLM 多标签响应非 JSON (%s): %s", image_path.name, text[:200])
        return MultiLabelClassification(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            raw_response=response.raw,
        )

    raw_matched = _coerce_str_list(payload.get("matched_parts"))
    matched: List[str] = []
    seen: Dict[str, int] = {}
    for raw in raw_matched:
        m = _match_candidate(raw, candidates)
        if m is None:
            LOGGER.info(
                "multi_label: 丢弃候选集外项 '%s'（panel=%s）",
                raw,
                image_path.name,
            )
            continue
        if m in seen:
            continue
        seen[m] = 1
        matched.append(m)

    dominant_raw = str(payload.get("dominant") or "").strip()
    dominant = _match_candidate(dominant_raw, candidates) or ""
    if dominant and dominant not in matched:
        dominant = ""
    if not matched:
        if dominant:
            matched = [dominant]
        else:
            matched = [candidates[0]]
            dominant = candidates[0]
            LOGGER.info(
                "multi_label: 空匹配，强制归入 '%s'（panel=%s）",
                candidates[0],
                image_path.name,
            )
    elif not dominant:
        dominant = matched[0]

    confidences = _coerce_confidences(payload.get("confidences"))
    confidences = {k: v for k, v in confidences.items() if k in matched}

    notes = str(payload.get("notes") or "").strip()

    return MultiLabelClassification(
        matched_parts=matched,
        is_background=False,
        dominant=dominant,
        confidences=confidences,
        notes=notes,
        raw_text=text,
        elapsed_seconds=response.elapsed_seconds,
        provider=response.provider,
        model=response.model,
        raw_response=response.raw,
    )


# ---------------------------------------------------------------------------
# Stage 3: repair 0-face semantics (split from donor block or delete)
# ---------------------------------------------------------------------------


@dataclass
class EmptySemanticRepair:
    """对单个 0 face 语义的补救决策。"""

    empty_semantic: str
    action: str  # "split" | "delete"
    donor_block_id: str = ""
    reason: str = ""


@dataclass
class EmptySemanticRepairPlan:
    """VLM 对全部 0 face 语义的补救方案。"""

    repairs: List[EmptySemanticRepair] = field(default_factory=list)
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.repairs) and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repairs": [
                {
                    "empty_semantic": r.empty_semantic,
                    "action": r.action,
                    "donor_block_id": r.donor_block_id,
                    "reason": r.reason,
                }
                for r in self.repairs
            ],
            "elapsed_seconds": self.elapsed_seconds,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
        }


def _normalize_repair_action(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in ("split", "carve", "divide"):
        return "split"
    if s in ("delete", "remove", "drop"):
        return "delete"
    return s


def resolve_empty_semantic_repairs(
    client: VLMClient,
    coverage_panel: str | Path,
    *,
    semantic_parts: Sequence[str],
    face_counts: Dict[str, int],
    empty_parts: Sequence[str],
    donor_blocks: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
) -> EmptySemanticRepairPlan:
    """让 VLM 对 0 face 语义决定 split（从 donor 块 carve）或 delete。"""
    empty_parts = [str(x).strip() for x in empty_parts if str(x).strip()]
    if not empty_parts:
        return EmptySemanticRepairPlan()

    prompt = build_empty_semantic_repair_prompt(
        semantic_parts=list(semantic_parts),
        face_counts=dict(face_counts),
        empty_parts=empty_parts,
        donor_blocks=list(donor_blocks),
    )
    coverage_panel = Path(coverage_panel)
    try:
        response: VLMResponse = client.chat(
            user_prompt=prompt.user,
            image_paths=[coverage_panel],
            system=prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 0-face 语义补救调用失败: %s", exc)
        return EmptySemanticRepairPlan(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning("VLM 0-face 补救响应非 JSON: %s", text[:200])
        return EmptySemanticRepairPlan(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            raw_response=response.raw,
        )

    raw_repairs = payload.get("repairs")
    if not isinstance(raw_repairs, list):
        return EmptySemanticRepairPlan(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error="repairs 字段缺失或不是 list",
            raw_response=response.raw,
        )

    empty_set = set(empty_parts)
    donor_ids = {
        str(b.get("block_id", "")).strip()
        for b in donor_blocks
        if str(b.get("block_id", "")).strip()
    }
    repairs: List[EmptySemanticRepair] = []
    seen_empty: Dict[str, int] = {}

    for item in raw_repairs:
        if not isinstance(item, dict):
            continue
        empty_sem = str(item.get("empty_semantic") or "").strip()
        if not empty_sem or empty_sem not in empty_set:
            continue
        if empty_sem in seen_empty:
            continue
        action = _normalize_repair_action(item.get("action"))
        if action not in ("split", "delete"):
            LOGGER.info(
                "0-face repair: 忽略未知 action '%s' (%s)",
                item.get("action"),
                empty_sem,
            )
            continue
        donor_id = str(item.get("donor_block_id") or "").strip()
        if action == "split":
            if donor_id not in donor_ids:
                LOGGER.info(
                    "0-face repair: split '%s' 的 donor '%s' 无效，改 delete",
                    empty_sem,
                    donor_id,
                )
                action = "delete"
                donor_id = ""
        else:
            donor_id = ""
        repairs.append(
            EmptySemanticRepair(
                empty_semantic=empty_sem,
                action=action,
                donor_block_id=donor_id,
                reason=str(item.get("reason") or "").strip(),
            )
        )
        seen_empty[empty_sem] = 1

    # 未覆盖的 0-face 语义：无 donor 则 delete，否则 split 到最大 donor
    for empty_sem in empty_parts:
        if empty_sem in seen_empty:
            continue
        if donor_blocks:
            best = max(donor_blocks, key=lambda b: int(b.get("faces", 0)))
            repairs.append(
                EmptySemanticRepair(
                    empty_semantic=empty_sem,
                    action="split",
                    donor_block_id=str(best.get("block_id", "")).strip(),
                    reason="VLM 未覆盖，fallback split",
                )
            )
        else:
            repairs.append(
                EmptySemanticRepair(
                    empty_semantic=empty_sem,
                    action="delete",
                    reason="VLM 未覆盖且无 donor，fallback delete",
                )
            )

    return EmptySemanticRepairPlan(
        repairs=repairs,
        raw_text=text,
        elapsed_seconds=response.elapsed_seconds,
        provider=response.provider,
        model=response.model,
        raw_response=response.raw,
    )


__all__ = [
    "EmptySemanticRepair",
    "EmptySemanticRepairPlan",
    "MultiLabelClassification",
    "SemanticEnumeration",
    "classify_multi_label",
    "enumerate_semantics",
    "resolve_empty_semantic_repairs",
]
