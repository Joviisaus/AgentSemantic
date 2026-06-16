"""
Full-coverage（全覆盖）模式下的两次 VLM 调用：

1. ``enumerate_semantics``：看整模型 12 视角拼图，返回**短名、粗粒度、全覆盖**部位清单
   （初稿，供后续切分细化）。
2. ``classify_multi_label``：在高亮分块图上，从封闭候选集中勾选实际覆盖项（0~多项）。
3. ``classify_multi_label_coverage_repair``：分裂后子块并集缺语义时，提醒 VLM 重新勾选。
4. ``resolve_empty_semantic_repairs``：对 0 face 枚举语义，决定 split 或 delete。
5. ``infer_topology_constraints`` / ``audit_topology`` / ``resolve_topology_repairs``
   / ``assign_geometric_split_children``：收尾审计；多了→reassign；
   少了→geometric_split（VLM 定 seed_count 与 child_assignments，几何切分后再看图指派）。

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .client import VLMClient, VLMResponse, parse_json_response
from .prompts import (
    build_empty_semantic_repair_prompt,
    build_enumerate_prompt,
    build_geometric_split_assign_prompt,
    build_multi_label_prompt,
    build_split_coverage_repair_prompt,
    build_topology_infer_prompt,
    build_topology_repair_prompt,
)

LOGGER = logging.getLogger("vlm.full_coverage")

# 父块未标注语义：仅当 VLM 把握度 ≥ 此阈值时才保留
PARENT_OVERRIDE_MIN_CONFIDENCE = 0.85


# ---------------------------------------------------------------------------
# Stage 1: enumerate
# ---------------------------------------------------------------------------


@dataclass
class TopologyConstraint:
    """单个语义部位的拓扑约束。"""

    part: str
    count: int = 1
    attach_to: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part": self.part,
            "count": self.count,
            "attach_to": list(self.attach_to),
        }


@dataclass
class SemanticEnumeration:
    """VLM 给出的全覆盖语义清单。"""

    parts: List[str] = field(default_factory=list)
    object_kind: str = ""
    topology_constraints: List[TopologyConstraint] = field(default_factory=list)
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
            "topology_constraints": [c.to_dict() for c in self.topology_constraints],
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


def _coerce_int(value: Any, default: int = 1) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _parse_topology_constraints(
    raw: Any,
    parts: Sequence[str],
) -> List[TopologyConstraint]:
    """解析 VLM 返回的 topology_constraints，对齐到 parts 清单。"""
    part_set = {str(p).strip() for p in parts if str(p).strip()}
    if not isinstance(raw, list):
        return []
    by_part: Dict[str, TopologyConstraint] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        part = str(item.get("part") or "").strip()
        if not part or part not in part_set:
            continue
        attach_raw = item.get("attach_to")
        attach: List[str] = []
        if isinstance(attach_raw, list):
            for a in attach_raw:
                s = str(a).strip()
                if s and s in part_set and s != part:
                    attach.append(s)
        by_part[part] = TopologyConstraint(
            part=part,
            count=max(1, _coerce_int(item.get("count"), 1)),
            attach_to=attach,
        )
    # 保持 parts 顺序；缺失项用 count=1、attach_to=[]
    out: List[TopologyConstraint] = []
    for p in parts:
        ps = str(p).strip()
        if not ps:
            continue
        out.append(by_part.get(ps, TopologyConstraint(part=ps, count=1)))
    return out


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

    topo = _parse_topology_constraints(
        payload.get("topology_constraints"), parts
    )

    return SemanticEnumeration(
        parts=parts,
        object_kind=str(payload.get("object_kind") or "").strip(),
        topology_constraints=topo,
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


def _apply_parent_matched_filter(
    matched: List[str],
    confidences: Dict[str, float],
    dominant: str,
    parent_matched: Sequence[str],
    candidates: Sequence[str],
) -> Tuple[List[str], str, List[str]]:
    """
    父块未标注的语义从 matched 中剔除，除非 confidences ≥ PARENT_OVERRIDE_MIN_CONFIDENCE。
    返回 (filtered_matched, dominant, dropped_parts)。
    """
    parent_set = {str(x).strip() for x in parent_matched if str(x).strip()}
    if not parent_set:
        return matched, dominant, []

    kept: List[str] = []
    dropped: List[str] = []
    for m in matched:
        if m in parent_set:
            kept.append(m)
            continue
        conf = float(confidences.get(m, 0.0))
        if conf >= PARENT_OVERRIDE_MIN_CONFIDENCE:
            kept.append(m)
            LOGGER.info(
                "parent filter: 保留父块未标注项 '%s'（conf=%.2f≥%.2f）",
                m,
                conf,
                PARENT_OVERRIDE_MIN_CONFIDENCE,
            )
        else:
            dropped.append(m)
            LOGGER.info(
                "parent filter: 剔除 '%s'（不在 parent_matched %s, conf=%.2f）",
                m,
                sorted(parent_set),
                conf,
            )

    if not kept:
        if dominant in parent_set:
            kept = [dominant]
        else:
            for p in parent_matched:
                ps = str(p).strip()
                if ps in candidates:
                    kept = [ps]
                    dominant = ps
                    break
            if not kept and parent_matched:
                kept = [str(parent_matched[0]).strip()]
                dominant = kept[0]

    if dominant not in kept and kept:
        dominant = kept[0]

    return kept, dominant, dropped


def classify_multi_label(
    client: VLMClient,
    image_path: str | Path,
    candidate_parts: Sequence[str],
    *,
    block_id: str = "",
    parent_id: str = "",
    parent_matched: Optional[Sequence[str]] = None,
    at_max_depth: bool = False,
    current_depth: int = 0,
    max_depth: int = 0,
    model: Optional[str] = None,
) -> MultiLabelClassification:
    """对一张高亮 12 视角图调 VLM，返回该分块覆盖到的候选语义子集。

    与单语义 ``classify_component`` 的区别：
      - 输入候选集是封闭的（来自 ``enumerate_semantics``）；
      - 输出是 **subset** 而非 full/partial/none；
      - 若提供 ``parent_matched``，子块原则上只从父块标注中选，高把握例外见 prompt/后处理。
      - 若 ``at_max_depth=True``，提示 VLM 已达切分上限，必须给出唯一最终归属。

    调用方据 len(matched_parts) 决定后续动作（接受 / 分裂）；空匹配会强制归入 dominant。
    """
    image_path = Path(image_path)
    candidates = [str(x).strip() for x in candidate_parts if str(x).strip()]
    if not candidates:
        raise ValueError("candidate_parts 不能为空")

    parent_list = [str(x).strip() for x in (parent_matched or []) if str(x).strip()]
    prompt = build_multi_label_prompt(
        candidates,
        block_id=block_id,
        parent_id=parent_id,
        parent_matched=parent_list or None,
        at_max_depth=at_max_depth,
        current_depth=current_depth,
        max_depth=max_depth,
    )
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

    dropped_parent: List[str] = []
    if parent_list:
        matched, dominant, dropped_parent = _apply_parent_matched_filter(
            matched, confidences, dominant, parent_list, candidates,
        )
        confidences = {k: v for k, v in confidences.items() if k in matched}

    if at_max_depth and len(matched) > 1:
        pick = dominant if dominant in matched else matched[0]
        LOGGER.info(
            "multi_label: max_depth 强制收束 %s [%s] → [%s]",
            block_id or image_path.name,
            ", ".join(matched),
            pick,
        )
        matched = [pick]
        dominant = pick
        confidences = {k: v for k, v in confidences.items() if k in matched}

    notes = str(payload.get("notes") or "").strip()
    if at_max_depth:
        notes = (notes + " | max_depth_final").strip(" |")
    if dropped_parent:
        notes = (
            notes + f" | parent_filter_dropped={dropped_parent}"
        ).strip(" |")

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


def classify_multi_label_coverage_repair(
    client: VLMClient,
    image_path: str | Path,
    candidate_parts: Sequence[str],
    *,
    parent_id: str,
    block_id: str,
    parent_matched: Sequence[str],
    sibling_matched: Dict[str, Sequence[str]],
    missing_parts: Sequence[str],
    model: Optional[str] = None,
) -> MultiLabelClassification:
    """分裂后语义覆盖缺失：提醒 VLM 结合父/兄弟上下文重新勾选本块 matched_parts。"""
    image_path = Path(image_path)
    candidates = [str(x).strip() for x in candidate_parts if str(x).strip()]
    if not candidates:
        raise ValueError("candidate_parts 不能为空")

    prompt = build_split_coverage_repair_prompt(
        parent_id=parent_id,
        block_id=block_id,
        parent_matched=parent_matched,
        sibling_matched=sibling_matched,
        missing_parts=missing_parts,
        candidate_parts=candidates,
    )
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
        LOGGER.warning(
            "VLM 覆盖修复重分类失败 (%s): %s", image_path.name, exc
        )
        return MultiLabelClassification(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning(
            "VLM 覆盖修复响应非 JSON (%s): %s", image_path.name, text[:200]
        )
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


# ---------------------------------------------------------------------------
# Stage 4: topology infer / audit / repair
# ---------------------------------------------------------------------------


def infer_topology_constraints(
    client: VLMClient,
    overview_panel: str | Path,
    semantic_parts: Sequence[str],
    *,
    model: Optional[str] = None,
) -> List[TopologyConstraint]:
    """用户已给定 semantic_parts 时，让 VLM 补推断拓扑约束。"""
    parts = [str(x).strip() for x in semantic_parts if str(x).strip()]
    if not parts:
        return []
    prompt = build_topology_infer_prompt(parts)
    overview_panel = Path(overview_panel)
    try:
        response: VLMResponse = client.chat(
            user_prompt=prompt.user,
            image_paths=[overview_panel],
            system=prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
        payload = parse_json_response(response.text or "")
    except (RuntimeError, ValueError) as exc:
        LOGGER.warning("拓扑约束推断失败: %s", exc)
        return [TopologyConstraint(part=p, count=1) for p in parts]
    return _parse_topology_constraints(payload.get("topology_constraints"), parts)


@dataclass
class TopologyComponentInfo:
    """单个语义连通块信息。"""

    semantic: str
    index: int  # 1-based within semantic
    component_id: int  # global component id
    faces: int = 0
    face_ids: List[int] = field(default_factory=list)


@dataclass
class TopologyAudit:
    """拓扑审计结果。"""

    component_counts: Dict[str, int] = field(default_factory=dict)
    components: List[TopologyComponentInfo] = field(default_factory=list)
    adjacency_pairs: List[Tuple[str, str]] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_counts": dict(self.component_counts),
            "components": [
                {
                    "semantic": c.semantic,
                    "index": c.index,
                    "component_id": c.component_id,
                    "faces": c.faces,
                }
                for c in self.components
            ],
            "adjacency_pairs": [list(p) for p in self.adjacency_pairs],
            "violations": list(self.violations),
        }


def audit_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    semantic_parts: Sequence[str],
    topology_constraints: Sequence[TopologyConstraint],
    *,
    min_component_faces: int = 20,
) -> TopologyAudit:
    """根据连通块统计与约束检测拓扑违规。"""
    from geometry.semantic_rig import connected_components_per_part

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    parts = [str(x).strip() for x in semantic_parts if str(x).strip()]
    fpi = np.asarray(face_semantic_index, dtype=np.int64)

    comp_of_face, comp_part = connected_components_per_part(
        vertices, faces, fpi, min_component_faces
    )
    n_comp = len(comp_part)
    constraint_by_part = {c.part: c for c in topology_constraints}

    # per-semantic component counts and details
    sem_comp_ids: Dict[str, List[int]] = {p: [] for p in parts}
    components: List[TopologyComponentInfo] = []
    local_idx: Dict[str, int] = {p: 0 for p in parts}

    for c in range(n_comp):
        if c >= len(comp_part):
            continue
        pidx = comp_part[c]
        if not (0 <= pidx < len(parts)):
            continue
        sem = parts[pidx]
        local_idx[sem] = local_idx.get(sem, 0) + 1
        mask = comp_of_face == c
        fids = np.where(mask)[0].tolist()
        sem_comp_ids.setdefault(sem, []).append(c)
        components.append(
            TopologyComponentInfo(
                semantic=sem,
                index=local_idx[sem],
                component_id=c,
                faces=len(fids),
                face_ids=[int(x) for x in fids],
            )
        )

    component_counts = {p: len(sem_comp_ids.get(p, [])) for p in parts}
    violations: List[Dict[str, Any]] = []

    for c in topology_constraints:
        actual = component_counts.get(c.part, 0)
        if actual < c.count:
            violations.append(
                {
                    "kind": "count_low",
                    "part": c.part,
                    "detail": (
                        f"期望 {c.count} 块，实际 {actual} 块"
                        f" → 考虑 geometric_split 或 reassign_component"
                    ),
                    "expected": c.count,
                    "actual": actual,
                    "hint": "geometric_split | reassign_component",
                }
            )
        elif actual > c.count:
            violations.append(
                {
                    "kind": "count_high",
                    "part": c.part,
                    "detail": (
                        f"期望 {c.count} 块，实际 {actual} 块"
                        f" → 考虑 reassign_component 把多余块改标到正确语义"
                    ),
                    "expected": c.count,
                    "actual": actual,
                    "hint": "reassign_component",
                }
            )

    # adjacency between semantics
    n_faces = int(faces.shape[0])
    adj: Dict[Tuple[int, int], set] = {}
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(n_faces):
        tri = faces[fi]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (min(int(a), int(b)), max(int(a), int(b)))
            edge_to_faces.setdefault(e, []).append(fi)
    for face_list in edge_to_faces.values():
        if len(face_list) < 2:
            continue
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                f1, f2 = face_list[i], face_list[j]
                s1 = int(fpi[f1]) if 0 <= fpi[f1] < len(parts) else -1
                s2 = int(fpi[f2]) if 0 <= fpi[f2] < len(parts) else -1
                if s1 < 0 or s2 < 0 or s1 == s2:
                    continue
                key = (min(s1, s2), max(s1, s2))
                adj.setdefault(key, set()).add((f1, f2))

    adjacency_pairs: List[Tuple[str, str]] = []
    for (i, j) in sorted(adj.keys()):
        adjacency_pairs.append((parts[i], parts[j]))

    # illegal adjacency (inferred from attach_to — e.g. head ↔ arm when head→torso only)
    try:
        from geometry.topology_repair import infer_forbidden_adjacencies

        forbidden_pairs = infer_forbidden_adjacencies(
            topology_constraints, parts
        )
        part_to_idx = {p: i for i, p in enumerate(parts)}
        for a, b in forbidden_pairs:
            ia = part_to_idx.get(a, -1)
            ib = part_to_idx.get(b, -1)
            if ia < 0 or ib < 0:
                continue
            key = (min(ia, ib), max(ia, ib))
            if key in adj:
                violations.append(
                    {
                        "kind": "illegal_adjacency",
                        "part": a,
                        "part_a": a,
                        "part_b": b,
                        "neighbor": b,
                        "detail": (
                            f"非法邻接 {a} ↔ {b}（违反 attach_to 约束）"
                            f" → 考虑 cut_adjacency / spectral+concave seam"
                        ),
                        "hint": "cut_adjacency | boundary_refine",
                    }
                )
    except ImportError:
        pass

    # attachment violations
    for c in topology_constraints:
        if not c.attach_to:
            continue
        pidx = parts.index(c.part) if c.part in parts else -1
        if pidx < 0:
            continue
        attached_to_any = False
        for parent in c.attach_to:
            if parent not in parts:
                continue
            pidx_parent = parts.index(parent)
            if (min(pidx, pidx_parent), max(pidx, pidx_parent)) in adj:
                attached_to_any = True
                break
        if component_counts.get(c.part, 0) > 0 and not attached_to_any:
            violations.append(
                {
                    "kind": "missing_attachment",
                    "part": c.part,
                    "detail": (
                        f"应连接 {c.attach_to}，但实际无共享边界"
                    ),
                    "attach_to": list(c.attach_to),
                }
            )

    return TopologyAudit(
        component_counts=component_counts,
        components=components,
        adjacency_pairs=adjacency_pairs,
        violations=violations,
    )


@dataclass
class TopologyChildAssignment:
    """几何切分后单个子块的语义指派。"""

    index: int  # 1-based
    semantic: str

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "semantic": self.semantic}


@dataclass
class TopologyRepair:
    """单条拓扑修复决策。"""

    action: str
    violation: str = ""
    from_semantic: str = ""
    to_semantic: str = ""
    target_semantic: str = ""
    part: str = ""
    component_index: int = 0
    donor_block_id: str = ""
    seed_count: int = 2
    child_assignments: List[TopologyChildAssignment] = field(default_factory=list)
    reason: str = ""


@dataclass
class TopologyRepairPlan:
    """VLM 拓扑修复方案。"""

    topology_ok: bool = False
    repairs: List[TopologyRepair] = field(default_factory=list)
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_ok": self.topology_ok,
            "repairs": [
                {
                    "violation": r.violation,
                    "action": r.action,
                    "from_semantic": r.from_semantic,
                    "to_semantic": r.to_semantic,
                    "target_semantic": r.target_semantic,
                    "part": r.part,
                    "component_index": r.component_index,
                    "donor_block_id": r.donor_block_id,
                    "seed_count": r.seed_count,
                    "child_assignments": [
                        a.to_dict() for a in r.child_assignments
                    ],
                    "reason": r.reason,
                }
                for r in self.repairs
            ],
            "elapsed_seconds": self.elapsed_seconds,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
        }


def _normalize_topology_action(value: Any) -> str:
    s = str(value or "").strip().lower()
    mapping = {
        "reassign": "reassign_component",
        "reassign_component": "reassign_component",
        "relabel": "reassign_component",
        "merge": "reassign_component",
        "merge_component": "reassign_component",
        "subdivide": "geometric_split",
        "split": "geometric_split",
        "carve": "geometric_split",
        "geometric_split": "geometric_split",
        "cut_adjacency": "cut_adjacency",
        "cut_illegal_adjacency": "cut_adjacency",
        "illegal_adjacency": "cut_adjacency",
        "cut_bridge": "cut_bridge",
        "cut_pseudo_bridge": "cut_bridge",
        "handle_loop": "cut_bridge",
        "boundary_refine": "boundary_refine",
        "graph_cut_refine": "boundary_refine",
        "delete": "reassign_component",
        "delete_part": "reassign_component",
    }
    return mapping.get(s, s)


def _parse_child_assignments(
    raw: Any,
    part_set: set[str],
) -> List[TopologyChildAssignment]:
    if not isinstance(raw, list):
        return []
    out: List[TopologyChildAssignment] = []
    seen: Dict[int, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = _coerce_int(item.get("index"), 0)
        sem = str(item.get("semantic") or "").strip()
        if idx < 1 or not sem or sem not in part_set:
            continue
        if idx in seen:
            continue
        seen[idx] = 1
        out.append(TopologyChildAssignment(index=idx, semantic=sem))
    out.sort(key=lambda a: a.index)
    return out


def _synthesize_child_assignments(
    *,
    seed_count: int,
    part_set: set[str],
    target_semantic: str = "",
    donor_semantic: str = "",
) -> List[TopologyChildAssignment]:
    """旧 subdivide/carve 格式缺 child_assignments 时的兜底合成。"""
    k = max(2, int(seed_count))
    out: List[TopologyChildAssignment] = []
    target = str(target_semantic or "").strip()
    donor = str(donor_semantic or "").strip()
    if k == 2 and target and donor and target != donor:
        if target in part_set:
            out.append(TopologyChildAssignment(index=1, semantic=target))
        if donor in part_set:
            out.append(TopologyChildAssignment(index=2, semantic=donor))
    elif target and target in part_set:
        for i in range(1, k + 1):
            out.append(TopologyChildAssignment(index=i, semantic=target))
    return out


@dataclass
class GeometricSplitAssignment:
    """几何切分后 VLM 对子块的语义指派结果。"""

    assignments: Dict[int, str] = field(default_factory=dict)
    raw_text: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": {
                str(k): v for k, v in sorted(self.assignments.items())
            },
            "error": self.error,
        }


def assign_geometric_split_children(
    client: VLMClient,
    split_panel: str | Path,
    *,
    donor_block_id: str,
    donor_semantic: str,
    child_count: int,
    candidate_semantics: Sequence[str],
    planned_assignments: Sequence[TopologyChildAssignment] = (),
    model: Optional[str] = None,
) -> GeometricSplitAssignment:
    """几何切分后渲染编号拼图，让 VLM 为每个子块指定语义。"""
    candidates = [str(x).strip() for x in candidate_semantics if str(x).strip()]
    if not candidates:
        return GeometricSplitAssignment(error="candidate_semantics 为空")
    prompt = build_geometric_split_assign_prompt(
        donor_block_id=donor_block_id,
        donor_semantic=donor_semantic,
        child_count=child_count,
        candidate_semantics=candidates,
        planned_assignments=[a.to_dict() for a in planned_assignments],
    )
    try:
        response: VLMResponse = client.chat(
            user_prompt=prompt.user,
            image_paths=[Path(split_panel)],
            system=prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
        payload = parse_json_response(response.text or "")
    except (RuntimeError, ValueError) as exc:
        LOGGER.warning("几何切分子块指派失败: %s", exc)
        return GeometricSplitAssignment(error=str(exc))

    cand_set = set(candidates)
    assignments: Dict[int, str] = {}
    raw_items = payload.get("assignments")
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            idx = _coerce_int(item.get("index"), 0)
            sem = str(item.get("semantic") or "").strip()
            if 1 <= idx <= int(child_count) and sem in cand_set:
                assignments[idx] = sem

    # 用预分配填补缺失编号
    for planned in planned_assignments:
        if planned.index not in assignments and planned.semantic in cand_set:
            assignments[planned.index] = planned.semantic

    # 仍缺失则按候选轮转
    for idx in range(1, int(child_count) + 1):
        if idx not in assignments:
            assignments[idx] = candidates[(idx - 1) % len(candidates)]

    return GeometricSplitAssignment(
        assignments=assignments,
        raw_text=response.text or "",
    )


def resolve_topology_repairs(
    client: VLMClient,
    coverage_panel: str | Path,
    *,
    semantic_parts: Sequence[str],
    topology_constraints: Sequence[TopologyConstraint],
    audit: TopologyAudit,
    donor_blocks: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    max_split: int = 8,
) -> TopologyRepairPlan:
    """让 VLM 根据渲染图与拓扑审计结果给出修复方案。"""
    component_details = [
        {
            "semantic": c.semantic,
            "index": c.index,
            "faces": c.faces,
        }
        for c in audit.components
    ]
    donor_by_id = {
        str(b.get("block_id", "")).strip(): b for b in donor_blocks
    }
    prompt = build_topology_repair_prompt(
        topology_constraints=[c.to_dict() for c in topology_constraints],
        component_counts=audit.component_counts,
        component_details=component_details,
        adjacency_pairs=audit.adjacency_pairs,
        violations=audit.violations,
        donor_blocks=list(donor_blocks),
        max_split=max_split,
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
        LOGGER.warning("VLM 拓扑修复调用失败: %s", exc)
        return TopologyRepairPlan(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        return TopologyRepairPlan(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            raw_response=response.raw,
        )

    part_set = {str(p).strip() for p in semantic_parts}
    donor_ids = {
        str(b.get("block_id", "")).strip()
        for b in donor_blocks
        if str(b.get("block_id", "")).strip()
    }
    raw_repairs = payload.get("repairs")
    repairs: List[TopologyRepair] = []
    if isinstance(raw_repairs, list):
        for item in raw_repairs:
            if not isinstance(item, dict):
                continue
            action = _normalize_topology_action(item.get("action"))
            if action not in (
                "reassign_component",
                "geometric_split",
                "cut_adjacency",
                "cut_bridge",
                "boundary_refine",
            ):
                continue
            from_sem = str(
                item.get("from_semantic")
                or item.get("part_a")
                or item.get("part")
                or ""
            ).strip()
            to_sem = str(
                item.get("to_semantic")
                or item.get("part_b")
                or item.get("neighbor")
                or ""
            ).strip()
            rec = TopologyRepair(
                action=action,
                violation=str(item.get("violation") or "").strip(),
                from_semantic=from_sem,
                to_semantic=to_sem,
                target_semantic=str(
                    item.get("target_semantic")
                    or item.get("empty_semantic")
                    or ""
                ).strip(),
                part=str(item.get("part") or "").strip(),
                component_index=_coerce_int(item.get("component_index"), 0),
                donor_block_id=str(item.get("donor_block_id") or "").strip(),
                seed_count=max(2, _coerce_int(item.get("seed_count"), 2)),
                reason=str(item.get("reason") or "").strip(),
            )
            if action == "geometric_split":
                if rec.donor_block_id not in donor_ids:
                    continue
                rec.seed_count = min(rec.seed_count, max(2, int(max_split)))
                rec.child_assignments = _parse_child_assignments(
                    item.get("child_assignments"), part_set
                )
                if not rec.child_assignments:
                    donor_meta = donor_by_id.get(rec.donor_block_id) or {}
                    donor_sem = str(donor_meta.get("semantic") or "").strip()
                    rec.child_assignments = _synthesize_child_assignments(
                        seed_count=rec.seed_count,
                        part_set=part_set,
                        target_semantic=rec.target_semantic,
                        donor_semantic=donor_sem,
                    )
                if not rec.child_assignments:
                    continue
                if len(rec.child_assignments) != rec.seed_count:
                    rec.seed_count = len(rec.child_assignments)
            elif action == "reassign_component":
                if rec.from_semantic not in part_set:
                    continue
                if rec.to_semantic not in part_set:
                    continue
                if rec.component_index < 1:
                    continue
            elif action in ("cut_adjacency", "cut_bridge", "boundary_refine"):
                if rec.from_semantic not in part_set:
                    continue
                if rec.to_semantic not in part_set:
                    continue
            repairs.append(rec)

    return TopologyRepairPlan(
        topology_ok=_coerce_bool(payload.get("topology_ok")),
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
    "GeometricSplitAssignment",
    "MultiLabelClassification",
    "SemanticEnumeration",
    "TopologyAudit",
    "TopologyChildAssignment",
    "TopologyComponentInfo",
    "TopologyConstraint",
    "TopologyRepair",
    "TopologyRepairPlan",
    "assign_geometric_split_children",
    "audit_topology",
    "classify_multi_label",
    "classify_multi_label_coverage_repair",
    "enumerate_semantics",
    "infer_topology_constraints",
    "resolve_empty_semantic_repairs",
    "resolve_topology_repairs",
]
